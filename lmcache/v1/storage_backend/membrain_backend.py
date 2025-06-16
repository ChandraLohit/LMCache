# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
import asyncio
import json
import mmap
import os
import threading
import time

# Third Party
import aiohttp
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj, MemoryFormat
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)


@dataclass
class LeaseInfo:
    """Information about a lease obtained from Membrain daemon."""
    lease_id: str
    offsets: List[Tuple[int, int]]  # (offset, length) pairs
    total_size: int


@dataclass
class MembrainCacheMetadata:
    """Metadata for cached items in Membrain."""
    key: CacheEngineKey
    bucket: str
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat = MemoryFormat.UNDEFINED
    lease_info: Optional[LeaseInfo] = None


class MembrainBackend(StorageBackendInterface):
    """
    A storage backend that uses Membrain KV cache daemon for layerwise caching.
    
    This backend is designed for layerwise mode operations and provides:
    - Direct shared memory access via leases (no local_cpu_backend buffer)
    - HTTP API integration with Membrain daemon
    - Efficient batch operations for layer-by-layer processing
    - Memory-mapped file access for zero-copy operations
    
    Configuration requires:
    - membrain_url: URL of the Membrain daemon (e.g., "http://localhost:9200")
    - shared_memory_name: Optional name for shared memory segment
    - bucket_name: Bucket name for priority/organization (default: "lmcache")
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device)
        
        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.dst_device = dst_device

        # Membrain configuration
        self.membrain_url = getattr(config, 'membrain_url', 'http://localhost:9200')
        self.shared_memory_name = getattr(config, 'shared_memory_name', None)
        self.bucket_name = getattr(config, 'membrain_bucket', 'lmcache')
        self.timeout_ms = getattr(config, 'membrain_timeout_ms', 5000)

        # Cache management
        self.cache_lock = threading.Lock()
        self.cache_metadata: OrderedDict[CacheEngineKey, MembrainCacheMetadata] = OrderedDict()
        
        # Put task tracking
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()
        
        # Lease management
        self.lease_lock = threading.Lock()
        self.active_leases: Dict[str, LeaseInfo] = {}
        
        # Shared memory mapping (will be initialized when needed)
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        self.shared_memory_lock = threading.Lock()

        logger.info(
            f"MembrainBackend initialized with URL: {self.membrain_url}, "
            f"bucket: {self.bucket_name}, shared_memory: {self.shared_memory_name}"
        )

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in Membrain cache."""
        logger.info(f"MembrainBackend: Checking contains() for key {key}")
        
        # First check local cache metadata
        with self.cache_lock:
            if key in self.cache_metadata:
                logger.info(f"MembrainBackend: Key {key} found in local metadata cache")
                return True
        
        # Check with Membrain daemon via HTTP
        logger.info(f"MembrainBackend: Key {key} not in local cache, checking Membrain daemon")
        result = asyncio.run_coroutine_threadsafe(
            self._async_contains(key, pin), self.loop
        ).result()
        logger.info(f"MembrainBackend: Membrain daemon contains() result for key {key}: {result}")
        return result

    async def _async_contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Async check if key exists in Membrain."""
        try:
            key_str = self._key_to_string(key)
            url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}/locations"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as response:
                    if response.status == 200:
                        locations_data = await response.json()
                        # Cache the metadata for future use
                        await self._cache_key_metadata(key, locations_data)
                        return True
                    elif response.status == 404:
                        return False
                    else:
                        logger.warning(f"Unexpected response {response.status} for key {key}")
                        return False
        except Exception as e:
            logger.debug(f"Failed to check key existence: {e}")
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key is currently being stored."""
        with self.put_lock:
            return key in self.put_tasks

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        """Submit batch of PUT tasks to Membrain."""
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(key, memory_obj)
            if future is not None:
                futures.append(future)
        return futures if futures else None

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """Submit a single PUT task to Membrain."""
        memory_obj.ref_count_up()
        
        with self.put_lock:
            self.put_tasks.add(key)
        
        future = asyncio.run_coroutine_threadsafe(
            self._async_put(key, memory_obj), self.loop
        )
        return future

    async def _async_put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Async PUT operation to Membrain daemon."""
        try:
            key_str = self._key_to_string(key)
            url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}"
            logger.info(f"MembrainBackend: Starting PUT operation for key {key} to URL {url}")
            
            # Convert memory object to bytes
            data = self._memory_obj_to_bytes(memory_obj)
            logger.info(f"MembrainBackend: Serialized {len(data)} bytes for key {key}")
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    url, 
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_ms/1000.0)
                ) as response:
                    logger.info(f"MembrainBackend: PUT HTTP response for key {key}: status={response.status}")
                    if response.status == 200:
                        # Store metadata locally
                        metadata = MembrainCacheMetadata(
                            key=key,
                            bucket=self.bucket_name,
                            size=memory_obj.get_size(),
                            shape=memory_obj.get_shape(),
                            dtype=memory_obj.get_dtype() or torch.float16,
                            fmt=memory_obj.get_memory_format()
                        )
                        with self.cache_lock:
                            self.cache_metadata[key] = metadata
                        logger.info(f"MembrainBackend: Successfully stored key {key}")
                    else:
                        logger.error(f"MembrainBackend: Failed to store key {key}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"MembrainBackend: CRITICAL - Exception during PUT for key {key}: {type(e).__name__}: {e}")
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """Submit prefetch task (acquire lease without immediate data transfer)."""
        return asyncio.run_coroutine_threadsafe(
            self._async_prefetch(key), self.loop
        )

    async def _async_prefetch(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Async prefetch - acquire lease for the key."""
        lease_info = await self._acquire_lease(key)
        if lease_info is None:
            return None
        
        # For prefetch, we don't immediately load data, just acquire the lease
        with self.cache_lock:
            if key in self.cache_metadata:
                self.cache_metadata[key].lease_info = lease_info
        
        return await self._create_memory_obj_from_lease(key, lease_info)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking GET operation from Membrain."""
        logger.info(f"MembrainBackend: Starting GET operation for key {key}")
        try:
            result = asyncio.run_coroutine_threadsafe(
                self._async_get_blocking(key), self.loop
            ).result()
            if result is not None:
                logger.info(f"MembrainBackend: GET operation successful for key {key}")
            else:
                logger.warning(f"MembrainBackend: GET operation failed for key {key}")
            return result
        except Exception as e:
            logger.error(f"MembrainBackend: GET operation exception for key {key}: {e}")
            return None

    async def _async_get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Async blocking GET operation."""
        logger.info(f"MembrainBackend: Acquiring lease for key {key}")
        # First acquire lease
        lease_info = await self._acquire_lease(key)
        if lease_info is None:
            logger.warning(f"MembrainBackend: Failed to acquire lease for key {key}")
            return None
        
        logger.info(f"MembrainBackend: Lease acquired {lease_info.lease_id}, creating memory object")
        try:
            result = await self._create_memory_obj_from_lease(key, lease_info)
            if result is not None:
                logger.info(f"MembrainBackend: Memory object created successfully for key {key}")
            return result
        except Exception as e:
            logger.error(f"MembrainBackend: Failed to create memory object for key {key}: {e}")
            # Release lease on failure
            await self._release_lease(lease_info.lease_id)
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Non-blocking GET operation."""
        return self.submit_prefetch_task(key)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Acquire a lease for the given key from Membrain daemon."""
        try:
            key_str = self._key_to_string(key)
            url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
            
            params = {"timeout_ms": self.timeout_ms}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_ms/1000.0)
                ) as response:
                    if response.status == 200:
                        lease_data = await response.json()
                        lease_info = LeaseInfo(
                            lease_id=lease_data["id"],
                            offsets=[(o["offset"], o["len"]) for o in lease_data["offsets"]],
                            total_size=sum(o["len"] for o in lease_data["offsets"])
                        )
                        
                        with self.lease_lock:
                            self.active_leases[lease_info.lease_id] = lease_info
                        
                        logger.info(f"MembrainBackend: Acquired lease {lease_info.lease_id} for key {key}, total_size={lease_info.total_size}, offsets={len(lease_info.offsets)}")
                        return lease_info
                    elif response.status == 404:
                        logger.debug(f"Key {key} not found for lease acquisition")
                        return None
                    else:
                        logger.error(f"Failed to acquire lease for key {key}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"MembrainBackend: CRITICAL - Exception during lease acquisition for key {key}: {type(e).__name__}: {e}")
            return None

    async def _release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        try:
            url = f"{self.membrain_url}/v1/leases/{lease_id}/release"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    timeout=aiohttp.ClientTimeout(total=2.0)
                ) as response:
                    success = response.status == 200
                    if success:
                        with self.lease_lock:
                            self.active_leases.pop(lease_id, None)
                        logger.debug(f"Released lease {lease_id}")
                    else:
                        logger.error(f"Failed to release lease {lease_id}: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Error releasing lease {lease_id}: {e}")
            return False

    async def _create_memory_obj_from_lease(self, key: CacheEngineKey, lease_info: LeaseInfo) -> Optional[MemoryObj]:
        """Create a MemoryObj by reading data from shared memory using lease offsets.
        
        Reconstructs tensor from serialized format: [4 bytes metadata size][metadata json][tensor bytes]
        This ensures PUT/GET consistency by properly deserializing the stored data.
        """
        # Initialize shared memory if needed
        if not await self._ensure_shared_memory():
            logger.error("Failed to initialize shared memory")
            return None
        
        try:
            # Read all data from shared memory offsets
            total_data = bytearray()
            for offset, length in lease_info.offsets:
                if self.shared_memory_map is not None:
                    chunk = bytes(self.shared_memory_map[offset:offset + length])
                    total_data.extend(chunk)
                    logger.debug(f"Read {length} bytes from offset {offset}")
            
            # Parse metadata header (first 4 bytes contain metadata size)
            if len(total_data) < 4:
                logger.error(f"Insufficient data to read metadata header for key {key}")
                return None
            
            # Extract metadata size
            metadata_size = int.from_bytes(total_data[:4], 'little')
            
            if len(total_data) < 4 + metadata_size:
                logger.error(f"Insufficient data to read metadata for key {key}")
                return None
            
            # Extract and parse metadata JSON
            metadata_json = total_data[4:4 + metadata_size].decode('utf-8')
            tensor_metadata = json.loads(metadata_json)
            
            # Extract tensor data
            tensor_bytes = bytes(total_data[4 + metadata_size:])
            expected_size = tensor_metadata['tensor_size']
            
            if len(tensor_bytes) != expected_size:
                logger.error(f"Tensor size mismatch for key {key}: expected {expected_size}, got {len(tensor_bytes)}")
                return None
            
            # Reconstruct tensor from metadata
            shape = torch.Size(tensor_metadata['shape'])
            original_dtype_str = tensor_metadata['original_dtype']
            serialized_dtype_str = tensor_metadata['serialized_dtype']
            memory_format = MemoryFormat(tensor_metadata['format'])
            
            # Parse dtype strings
            original_dtype = getattr(torch, original_dtype_str.replace('torch.', ''))
            serialized_dtype = getattr(torch, serialized_dtype_str.replace('torch.', ''))
            
            # Reconstruct tensor from bytes
            import numpy as np
            
            # Convert bytes to numpy array with serialized dtype
            if serialized_dtype == torch.float32:
                numpy_dtype = np.float32
            elif serialized_dtype == torch.float16:
                numpy_dtype = np.float16
            elif serialized_dtype == torch.bfloat16:
                numpy_dtype = np.float32  # bfloat16 is stored as float32
            else:
                # Handle other dtypes as needed
                numpy_dtype = np.float32
                logger.warning(f"Unknown serialized dtype {serialized_dtype}, using float32")
            
            numpy_array = np.frombuffer(tensor_bytes, dtype=numpy_dtype)
            reconstructed_tensor = torch.from_numpy(numpy_array.copy()).reshape(shape)
            
            # Convert back to original dtype if needed
            if original_dtype != serialized_dtype:
                reconstructed_tensor = reconstructed_tensor.to(original_dtype)
            
            # Allocate memory object with correct format
            memory_obj = self.memory_allocator.allocate(shape, original_dtype, memory_format)
            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key}")
                return None
            
            # Copy reconstructed tensor to allocated memory
            if memory_obj.tensor is not None:
                # Move tensor to target device and copy
                target_tensor = reconstructed_tensor.to(memory_obj.tensor.device)
                memory_obj.tensor.copy_(target_tensor)
                
                logger.info(f"MembrainBackend: Successfully reconstructed tensor for key {key}: shape={shape}, dtype={original_dtype}, format={memory_format}")
                return memory_obj
            else:
                logger.error(f"Allocated memory object has no tensor for key {key}")
                memory_obj.ref_count_down()
                return None
                
        except Exception as e:
            logger.error(f"Error reading data from shared memory for key {key}: {e}")
            return None

    async def _ensure_shared_memory(self) -> bool:
        """Ensure shared memory is initialized and accessible."""
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                return True
            
            if self.shared_memory_name is None:
                logger.error("No shared memory name configured")
                return False
            
            try:
                # Try to open existing shared memory segment created by Membrain daemon
                self.shared_memory_obj = shared_memory.SharedMemory(
                    name=self.shared_memory_name, create=False
                )
                self.shared_memory_map = memoryview(self.shared_memory_obj.buf)
                
                logger.info(f"MembrainBackend: Successfully opened shared memory: {self.shared_memory_name} (size: {len(self.shared_memory_map)} bytes)")
                return True
                
            except FileNotFoundError:
                logger.error(f"MembrainBackend: CRITICAL - Shared memory segment '{self.shared_memory_name}' not found. Is Membrain daemon running and creating shared memory?")
                return False
            except Exception as e:
                logger.error(f"MembrainBackend: CRITICAL - Failed to initialize shared memory: {e}")
                return False

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin operation - not implemented for Membrain."""
        logger.warning("Pin operation not supported by MembrainBackend")
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin operation - not implemented for Membrain."""
        logger.warning("Unpin operation not supported by MembrainBackend")
        return True

    def close(self) -> None:
        """Close the backend and release resources."""
        # Release all active leases
        if self.active_leases:
            logger.info(f"Releasing {len(self.active_leases)} active leases")
            for lease_id in list(self.active_leases.keys()):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._release_lease(lease_id), self.loop
                    ).result(timeout=5.0)
                except Exception as e:
                    logger.error(f"Error releasing lease {lease_id}: {e}")
        
        # Close shared memory safely
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                try:
                    self.shared_memory_map.release()
                except Exception as e:
                    logger.error(f"Error releasing shared memory map: {e}")
                self.shared_memory_map = None
                
            if self.shared_memory_obj is not None:
                try:
                    self.shared_memory_obj.close()
                except Exception as e:
                    logger.error(f"Error closing shared memory: {e}")
                self.shared_memory_obj = None
        
        logger.info("MembrainBackend closed.")

    # Helper methods

    def _key_to_string(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to string format for HTTP API.
        
        Use URL encoding for complete safety instead of character replacement.
        This avoids conflicts with existing underscores in keys.
        """
        import urllib.parse
        key_str = key.to_string()
        # URL encode the entire key to handle all special characters safely
        return urllib.parse.quote(key_str, safe="")

    async def _cache_key_metadata(self, key: CacheEngineKey, locations_data: dict) -> None:
        """Cache metadata from locations response."""
        if "locations" in locations_data and locations_data["locations"]:
            location = locations_data["locations"][0]
            # Extract size and other metadata if available
            size = location.get("length", 0)
            
            # Create basic metadata - in real implementation you'd need more info
            metadata = MembrainCacheMetadata(
                key=key,
                bucket=self.bucket_name,
                size=size,
                shape=torch.Size([size]),  # Placeholder
                dtype=torch.uint8,  # Placeholder
                fmt=MemoryFormat.UNDEFINED  # Will be updated during PUT
            )
            
            with self.cache_lock:
                self.cache_metadata[key] = metadata

    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        """Convert MemoryObj to bytes for HTTP transmission with metadata header.
        
        Format: [4 bytes metadata size][metadata json][tensor bytes]
        This ensures PUT/GET consistency by preserving all tensor metadata.
        """
        tensor = memory_obj.tensor
        if tensor is None:
            return b""
        
        # Store original properties before conversion
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_format = memory_obj.get_memory_format()
        
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Handle BFloat16 which can't convert to numpy directly
        # Following GDS pattern for dtype handling
        serialized_dtype = original_dtype
        if tensor.dtype == torch.bfloat16:
            # Convert BFloat16 to Float32 for numpy compatibility
            tensor = tensor.to(torch.float32)
            serialized_dtype = torch.float32
        elif tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2:
            # Handle other unsupported dtypes
            tensor = tensor.to(torch.float32)
            serialized_dtype = torch.float32
        
        try:
            tensor_bytes = tensor.numpy().tobytes()
        except Exception as e:
            logger.error(f"Failed to convert tensor to bytes, dtype={tensor.dtype}: {e}")
            # Fallback: convert to float32 and try again
            tensor = tensor.to(torch.float32)
            serialized_dtype = torch.float32
            tensor_bytes = tensor.numpy().tobytes()
        
        # Create metadata header for proper reconstruction
        metadata_dict = {
            'shape': list(original_shape),
            'original_dtype': str(original_dtype),
            'serialized_dtype': str(serialized_dtype),
            'format': original_format.value,
            'tensor_size': len(tensor_bytes)
        }
        
        # Serialize metadata as JSON bytes
        metadata_json = json.dumps(metadata_dict).encode('utf-8')
        metadata_size = len(metadata_json)
        
        # Format: [4 bytes metadata size][metadata json][tensor bytes]
        result = metadata_size.to_bytes(4, 'little') + metadata_json + tensor_bytes
        
        logger.info(f"MembrainBackend: Serialized tensor: shape={original_shape}, dtype={original_dtype}, format={original_format}, size={len(result)} bytes")
        return result