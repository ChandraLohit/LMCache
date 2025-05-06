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

import asyncio
import hashlib
import base64
from typing import List, Optional, no_type_check, Dict

from lmcache.experimental.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.experimental.protocol import RedisMetadata  # Reusing Redis metadata format
from lmcache.experimental.storage_backend.connector.base_connector import RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

# Import the Membrain client
from lmcache.clients.membrain_client import MembrainClient, MembrainConfig, MembrainError, MembrainKeyError

logger = init_logger(__name__)


class MembrainConnector(RemoteConnector):
    """
    Connector for Membrain key-value store in the experimental package.
    The remote url should start with "membrain://" and include a host and port.
    """

    def __init__(self, endpoint: str, namespace: str,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        """Initialize the Membrain connector.
        
        Args:
            endpoint: The Membrain endpoint URL (e.g., "http://localhost:9201")
            namespace: The namespace to use in Membrain
            loop: The event loop to use for async operations
            memory_allocator: The memory allocator for MemoryObj handling
        """
        self.config = MembrainConfig(
            endpoint=endpoint,
            namespace=namespace,
            timeout=30.0  # Reasonable default timeout
        )
        self.client = MembrainClient(self.config)
        self.memory_allocator = memory_allocator
        self.loop = loop
        # Keep a key mapping cache to be able to track original keys
        self._key_mapping: Dict[str, str] = {}
        logger.info(f"Initialized experimental Membrain connector with endpoint {endpoint}, namespace {namespace}")
        
    def _hash_key(self, key_str: str) -> str:
        """
        Hash the long key into a shorter, URL-safe string.
        Store the original->hashed mapping for debugging.
        
        Args:
            key_str: The original long key string
            
        Returns:
            A URL-safe hashed key string (base64 of SHA-256 hash)
        """
        # Create a hash of the key
        key_hash = hashlib.sha256(key_str.encode()).digest()
        # Convert to URL-safe base64 and remove padding
        safe_key = base64.urlsafe_b64encode(key_hash).decode().rstrip('=')
        # Store mapping for debug and reference
        self._key_mapping[key_str] = safe_key
        logger.debug(f"Hashed key: {key_str} -> {safe_key}")
        return safe_key

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if the key exists in Membrain using hashed keys."""
        try:
            original_key = key.to_string()
            # Use hashed keys for Membrain
            hashed_key = self._hash_key(original_key)
            metadata_key = f"{hashed_key}_meta"
            kv_bytes_key = f"{hashed_key}_data"
            
            # Check metadata key first
            metadata_exists = await self.client.exists(metadata_key)
            if not metadata_exists:
                return False
                
            # Then check KV bytes key
            kv_bytes_exists = await self.client.exists(kv_bytes_key)
            logger.debug(f"Key existence check for {original_key} (hash: {hashed_key}): metadata={metadata_exists}, data={kv_bytes_exists}")
            
            # Both keys must exist for a valid cache entry
            return metadata_exists and kv_bytes_exists
                
        except Exception as e:
            logger.error(f"Error checking existence for key {key.to_string()}: {e}")
            return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get and deserialize data from Membrain using hashed keys."""
        try:
            original_key = key.to_string()
            # Use hashed keys for Membrain
            hashed_key = self._hash_key(original_key)
            metadata_key = f"{hashed_key}_meta"
            kv_bytes_key = f"{hashed_key}_data"
            
            logger.debug(f"Getting key {original_key} (hash: {hashed_key})")
            
            # Get metadata first
            try:
                metadata_bytes = await self.client.get(metadata_key)
                if not metadata_bytes:
                    logger.debug(f"No metadata found for {hashed_key}")
                    return None
                    
                # Deserialize metadata
                redis_metadata = RedisMetadata.deserialize(memoryview(metadata_bytes))
                
                # Allocate memory object
                memory_obj = self.memory_allocator.allocate(
                    redis_metadata.shape,
                    redis_metadata.dtype,
                    redis_metadata.fmt,
                )
                
                if memory_obj is None:
                    logger.warning(f"Failed to allocate memory for key: {original_key}")
                    return None
    
                # Get actual KV cache data
                kv_bytes = await self.client.get(kv_bytes_key)
                
                if kv_bytes is None:
                    logger.warning(f"KV cache data missing for key: {original_key}")
                    return None
    
                # Copy data into memory object
                view = memoryview(memory_obj.byte_array)
                view[:redis_metadata.length] = kv_bytes
                logger.debug(f"Retrieved {redis_metadata.length} bytes for key: {original_key}")
                
                return memory_obj
                
            except Exception as e:
                logger.error(f"Error retrieving key {original_key}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error getting key {key.to_string()}: {e}")
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data in Membrain with metadata using hashed keys."""
        try:
            original_key = key.to_string()
            # Use hashed keys for Membrain
            hashed_key = self._hash_key(original_key)
            metadata_key = f"{hashed_key}_meta"
            kv_bytes_key = f"{hashed_key}_data"
            
            logger.debug(f"Storing key {original_key} (hash: {hashed_key})")
            
            # Extract metadata from memory object
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()

            # Create and serialize metadata
            redis_metadata_bytes = RedisMetadata(
                len(kv_bytes), kv_shape, kv_dtype, memory_format).serialize()

            # Store metadata
            try:
                await self.client.put(metadata_key, redis_metadata_bytes)
                logger.debug(f"Stored metadata for key: {original_key}")
            except Exception as e:
                logger.error(f"Error storing metadata for key {original_key}: {e}")
                return
                
            # Store KV bytes
            try:
                await self.client.put(kv_bytes_key, kv_bytes)
                logger.debug(f"Stored {len(kv_bytes)} bytes for key: {original_key}")
            except Exception as e:
                logger.error(f"Error storing KV bytes for key {original_key}: {e}")
                return

            # Decrease reference count
            self.memory_allocator.ref_count_down(memory_obj)
            
        except Exception as e:
            logger.error(f"Error putting key {key.to_string()}: {e}")

    @no_type_check
    async def list(self) -> List[str]:
        """List keys in Membrain (not implemented)."""
        logger.warning("List operation not supported by Membrain client")
        return []

    async def close(self):
        """Close the Membrain client."""
        try:
            await self.client.close()
            logger.info("Closed experimental Membrain connector")
        except Exception as e:
            logger.error(f"Error closing Membrain connector: {e}")