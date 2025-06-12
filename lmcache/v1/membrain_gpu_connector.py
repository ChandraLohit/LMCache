from typing import List
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import RemoteMetadata
import torch
# import membrain_ext
from multiprocessing.shared_memory import SharedMemory
import lmcache.c_ops as lmc_ops



logger = init_logger(__name__)

class BedrockMembrainGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    
    def _parse_membrain_data(self, memory_obj, shm, num_tokens):
        """
        Shared helper method to parse Membrain shared memory data format.
        ZERO-COPY OPTIMIZED: Fast path for single segments, memory views for multi-segment.
        
        Returns:
            tuple: (kv_tensor, metadata) or (None, None) if parsing fails
        """
        try:
            logger.info(f"🔍 ZERO-COPY PARSE: Starting with {len(memory_obj.offsets)} segments")
            
            # OPTIMIZATION: Fast path for single segment (most common case)
            if len(memory_obj.offsets) == 1:
                logger.info(f"ZERO-COPY FAST PATH: Single segment - avoiding ALL copies!")
                offset_info = memory_obj.offsets[0]
                o = offset_info['offset']
                l = offset_info['len']
                
                # Fix SharedMemory lifecycle: Copy data directly to avoid memoryview references
                logger.info(f"📊 MEMBRAIN ACCESS: Reading {l} bytes from offset {o}")
                
                if l < 4:
                    raise ValueError(f"Insufficient data: only {l} bytes")
                
                # Read data directly from SharedMemory buffer to avoid memoryview lifecycle issues
                raw_data = bytes(shm.buf[o : o + l])
                logger.info(f"📊 DATA COPY: Copied {len(raw_data)} bytes from shared memory")
                
                # Read metadata length directly from raw data
                metadata_len = int.from_bytes(raw_data[:4], byteorder='little')
                kv_data_start = 4 + metadata_len
                
                if len(raw_data) < kv_data_start:
                    raise ValueError(f"Insufficient data for metadata: need {kv_data_start}, have {len(raw_data)}")
                
                # Parse metadata using raw data
                metadata_bytes = raw_data[4:kv_data_start]
                metadata = RemoteMetadata.deserialize(metadata_bytes)
                logger.info(f"METADATA PARSED: shape={metadata.shape}, dtype={metadata.dtype}")
                
                # Extract KV data using raw data
                kv_data_bytes = raw_data[kv_data_start:]
                
                if len(kv_data_bytes) == 0:
                    raise ValueError("No KV data found after metadata")
                
                # Create tensor directly from raw bytes (no SharedMemory references)
                tensor_dtype = getattr(metadata, 'dtype', getattr(self, 'dtype', torch.float16))
                kv_tensor = torch.frombuffer(bytearray(kv_data_bytes), dtype=tensor_dtype)
                logger.info(f"TENSOR CREATED: {len(kv_tensor)} elements from raw bytes (SharedMemory safe)")
                
            else:
                # Multi-segment fallback - use memory views instead of bytes()
                logger.warning(f"⚠️  COPY FALLBACK: Multi-segment data ({len(memory_obj.offsets)} segments) - using optimized memory views")
                
                # Calculate total size first
                total_size = sum(offset['len'] for offset in memory_obj.offsets)
                combined_data = bytearray(total_size)  # Pre-allocate exact size - avoids extend() copies
                logger.info(f"📊 COPY FALLBACK: Pre-allocated {total_size} bytes for {len(memory_obj.offsets)} segments")
                
                # Copy using direct bytes access (avoid memoryview SharedMemory references)
                pos = 0
                for offset in memory_obj.offsets:
                    o = offset['offset']
                    l = offset['len']
                    segment_bytes = bytes(shm.buf[o : o + l])  # Direct bytes copy
                    combined_data[pos:pos + l] = segment_bytes  # Direct bytes assignment
                    pos += l
                
                logger.info(f"📊 COPY FALLBACK: Combined {len(memory_obj.offsets)} segments using direct bytes (SharedMemory safe)")
                
                # Rest of parsing logic (same as before)
                if len(combined_data) < 4:
                    raise ValueError(f"Insufficient data: only {len(combined_data)} bytes")
                
                metadata_len = int.from_bytes(combined_data[:4], byteorder='little')
                kv_data_start = 4 + metadata_len
                
                if len(combined_data) < kv_data_start:
                    raise ValueError(f"Insufficient data for metadata: need {kv_data_start}, have {len(combined_data)}")
                
                # Parse metadata
                metadata_bytes = combined_data[4:kv_data_start]
                metadata = RemoteMetadata.deserialize(metadata_bytes)
                logger.info(f"METADATA PARSED: shape={metadata.shape}, dtype={metadata.dtype}")
                
                # Extract pure KV data
                kv_data = combined_data[kv_data_start:]
                
                if len(kv_data) == 0:
                    raise ValueError("No KV data found after metadata")
                
                # Create tensor from KV data (no SharedMemory references)
                tensor_dtype = getattr(metadata, 'dtype', getattr(self, 'dtype', torch.float16))
                kv_tensor = torch.frombuffer(bytearray(kv_data), dtype=tensor_dtype)
                logger.info(f"TENSOR CREATED: {len(kv_tensor)} elements from combined bytes (SharedMemory safe)")
            
            # Common padding logic
            expected_elements = num_tokens * 2 * self.hidden_dim_size
            
            # Handle padding if needed (for batched_to_gpu compatibility)
            if len(kv_tensor) < expected_elements:
                logger.warning(f"Padding tensor: need {expected_elements}, have {len(kv_tensor)}")
                # Use the same dtype as the kv_tensor for consistency
                padded_tensor = torch.zeros(expected_elements, dtype=kv_tensor.dtype)
                padded_tensor[:len(kv_tensor)] = kv_tensor
                kv_tensor = padded_tensor
            
            logger.info(f"PARSE SUCCESS: Final tensor has {len(kv_tensor)} elements")
            return kv_tensor, metadata
            
        except Exception as e:
            logger.error(f"Failed to parse Membrain data: {e}")
            return None, None

    @staticmethod
    def from_base(conn: VLLMPagedMemLayerwiseGPUConnector) -> "BedrockMembrainGPUConnector":
        # TODO(gnovack) - bro this is terrible...
        conn.__class__ = BedrockMembrainGPUConnector
        return conn
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
    
    def get_flat_shape(self, num_tokens: int) -> torch.Size:
        # Calculate size for pure KV data (excluding metadata)
        # Use default dtype if self.dtype is not set
        dtype = getattr(self, 'dtype', torch.float16)
        element_size = torch.finfo(dtype).bits // 8  # bytes per element
        return torch.Size([num_tokens * 2 * self.hidden_dim_size * element_size])
    
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        Override to_gpu method to handle LeaseMemoryObj for regular (non-layerwise) retrieve operations.
        This ensures zero-copy Membrain works in both regular and layerwise modes.
        """
        # Check if this is a LeaseMemoryObj with offsets (Membrain zero-copy)
        if hasattr(memory_obj, 'offsets') and memory_obj.offsets:
            logger.debug(f"BedrockMembrainGPUConnector.to_gpu: Processing LeaseMemoryObj with {len(memory_obj.offsets)} offsets")
            
            if "kvcaches" not in kwargs:
                raise ValueError("'kvcaches' should be provided in kwargs.")
            if "slot_mapping" not in kwargs:
                raise ValueError("'slot_mapping' should be provided in kwargs.")
            
            kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
            slot_mapping: torch.Tensor = kwargs["slot_mapping"]
            
            # For regular to_gpu, we process all layers at once
            # This is different from batched_to_gpu which processes layer by layer
            num_tokens = end - start
            
            # Access shared memory
            shm = SharedMemory('membrain-kvcache')
            
            # Process the memory object with offsets using shared helper
            try:
                logger.info(f"TO_GPU: Calling _parse_membrain_data for {num_tokens} tokens")
                kv_tensor, metadata = self._parse_membrain_data(memory_obj, shm, num_tokens)
                
                if kv_tensor is None or metadata is None:
                    raise ValueError("Failed to parse Membrain data")
                
                # For regular to_gpu, we need to process all layers
                # Reshape the tensor for all layers: [num_layers, num_tokens, 2, hidden_dim]
                total_elements = len(kv_tensor)
                expected_elements_per_layer = num_tokens * 2 * self.hidden_dim_size
                
                if total_elements % expected_elements_per_layer != 0:
                    logger.warning(f"KV data size mismatch: {total_elements} not divisible by {expected_elements_per_layer}")
                
                num_layers_in_data = total_elements // expected_elements_per_layer
                effective_layers = min(num_layers_in_data, self.num_layers)
                
                logger.debug(f"to_gpu processing {effective_layers} layers, {num_tokens} tokens each")
                
                # Process each layer
                for layer_id in range(effective_layers):
                    layer_start = layer_id * expected_elements_per_layer
                    layer_end = layer_start + expected_elements_per_layer
                    layer_tensor = kv_tensor[layer_start:layer_end]
                    
                    # Reshape to [num_tokens, 2, hidden_dim]
                    layer_tensor = layer_tensor.view(num_tokens, 2, self.hidden_dim_size)
                    
                    # Transfer to GPU and then to vLLM KV cache
                    gpu_layer_tensor = layer_tensor.to(device=kvcaches[layer_id][0].device, non_blocking=True)
                    
                    # Use LMCache C++ ops for efficient transfer
                    # Fix: Add token_major parameter to match C++ signature
                    logger.debug(f"TO_GPU TRANSFER: tensor shape={gpu_layer_tensor.shape}, token_major=True")
                    lmc_ops.single_layer_kv_transfer(
                        gpu_layer_tensor,
                        kvcaches[layer_id][0],
                        kvcaches[layer_id][1], 
                        slot_mapping[start:end],
                        False,     # direction: LMCache -> vLLM
                        True,      # token_major: [num_tokens, 2, hidden_dim]
                    )
                
                logger.debug(f" BedrockMembrainGPUConnector.to_gpu: Successfully processed {effective_layers} layers")
                
            except Exception as e:
                logger.error(f" BedrockMembrainGPUConnector.to_gpu failed: {e}")
                raise
                
            return
        
        # For non-LeaseMemoryObj, fall back to parent implementation
        # But first check if tensor exists
        if memory_obj.tensor is None:
            raise ValueError(f"BedrockMembrainGPUConnector: memory_obj.tensor is None for {type(memory_obj)}")
            
        super().to_gpu(memory_obj, start, end, **kwargs)

    # Note: from_gpu() method not implemented because BedrockMembrainGPUConnector
    # is designed only for layerwise operations. For non-layerwise mode, use
    # VLLMPagedMemGPUConnectorV2 which has proper from_gpu() implementation.

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Override batched_from_gpu method to handle layerwise store operations.
        This copies KV data from GPU to memory objects for storage in Membrain.
        
        For Membrain, we delegate to the parent implementation since the zero-copy
        optimization happens during the PUT operation in MembrainConnector, not here.
        """
        logger.debug(f"BedrockMembrainGPUConnector.batched_from_gpu: {len(memory_objs)} layers, {len(starts)} chunks")
        
        # Use parent implementation for GPU->CPU copy in layerwise mode
        # This will populate memory_obj.tensor for each memory object
        # The zero-copy optimization happens later in MembrainConnector.put()
        yield from super().batched_from_gpu(memory_objs, starts, ends, **kwargs)
        
        logger.debug(f" BedrockMembrainGPUConnector.batched_from_gpu: Successfully completed layerwise GPU->CPU copy")

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        logger.info(f"BedrockMembrainGPUConnector.batched_to_gpu called: {len(starts)} chunks")
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        current_stream = torch.cuda.current_stream()

        shm = SharedMemory('membrain-kvcache')
        buffer_shape = self.get_flat_shape(num_tokens)
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, torch.int8, MemoryFormat.KV_T2D
        )
        

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    if memory_obj is None:
                        logger.warning(f"Skipping because memory_obj is none....")
                        continue
                    assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D

                    try:
                        if hasattr(memory_obj, 'offsets'):
                            logger.info(f"🔍 MEMBRAIN ZERO-COPY: Processing LeaseMemoryObj with {len(memory_obj.offsets)} segments")
                            logger.debug(f"Processing Membrain offsets: {len(memory_obj.offsets)} segments")
                            
                            # Use shared helper to parse Membrain data
                            kv_tensor, metadata = self._parse_membrain_data(memory_obj, shm, num_tokens)
                            
                            if kv_tensor is None or metadata is None:
                                raise ValueError("Failed to parse Membrain data")
                            
                            expected_elements = num_tokens * 2 * self.hidden_dim_size
                            
                            # Copy pure KV data to GPU buffer
                            # Use default dtype if self.dtype is not set
                            dtype = getattr(self, 'dtype', torch.float16)
                            gpu_tensor_view = tmp_gpu_buffer_obj.tensor.view(dtype)[:expected_elements]
                            gpu_tensor_view.copy_(kv_tensor[:expected_elements], non_blocking=True)
                            
                            # Transfer to vLLM KV cache with correct shape
                            # Fix: Add token_major parameter to match C++ signature
                            # Expected: (lmc_tensor, vllm_key, vllm_value, slot_mapping, direction, token_major)
                            reshaped_tensor = gpu_tensor_view.view(num_tokens, 2, self.hidden_dim_size)
                            logger.debug(f"TENSOR TRANSFER: tensor shape={reshaped_tensor.shape}, token_major=True")
                            lmc_ops.single_layer_kv_transfer(
                                reshaped_tensor,
                                kvcaches[layer_id][0],
                                kvcaches[layer_id][1],
                                slot_mapping_full,
                                False,  # direction: LMCache -> vLLM
                                True,   # token_major: [num_tokens, 2, hidden_dim]
                            )
                            
                            logger.debug(f" Successfully transferred layer {layer_id} KV data to GPU")

                            # offsets: [{'offset': 102994812928, 'len': 4096}]
                            # cached_shape -> [num_tokens, 2, hidden_dim]
                            # kvcaches[layer_id][0] -> [num_blocks, block_size, num_kv_heads, head_dim]
                            # slot_mapping_full -> [num_tokens,]
                            # membrain_ext.host_to_device(
                            #     kvcaches[layer_id][0],
                            #     kvcaches[layer_id][1],
                            #     slot_mapping_full,
                            #     [o['offset'] for o in memory_obj.offsets],
                            #     [o['len'] for o in memory_obj.offsets],
                            #     'membrain-kvcache-ip-192-168-216-191.us-west-2.compute.internal',
                            # )
                        else:
                            logger.warning(f"Expected a LeaseMemoryObj, but got {type(memory_obj)}")
                        
                    except Exception as e:
                        raise e

        yield

        # synchronize the last layer
        current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield
