import asyncio
import time
from typing import Dict, Optional

from lmcache.experimental.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.experimental.storage_backend.connector.membrain_connector import MembrainConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

# Constants
PUT_TIMEOUT_THRESHOLD_SECONDS = 10.0


class MembrainConnectorV2(MembrainConnector):
    """
    Enhanced connector for Membrain key-value store with additional functionality.
    Extends the base MembrainConnector with cluster management capabilities.
    Includes automatic reset if put operations take too long.
    """

    def __init__(self, endpoint: str, namespace: str,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        """Initialize the enhanced Membrain connector."""
        super().__init__(endpoint, namespace, loop, memory_allocator)
        self._reset_in_progress = False
        self._reset_lock = asyncio.Lock()
        logger.info(f"Initialized MembrainConnectorV2 with enhanced cluster management capabilities")
        logger.info(f"Automatic reset enabled for put operations exceeding {PUT_TIMEOUT_THRESHOLD_SECONDS}s")
    
    async def reset(self, timeout: Optional[float] = None) -> bool:
        """Reset the Membrain cluster using the supernova API."""
        # Use a lock to prevent multiple concurrent resets
        async with self._reset_lock:
            if self._reset_in_progress:
                logger.warning("Reset already in progress, waiting for completion")
                return False
                
            try:
                self._reset_in_progress = True
                logger.info("Initiating cluster reset...")
                start_time = time.time()
                await self.client.reset(timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Reset local stats counters
                self.cache_hits = 0
                self.cache_misses = 0
                self.total_bytes_get = 0
                self.total_bytes_put = 0
                
                logger.info(f"Successfully reset Membrain cluster in {elapsed_ms:.2f}ms")
                return True
            except Exception as e:
                logger.error(f"Failed to reset Membrain cluster: {e}")
                return False
            finally:
                self._reset_in_progress = False
    
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Store data in Membrain with metadata using hashed keys.
        Monitors the operation time and triggers a reset if it takes too long.
        """
        start_time = time.time()
        original_key = key.to_string() if key else "unknown"
        
        # Execute the parent implementation
        try:
            await super().put(key, memory_obj)
        except Exception as e:
            logger.error(f"Error in put operation for key {original_key}: {e}")
        
        # After completion, check the elapsed time
        elapsed_time = time.time() - start_time
        logger.debug(f"Put operation for key {original_key} took {elapsed_time:.2f}s")
        
        # If operation took too long, trigger a reset
        if elapsed_time > PUT_TIMEOUT_THRESHOLD_SECONDS:
            logger.warning(f"Put operation for key {original_key} took {elapsed_time:.2f}s, "
                          f"exceeding threshold of {PUT_TIMEOUT_THRESHOLD_SECONDS}s")
            
            # Trigger reset in background task
            logger.info(f"Initiating reset due to slow PUT operation")
            asyncio.create_task(self._trigger_reset())
    
    async def _trigger_reset(self):
        """Performs a reset operation in response to a slow put."""
        try:
            reset_success = await self.reset()
            if reset_success:
                logger.info("Reset completed successfully after slow PUT operation")
            else:
                logger.error("Reset failed after slow PUT operation")
        except Exception as e:
            logger.error(f"Error during reset after slow PUT operation: {e}")
    
    async def get_cluster_info(self) -> Dict:
        """
        Get information about the Membrain cluster.
        """
        return {
            "connector_type": "MembrainConnectorV2",
            "endpoint": self.config.endpoint,
            "namespace": self.config.namespace,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_transferred_get": self.total_bytes_get,
            "bytes_transferred_put": self.total_bytes_put,
            "auto_reset_threshold": f"{PUT_TIMEOUT_THRESHOLD_SECONDS}s",
            "reset_in_progress": self._reset_in_progress
        }