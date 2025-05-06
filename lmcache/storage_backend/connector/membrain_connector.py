import asyncio
import inspect
import hashlib
import base64
from typing import List, Optional, Dict

from lmcache.logging import init_logger
from lmcache.storage_backend.connector.base_connector import RemoteBytesConnector

# Import the Membrain client
from lmcache.clients.membrain_client import MembrainClient, MembrainConfig, MembrainError, MembrainKeyError

logger = init_logger(__name__)


class MembrainConnector(RemoteBytesConnector):
    """
    Connector for Membrain key-value store.
    The remote url should start with "membrain://" and include a host and port.
    """

    def __init__(self, endpoint: str, namespace: str = "lmcache"):
        # Create Membrain client configuration
        self.config = MembrainConfig(
            endpoint=endpoint,
            namespace=namespace,
            timeout=300.0  # Reasonable default timeout
        )
        # Initialize client
        self.client = MembrainClient(self.config)
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        # Key mapping to keep track of original to hashed keys
        self._key_mapping: Dict[str, str] = {}
        logger.info(f"Initialized Membrain connector with endpoint {endpoint}, namespace {namespace}")
        
    def _hash_key(self, key: str) -> str:
        """
        Hash the long key into a shorter, URL-safe string.
        
        Args:
            key: The original long key string
            
        Returns:
            A URL-safe hashed key string
        """
        # First check if we already hashed this key
        if key in self._key_mapping:
            return self._key_mapping[key]
            
        # Create a hash of the key
        key_hash = hashlib.sha256(key.encode()).digest()
        # Convert to URL-safe base64 and remove padding
        safe_key = base64.urlsafe_b64encode(key_hash).decode().rstrip('=')
        # Store mapping for debug and reference
        self._key_mapping[key] = safe_key
        logger.debug(f"Hashed key: {key} -> {safe_key}")
        return safe_key

    def exists(self, key: str) -> bool:
        """Check if the key exists in Membrain."""
        try:
            # Use hashed key for Membrain
            hashed_key = self._hash_key(key)
            logger.info(f"MEMBRAIN EXISTS: namespace={self.config.namespace}, key={hashed_key}")
            result = self.loop.run_until_complete(self.client.exists(hashed_key))
            logger.info(f"MEMBRAIN EXISTS RESPONSE: {result} for key {hashed_key}")
            logger.debug(f"Key existence check for {key} (hash: {hashed_key}): {result}")
            return result
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """Get value for key from Membrain."""
        try:
            # Use hashed key for Membrain
            hashed_key = self._hash_key(key)
            logger.info(f"MEMBRAIN GET: namespace={self.config.namespace}, key={hashed_key}")
            result = self.loop.run_until_complete(self.client.get(hashed_key))
            # Ensure result is not a coroutine
            assert not inspect.isawaitable(result)
            if result:
                logger.info("CHEN WORKING ON MEMBRAIN HERE =====>")
                logger.info(f"MEMBRAIN GET SUCCESS: key={hashed_key}, size={len(result)} bytes")
            else:
                logger.info(f"MEMBRAIN GET FAILED: key={hashed_key} not found")
            logger.debug(f"Got value for key {key} (hash: {hashed_key}), size: {len(result) if result else 0} bytes")
            return result
        except MembrainKeyError:
            # Key not found
            logger.debug(f"Key not found: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def set(self, key: str, obj: bytes) -> None:  # type: ignore[override]
        """Set value for key in Membrain."""
        try:
            # Use hashed key for Membrain
            hashed_key = self._hash_key(key)
            logger.info(f"MEMBRAIN PUT: namespace={self.config.namespace}, key={hashed_key}, size={len(obj)} bytes")
            response = self.loop.run_until_complete(self.client.put(hashed_key, obj))
            logger.info(f"MEMBRAIN PUT RESPONSE: {response} for key {hashed_key}")
            logger.debug(f"Set value for key {key}, size: {len(obj)} bytes")
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")

    def list(self) -> List[str]:
        """
        List all keys in Membrain.
        Note: Membrain client doesn't support listing keys,
        so this returns an empty list.
        """
        logger.warning("List operation not supported by Membrain client")
        return []

    def close(self) -> None:
        """Close the Membrain client."""
        try:
            self.loop.run_until_complete(self.client.close())
            self.loop.close()
            logger.info("Closed Membrain connector")
        except Exception as e:
            logger.error(f"Error closing Membrain connector: {e}")