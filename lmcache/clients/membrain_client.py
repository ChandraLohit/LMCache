"""
Membrain Client Library - A high-performance client for Membrain key-value store.
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import threading
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MembrainError(Exception):
    """Base exception for all Membrain client errors."""
    pass

class MembrainConnectionError(MembrainError):
    """Raised when connection to Membrain fails."""
    pass

class MembrainKeyError(MembrainError):
    """Raised when a key operation fails."""
    pass

class MembrainTimeoutError(MembrainError):
    """Raised when an operation times out."""
    pass

@dataclass
class MembrainConfig:
    """Configuration for Membrain client.
    
    Args:
        endpoint: Base URL for Membrain aggregator (e.g., "http://localhost:9201")
        namespace: Namespace for keys (default: "default")
        timeout: Default operation timeout in seconds (default: 30.0)
        max_retries: Maximum number of retries for operations (default: 3)
        retry_delay: Base delay between retries in seconds (default: 0.1)
        pool_connections: Number of connection pools to maintain (default: 10)
        pool_maxsize: Maximum number of connections per pool (default: 10)
    """
    endpoint: str
    namespace: str = "default"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.1
    pool_connections: int = 10
    pool_maxsize: int = 10

class MembrainClient:
    """Thread-safe client for Membrain operations."""

    def __init__(self, config: MembrainConfig):
        """Initialize the client with given configuration."""
        self._config = config
        self._session = None
        self._lock = threading.RLock()
        self._closed = False

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit_per_host=self._config.pool_maxsize,
                limit=self._config.pool_connections
            )
            self._session = aiohttp.ClientSession(connector=connector)

    async def _request(
        self,
        method: str,
        key: str,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None
    ) -> bytes:
        """Make HTTP request with retries."""
        if self._closed:
            raise MembrainError("Client is closed")

        timeout = timeout or self._config.timeout
        url = urljoin(self._config.endpoint, f"/memory/{self._config.namespace}/{key}")

        for attempt in range(self._config.max_retries):
            try:
                await self._ensure_session()
                async with self._session.request(
                    method=method,
                    url=url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    # Only check for 404 on GET requests, not for PUT
                    if method == 'GET' and response.status == 404:
                        raise MembrainKeyError(f"Key not found: {key}")
                    elif response.status != 200:
                        raise MembrainError(f"HTTP {response.status}: {await response.text()}")
                    
                    return await response.read()

            except asyncio.TimeoutError:
                raise MembrainTimeoutError(f"Operation timed out after {timeout}s")
            except aiohttp.ClientError as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainConnectionError(f"Connection failed: {e}")
                await asyncio.sleep(self._config.retry_delay * (2 ** attempt))

    async def put(self, key: str, value: bytes, timeout: Optional[float] = None) -> None:
        """Put a value into Membrain.
        
        Args:
            key: Key to store the value under
            value: Bytes to store
            timeout: Optional operation timeout in seconds
            
        Raises:
            MembrainError: If the operation fails
        """
        await self._request('PUT', key, value, timeout)

    async def get(self, key: str, timeout: Optional[float] = None) -> bytes:
        """Get a value from Membrain.
        
        Args:
            key: Key to retrieve
            timeout: Optional operation timeout in seconds
            
        Returns:
            The stored bytes
            
        Raises:
            MembrainKeyError: If the key doesn't exist
            MembrainError: If the operation fails
        """
        return await self._request('GET', key, timeout=timeout)

    async def delete(self, key: str, timeout: Optional[float] = None) -> None:
        """Delete a value from Membrain.
        
        Args:
            key: Key to delete
            timeout: Optional operation timeout in seconds
            
        Raises:
            MembrainError: If the operation fails
        """
        await self._request('DELETE', key, timeout=timeout)

    async def exists(self, key: str, timeout: Optional[float] = None) -> bool:
        """Check if a key exists in Membrain.
        
        Args:
            key: Key to check
            timeout: Optional operation timeout in seconds
            
        Returns:
            True if the key exists, False otherwise
        """
        try:
            # Use a direct HEAD request if supported by your API
            # For now we'll use GET and catch 404 errors
            try:
                await self.get(key, timeout)
                return True
            except MembrainKeyError:
                return False
        except Exception:
            # Gracefully handle any errors and just return False
            return False

    async def batch_put(
        self,
        items: Dict[str, bytes],
        timeout: Optional[float] = None
    ) -> Dict[str, Exception]:
        """Put multiple key-values in parallel.
        
        Args:
            items: Dictionary of key-value pairs to store
            timeout: Optional operation timeout in seconds
            
        Returns:
            Dictionary of keys to exceptions for failed operations
        """
        async def _put(key: str, value: bytes) -> Tuple[str, Optional[Exception]]:
            try:
                await self.put(key, value, timeout)
                return key, None
            except Exception as e:
                return key, e

        tasks = [_put(k, v) for k, v in items.items()]
        results = await asyncio.gather(*tasks)
        return {k: e for k, e in results if e is not None}

    async def batch_get(
        self,
        keys: List[str],
        timeout: Optional[float] = None
    ) -> Tuple[Dict[str, bytes], Dict[str, Exception]]:
        """Get multiple keys in parallel.
        
        Args:
            keys: List of keys to retrieve
            timeout: Optional operation timeout in seconds
            
        Returns:
            Tuple of (values, errors) where:
                values: Dictionary of successfully retrieved key-value pairs
                errors: Dictionary of keys to exceptions for failed operations
        """
        async def _get(key: str) -> Tuple[str, Union[bytes, Exception]]:
            try:
                value = await self.get(key, timeout)
                return key, value
            except Exception as e:
                return key, e

        tasks = [_get(key) for key in keys]
        results = await asyncio.gather(*tasks)

        values = {}
        errors = {}
        for key, result in results:
            if isinstance(result, Exception):
                errors[key] = result
            else:
                values[key] = result

        return values, errors

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> 'MembrainClient':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()