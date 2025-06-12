import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from typing import Optional
import threading
from urllib.parse import urljoin
from contextlib import asynccontextmanager

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
        endpoint: Base URL for Membrain KV Cache Server (e.g., "http://localhost:9200")
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
                limit=self._config.pool_connections,
            )
            self._session = aiohttp.ClientSession(connector=connector)
    
    async def _request(
        self,
        method: str,
        key: str,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Make HTTP request with retries."""
        if self._closed:
            raise MembrainError("Client is closed")

        timeout = timeout or self._config.timeout
        url = urljoin(self._config.endpoint, f"/v1/kv/{self._config.namespace}/{key}")

        for attempt in range(self._config.max_retries):
            try:
                await self._ensure_session()
                async with self._session.request(
                    method=method,
                    url=url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    # Only check for 404 on GET requests, not for PUT
                    if method == "GET" and response.status == 404:
                        raise MembrainKeyError(f"Key not found: {key}")
                    elif response.status != 200:
                        raise MembrainError(
                            f"HTTP {response.status}: {await response.text()}"
                        )

                    return await response.read()

            except asyncio.TimeoutError:
                raise MembrainTimeoutError(f"Operation timed out after {timeout}s")
            except aiohttp.ClientError as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainConnectionError(f"Connection failed: {e}")
                await asyncio.sleep(self._config.retry_delay * (2**attempt))

    async def put(
        self, key: str, value: bytes, timeout: Optional[float] = None
    ) -> None:
        """Put a value into Membrain.

        Args:
            key: Key to store the value under
            value: Bytes to store
            timeout: Optional operation timeout in seconds

        Raises:
            MembrainError: If the operation fails
        """
        return await self._request("PUT", key, value, timeout)

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
        return await self._request("GET", key, timeout=timeout)

    async def exists(self, key: str, timeout: Optional[float] = None) -> bool:
        """Check if a key exists in Membrain using HEAD request.

        Args:
            key: Key to check
            timeout: Optional operation timeout in seconds

        Returns:
            True if the key exists, False otherwise
        """
        if self._closed:
            raise MembrainError("Client is closed")

        timeout = timeout or self._config.timeout
        url = urljoin(self._config.endpoint, f"/v1/kv/{self._config.namespace}/{key}")

        try:
            await self._ensure_session()
            async with self._session.request(
                method="HEAD",  # Use HEAD instead of lease acquisition
                url=url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status == 200:
                    return True
                elif response.status == 404:
                    return False
                else:
                    logger.warning(f"Unexpected status {response.status} checking existence for key {key}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout checking existence for key {key}")
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"Connection error checking existence for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking existence for key {key}: {e}")
            return False

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "MembrainClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_lease_request(
        self,
        key: str,
        timeout: Optional[float] = None,
    ) -> dict:
        """Make HTTP request with retries."""
        if self._closed:
            raise MembrainError("Client is closed")

        timeout = timeout or self._config.timeout
        url = urljoin(self._config.endpoint, f"/v1/kv/{self._config.namespace}/{key}/leases")

        for attempt in range(self._config.max_retries):
            try:
                await self._ensure_session()
                async with self._session.request(
                    method="POST",
                    url=url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    # Only check for 404 on GET requests, not for PUT
                    if response.status != 200:
                        raise MembrainError(
                            f"HTTP {response.status}: {await response.text()}"
                        )

                    return await response.json()

            except asyncio.TimeoutError:
                raise MembrainTimeoutError(f"Operation timed out after {timeout}s")
            except aiohttp.ClientError as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainConnectionError(f"Connection failed: {e}")
                await asyncio.sleep(self._config.retry_delay * (2**attempt))
    
    async def _free_lease_request(
        self,
        lease: str,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Make HTTP request with retries."""
        if self._closed:
            raise MembrainError("Client is closed")

        timeout = timeout or self._config.timeout
        url = urljoin(self._config.endpoint, f"/v1/leases/{lease}/release")

        for attempt in range(self._config.max_retries):
            try:
                await self._ensure_session()
                async with self._session.request(
                    method="POST",
                    url=url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    # Only check for 404 on GET requests, not for PUT
                    if response.status != 200:
                        raise MembrainError(
                            f"HTTP {response.status}: {await response.text()}"
                        )

                    return await response.json()

            except asyncio.TimeoutError:
                raise MembrainTimeoutError(f"Operation timed out after {timeout}s")
            except aiohttp.ClientError as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainConnectionError(f"Connection failed: {e}")
                await asyncio.sleep(self._config.retry_delay * (2**attempt))

    async def acquire_kv_lease_manual(self, key):
        """Acquire lease manually without context manager - caller must release."""
        return await self._get_lease_request(key)
    
    async def release_lease_manual(self, lease_id):
        """Release lease manually.""" 
        return await self._free_lease_request(lease_id)

    @asynccontextmanager
    async def acquire_kv_lease(self, key):
        response_body = await self._get_lease_request(key)
        lease_id = response_body.get('id')
        try:
            yield response_body
        finally:
            await self._free_lease_request(lease_id)

        


