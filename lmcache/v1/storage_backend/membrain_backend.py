import asyncio
import re
import time
from concurrent.futures import Future
from typing import Optional
from membrain_connector import (
    MembrainConnector,
)
from lmcache.v1.memory_management import MemoryAllocatorInterface
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import parse_remote_url
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector


logger = init_logger(__name__)


def create_membrain_connector(
    url: str,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: LocalCPUBackend,
) -> RemoteConnector:
    """
    Creates the corresponding remote connector from the given URL.
    """
    m = re.match(r"(.*)://(.*):(\d+)", url)
    if m is None:
        raise ValueError(f"Invalid remote url {url}")

    parsed_url = parse_remote_url(url)
    num_hosts = len(parsed_url.hosts)

    if num_hosts == 1:
        host, port = parsed_url.hosts[0], parsed_url.ports[0]
        endpoint = f"http://{host}:{port}"
        namespace = parsed_url.query_params[0].get("namespace", "lmcache")
        connector = MembrainConnector(endpoint, namespace, loop, local_cpu_backend)
    else:
        raise ValueError(
            f"Membrain connector only supports a single host, but got url: {url}"
        )

    logger.info(f"Created connector {connector} for membrain")
    return connector


class MembrainBackend(RemoteBackend):
    def _init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            assert self.config.remote_url is not None
            self.connection = InstrumentedRemoteConnector(
                create_membrain_connector(
                    self.config.remote_url, self.loop, self.local_cpu_backend
                )
            )
            logger.info(
                f"Connection initialized/re-established at {self.config.remote_url}"
            )
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(f"FaiI fed to initialize/re-establish remote connection: {e}")
            self.connection = None
    
    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non blocking get function.
        """
        if self.connection is None:
            logger.warning("Connection is None in get_non_blocking, returning None")
            return None
        return asyncio.run_coroutine_threadsafe(
            self._get_non_blocking_async(key), loop=self.loop
        )

    # async version of get_non_blocking to await the connector
    async def _get_non_blocking_async(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        t1 = time.perf_counter()
        try:
            memory_obj = await self.connection.get(key)
        except Exception as e:
            with self.lock:
                self.connection = None
                self.failure_time = time.time()
            logger.warning(f"Error occurred in get_non_blocking: {e}")
            logger.warning("Returning None")
            return None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get((t2 - t1) * 1000)
        if memory_obj is None:
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            f"Get non blocking takes {(t2 - t1) * 1000:.6f} msec, "
            f"deserialization takes {(t3 - t2) * 1000:.6f} msec"
        )
        return decompressed_memory_obj

    def unpin(self, key):
        return True