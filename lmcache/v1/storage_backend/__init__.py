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
from typing import TYPE_CHECKING, Optional
import asyncio

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.weka_gds_backend import WekaGdsBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    lookup_server: Optional[LookupServerInterface] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    # ZERO-COPY FIX: Create minimal LocalCPU backend if max_local_cpu_size is 0
    logger.info(f"LocalCPU Backend Creation: max_local_cpu_size={config.max_local_cpu_size} (type: {type(config.max_local_cpu_size)}), remote_url={config.remote_url}")
    
    # Handle various representations of zero: 0, 0.0, or very small values
    is_zero_cpu = (
        config.max_local_cpu_size == 0 or 
        config.max_local_cpu_size == 0.0 or 
        (isinstance(config.max_local_cpu_size, (int, float)) and abs(config.max_local_cpu_size) < 0.001)
    )
    has_remote = config.remote_url is not None and config.remote_url.strip() != ""
    
    if is_zero_cpu and has_remote:
        # For zero-copy remote backends, we need minimal LocalCPU buffer for temporary storage
        # during PUT operations before transfer to remote
        try:
            from dataclasses import replace
            minimal_config = replace(config, max_local_cpu_size=0.1)  # 100MB minimal buffer
            logger.info(" ZERO-COPY MODE: Creating minimal LocalCPU buffer (100MB) for remote backend operations")
            local_cpu_backend = LocalCPUBackend(
                minimal_config,
                memory_allocator,
                lookup_server,
                lmcache_worker,
            )
        except Exception as e:
            logger.error(f"Failed to create minimal LocalCPU backend: {e}")
            # Fallback to original config
            local_cpu_backend = LocalCPUBackend(
                config,
                memory_allocator,
                lookup_server,
                lmcache_worker,
            )
    else:
        logger.info(f"Standard LocalCPU backend: is_zero_cpu={is_zero_cpu}, has_remote={has_remote}")
        local_cpu_backend = LocalCPUBackend(
            config,
            memory_allocator,
            lookup_server,
            lmcache_worker,
        )
    backend_name = str(local_cpu_backend)
    storage_backends[backend_name] = local_cpu_backend

    if config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(
            config,
            loop,
            local_cpu_backend,
            dst_device,
            lmcache_worker,
            lookup_server,
        )
        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.weka_path is not None:
        weka_backend = WekaGdsBackend(config, loop, memory_allocator, dst_device)
        # TODO(Serapheim): there's a chance we don't want the local
        # CPU cache in front of ours. Let's experiment and potentially
        # change that in the future.
        storage_backends[str(weka_backend)] = weka_backend
    if config.gds_path is not None:
        gds_backend = GdsBackend(config, loop, memory_allocator, dst_device)
        storage_backends[str(gds_backend)] = gds_backend
    if config.remote_url is not None:
        remote_backend = RemoteBackend(
            config, metadata, loop, local_cpu_backend, dst_device, lookup_server
        )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    return storage_backends
