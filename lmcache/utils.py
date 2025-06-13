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

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import hashlib
import threading

# Third Party
from nvtx import annotate  # type: ignore
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: MemoryFormat = None
    is_pin: bool = False

    def pin(self) -> bool:
        self.is_pin = True
        return True

    def unpin(self) -> bool:
        self.is_pin = False
        return True

    @property
    def is_pinned(self) -> bool:
        return self.is_pin


TORCH_DTYPE_TO_STR_DTYPE = {
    torch.half: "half",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double",
    torch.uint8: "fp8",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
}


@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
            )
        )

    def to_string(self):
        # FIXED: Remove worker_id from cache key to enable shared caching across workers
        return (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """Return the key for the first layer"""
        key = LayerCacheEngineKey(
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        # Handle both old format (5 parts with worker_id) and new format (4 parts without worker_id)
        if len(parts) == 5:
            # Old format: fmt@model@world_size@worker_id@chunk_hash  
            return CacheEngineKey(
                parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4]
            )
        elif len(parts) == 4:
            # New format: fmt@model@world_size@chunk_hash (worker_id = 0 for compatibility)
            return CacheEngineKey(
                parts[0], parts[1], int(parts[2]), 0, parts[3]
            )
        else:
            raise ValueError(f"Invalid key string: {s} (expected 4 or 5 parts, got {len(parts)})")

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "CacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
        }

    @staticmethod
    def from_dict(d):
        return CacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
        )


@dataclass(order=True)
class LayerCacheEngineKey(CacheEngineKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        # FIXED: Remove worker_id from cache key to enable shared caching across workers
        key_str = (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.chunk_hash}@{self.layer_id}"
        )
        # Debug key generation (worker_id kept for internal use but not in key string)
        # Note: worker_id is still part of the object but not the serialized key
        return key_str

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        # Handle both old format (6 parts with worker_id) and new format (5 parts without worker_id)
        if len(parts) == 6:
            # Old format: fmt@model@world_size@worker_id@chunk_hash@layer_id
            return LayerCacheEngineKey(
                parts[0],
                parts[1],
                int(parts[2]),
                int(parts[3]),
                parts[4],
                int(parts[5]),
            )
        elif len(parts) == 5:
            # New format: fmt@model@world_size@chunk_hash@layer_id (worker_id = 0 for compatibility)
            return LayerCacheEngineKey(
                parts[0],
                parts[1],
                int(parts[2]),
                0,  # worker_id = 0 for compatibility
                parts[3],
                int(parts[4]),
            )
        else:
            raise ValueError(f"Invalid LayerCacheEngineKey string: {s} (expected 5 or 6 parts, got {len(parts)})")


##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)


##### Observability Threading related #####
_shared_observability_lock = threading.Lock()


def thread_safe(func):
    def wrapper(*args, **kwargs):
        with _shared_observability_lock:
            result = func(*args, **kwargs)
        return result

    return wrapper
