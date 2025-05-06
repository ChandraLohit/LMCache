from typing import TYPE_CHECKING

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:

    @staticmethod
    def create_connector(rank: int, local_rank: int,
                         config: "VllmConfig") -> KVConnectorBase:
        supported_kv_connector = [
            "PyNcclConnector", "MooncakeConnector", "LMCacheConnector", "LMCacheConnectorV1"
        ]
        kv_connector = config.kv_transfer_config.kv_connector
        if kv_connector in supported_kv_connector:
            if kv_connector in ["PyNcclConnector", "MooncakeConnector"]:
                from .simple_connector import SimpleConnector
                return SimpleConnector(rank, local_rank, config)
            elif kv_connector in ["LMCacheConnector"]:
                from .lmcache_connector import LMCacheConnector
                return LMCacheConnector(rank, local_rank, config)
        else:
            raise ValueError(f"Unsupported connector type: "
                             f"{config.kv_connector}")
                             
    @staticmethod
    def create_connector_v1(config: "VllmConfig", role=None):
        """Create connector for v1 API."""
        kv_connector = config.kv_transfer_config.kv_connector
        
        # Simplified implementation - just use the same connectors
        # but ignore the role parameter
        if kv_connector in ["PyNcclConnector", "MooncakeConnector"]:
            from .simple_connector import SimpleConnector
            return SimpleConnector(0, 0, config)
        elif kv_connector in ["LMCacheConnectorV1"]:
            from .lmcache_connector import LMCacheConnector
            return LMCacheConnectorV1(0, 0, config)
        else:
            raise ValueError(f"Unsupported connector type: {kv_connector}")