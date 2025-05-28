import os
from aws_embedded_metrics import metric_scope
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Get namespace from environment variables or use default
METRICS_NAMESPACE = os.environ.get("LMCACHE_METRICS_NAMESPACE", "LMCache-Metrics")

# Define units as string constants instead of using the enum
class Unit:
    COUNT = "Count"
    MILLISECONDS = "Milliseconds"
    BYTES = "Bytes"
    PERCENT = "Percent"

@metric_scope
async def emit_cache_hit(cache_level, operation, metrics):
    """Emit a cache hit metric."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "Operation": operation})
    metrics.put_metric("CacheHit", 1, Unit.COUNT)

@metric_scope
async def emit_cache_miss(cache_level, operation, metrics):
    """Emit a cache miss metric."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "Operation": operation})
    metrics.put_metric("CacheMiss", 1, Unit.COUNT)

@metric_scope
async def emit_cache_error(cache_level, operation, metrics):
    """Emit a cache error metric."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "Operation": operation})
    metrics.put_metric("CacheError", 1, Unit.COUNT)

@metric_scope
async def emit_cache_latency(cache_level, operation, latency_ms, metrics):
    """Emit a cache latency metric in milliseconds."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "Operation": operation})
    metrics.put_metric("CacheLatency", latency_ms, Unit.MILLISECONDS)

@metric_scope
async def emit_cache_bytes(cache_level, operation, bytes_count, metrics):
    """Emit a cache size metric in bytes."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "Operation": operation})
    metrics.put_metric("CacheBytes", bytes_count, Unit.BYTES)

@metric_scope
async def emit_cache_hit_rate(cache_level, hit_rate, metrics):
    """Emit a cache hit rate metric as percentage."""
    metrics.set_namespace(METRICS_NAMESPACE)
    metrics.set_dimensions({"CacheLevel": cache_level, "MetricType": "Summary"})
    metrics.put_metric("CacheHitRate", hit_rate * 100, Unit.PERCENT)