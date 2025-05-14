import boto3
import time
import threading
from typing import Dict, List, Optional
import os
from lmcache.logging import init_logger

logger = init_logger(__name__)

class CloudWatchMetricsPublisher:
    """Class for publishing cache metrics to CloudWatch."""
    
    def __init__(self, 
                namespace: str = "LMCache", 
                region: Optional[str] = None,
                publish_interval: int = 60):
        """Initialize CloudWatch metrics publisher.
        
        Args:
            namespace: CloudWatch namespace for metrics
            region: AWS region (defaults to env var or boto3 default)
            publish_interval: Interval in seconds for publishing metrics
        """
        self.namespace = namespace
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.publish_interval = publish_interval
        self.client = boto3.client('cloudwatch', region_name=self.region)
        
        # Metrics storage
        self._metrics_buffer: List[Dict] = []
        self._lock = threading.RLock()
        
        # Start background publishing thread
        self._should_run = True
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()
        
        logger.info(f"Initialized CloudWatch metrics publisher with namespace '{namespace}', region '{self.region}'")
    
    def add_metric(self, metric_name: str, value: float, 
                   unit: str = 'Count', dimensions: Optional[Dict[str, str]] = None):
        """Add a metric to be published to CloudWatch.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit ('Count', 'Bytes', 'Seconds', etc.)
            dimensions: Optional dimensions for the metric (e.g., {'CacheLevel': 'L1'})
        """
        with self._lock:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': key, 'Value': value} 
                    for key, value in dimensions.items()
                ]
                
            self._metrics_buffer.append(metric_data)
    
    def _publish_loop(self):
        """Background thread that publishes metrics at regular intervals."""
        while self._should_run:
            time.sleep(self.publish_interval)
            self._flush_metrics()
    
    def _flush_metrics(self):
        """Send accumulated metrics to CloudWatch."""
        with self._lock:
            if not self._metrics_buffer:
                return
                
            # Group metrics into batches (max 20 per request)
            batch_size = 20
            for i in range(0, len(self._metrics_buffer), batch_size):
                batch = self._metrics_buffer[i:i+batch_size]
                
                try:
                    self.client.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch
                    )
                    logger.debug(f"Published {len(batch)} metrics to CloudWatch")
                except Exception as e:
                    logger.error(f"Failed to publish metrics to CloudWatch: {e}")
            
            # Clear the buffer after sending
            self._metrics_buffer.clear()
    
    def shutdown(self):
        """Shutdown the metrics publisher."""
        self._should_run = False
        self._flush_metrics()  # Final flush
        if self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=5.0)
        logger.info("CloudWatch metrics publisher shutdown")
