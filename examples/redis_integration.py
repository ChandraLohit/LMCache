"""
This example demonstrates using Redis as a backend for LMCache with vLLM.
It runs two requests to show the improved performance on the second request
when KV cache is retrieved from Redis.

Requirements:
- Redis server running on localhost:65432
- lmcache installed in your environment
- vllm installed in your environment
"""
import logging
import os
import subprocess
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("lmcache").setLevel(logging.DEBUG)
logging.getLogger("lmcache.storage_backend.connector.redis_connector").setLevel(logging.DEBUG)
logging.getLogger("lmcache.v1.storage_backend").setLevel(logging.DEBUG)

# Configure LMCache to use Redis backend
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = "/home/ec2-user/LMCache/examples/sample_integration/redis_example.yaml"

def check_redis_keys():
    """Check Redis keys before and after operations."""
    print("\n[Checking Redis keys]")
    result = subprocess.run(["redis6-cli", "-p", "65432", "KEYS", "*"], 
                           capture_output=True, text=True)
    keys = result.stdout.strip().split("\n")
    if not keys or (len(keys) == 1 and keys[0] == ''):
        print("No keys found in Redis")
    else:
        print(f"Found {len(keys)} keys in Redis:")
        for key in keys[:5]:  # Show just first 5 keys if there are many
            print(f"- {key}")
        if len(keys) > 5:
            print(f"...and {len(keys)-5} more")
        
        # Get memory usage of a sample key
        if keys:
            result = subprocess.run(["redis6-cli", "-p", "65432", "MEMORY", "USAGE", keys[0]], 
                                  capture_output=True, text=True)
            print(f"Memory usage of first key: {result.stdout.strip()}")

def print_output(llm, prompt, sampling_params, req_str):
    """Run the generation and print timing information."""
    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {elapsed:.4f} seconds, {req_str} request done.")
    print("-" * 50)
    return elapsed

def main():
    # Check Redis keys before starting
    check_redis_keys()
    
    # Configure KV transfer to use LMCache
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1", 
        kv_role="kv_both",
    )
    
    # Initialize the model
    llm = LLM(
        model="/home/ec2-user/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
        kv_transfer_config=ktc,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    
    # Define prompts - using a longer prompt to show the benefit of caching
    shared_prompt = "Hello, how are you? " * 50
    prompt = shared_prompt + "Tell me about yourself."
    
    # Configure sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
    
    # First request - will be stored in Redis
    print("\n[First Request - Storing KV Cache in Redis]")
    time1 = print_output(llm, prompt, sampling_params, "first")
    
    # Check Redis keys after first request
    check_redis_keys()
    
    print("\nRunning the same prompt again to demonstrate cache retrieval from Redis...\n")
    
    # Second request - should retrieve from Redis and be faster
    print("[Second Request - Retrieving KV Cache from Redis]")
    time2 = print_output(llm, prompt, sampling_params, "second")
    
    # Calculate the speedup
    if time1 > 0:
        speedup = time1 / time2
        print(f"\nSpeedup from using Redis KV cache: {speedup:.2f}x")

    # Final check of Redis keys
    check_redis_keys()

if __name__ == "__main__":
    main()