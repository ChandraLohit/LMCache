"""
This example demonstrates using Membrain as a backend for LMCache with vLLM.
It runs two requests to show the improved performance on the second request
when KV cache is retrieved from Membrain.

Requirements:
- Membrain service running on localhost:9201
- lmcache installed in your environment
- vllm installed in your environment
- membrain_client available in your environment
"""
import logging
import os
import time
import asyncio

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("lmcache").setLevel(logging.DEBUG)
logging.getLogger("lmcache.storage_backend.connector.membrain_connector").setLevel(logging.DEBUG)
logging.getLogger("lmcache.experimental.storage_backend").setLevel(logging.DEBUG)

# Import Membrain client for checking keys
from lmcache.clients.membrain_client import MembrainClient, MembrainConfig

# Configuration for Membrain
MEMBRAIN_ENDPOINT = "http://localhost:9201"
MEMBRAIN_NAMESPACE = "lmcache"

# Create a configuration file for Membrain integration
MEMBRAIN_CONFIG = """chunk_size: 256
local_cpu: true
max_local_cpu_size: 5
remote_url: "membrain://localhost:9201?namespace=lmcache"
remote_serde: "cachegen"
save_decode_cache: false
enable_blending: false
"""

CONFIG_PATH = "/home/ec2-user/LMCache/examples/membrain_example.yaml"

# Write the configuration file
with open(CONFIG_PATH, "w") as f:
    f.write(MEMBRAIN_CONFIG)

# Configure LMCache to use Membrain backend
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = CONFIG_PATH

async def check_membrain_keys():
    """Check Membrain keys before and after operations."""
    print("\n[Checking Membrain keys]")
    try:
        # Create Membrain client
        config = MembrainConfig(
            endpoint=MEMBRAIN_ENDPOINT,
            namespace=MEMBRAIN_NAMESPACE
        )
        client = MembrainClient(config)
        
        # We don't have a direct way to list keys in the Membrain API,
        # so we'll just report that we're connected
        print(f"Connected to Membrain at {MEMBRAIN_ENDPOINT}")
        print("Note: Membrain client doesn't support listing keys.")
        
        # Close the client
        await client.close()
    except Exception as e:
        print(f"Error connecting to Membrain: {e}")

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

async def main():
    # Check Membrain connectivity before starting
    await check_membrain_keys()
    
    # Configure KV transfer to use LMCache
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1", 
        kv_role="kv_both",
    )
    
    # Initialize the model - modify the model path if needed
    model_path = "/home/ec2-user/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
    
    try:
        llm = LLM(
            model=model_path,
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
        
        # First request - will be stored in Membrain
        print("\n[First Request - Storing KV Cache in Membrain]")
        time1 = print_output(llm, prompt, sampling_params, "first")
        
        # Check Membrain after first request
        await check_membrain_keys()
        
        print("\nRunning the same prompt again to demonstrate cache retrieval from Membrain...\n")
        
        # Second request - should retrieve from Membrain and be faster
        print("[Second Request - Retrieving KV Cache from Membrain]")
        time2 = print_output(llm, prompt, sampling_params, "second")
        
        # Calculate the speedup
        if time1 > 0:
            speedup = time1 / time2
            print(f"\nSpeedup from using Membrain KV cache: {speedup:.2f}x")

        # Final check of Membrain keys
        await check_membrain_keys()
        
    except Exception as e:
        print(f"Error in LLM execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())