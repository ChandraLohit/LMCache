#!/bin/bash

# Configuration
IMAGE=vllm/openai:membrain
MODEL="mistralai/Mistral-7B-Instruct-v0.2"

# Create the config directory if it doesn't exist
mkdir -p $(pwd)/config

# Create the Membrain configuration file
cat > $(pwd)/config/membrain_config.yaml << EOL
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5
# For local testing - replace with your Membrain endpoint
remote_url: "membrain://localhost:9201?namespace=lmcache"
remote_serde: "cachegen"
save_decode_cache: false
enable_blending: false
EOL

# Run the container with mounted config and debug logs enabled
docker run --runtime nvidia --gpus all \
    --env "VLLM_USE_V1=1" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "LMCACHE_CONFIG_FILE=/config/membrain_config.yaml" \
    --env "LMCACHE_LOGGING_LEVEL=DEBUG" \
    --env "VLLM_LOGGING_LEVEL=DEBUG" \
    --env "VLLM_CONFIGURE_LOGGING=1" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/config:/config \
    --network host \
    -p 8000:8000 \
    --name vllm-membrain \
    --entrypoint "/usr/local/bin/vllm" \
    $IMAGE \
    serve $MODEL --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --uvicorn-log-level debug