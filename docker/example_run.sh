IMAGE=vllm/openai:membrain
docker run --runtime nvidia --gpus all \
    --env "HF_TOKEN=<SOMETHING>" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "chunk_size=256" \
    --env "local_cpu=True" \
    --env "max_local_cpu_size=5" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    $IMAGE \
    $HF_MODEL_NAME --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
