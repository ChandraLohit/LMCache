# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.
# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png
ARG CUDA_VERSION=12.8.0
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04 AS base
ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo \
        python3 python3-dev python3-venv python3-pip tzdata \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python3 --version
# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/
WORKDIR /workspace
RUN apt-get update -y && \
    apt-get -y install \
    ninja-build \
    pybind11-dev \
    python3.12-dev \
    cmake
RUN export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
RUN export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins
RUN cd /workspace
RUN git clone https://github.com/ai-dynamo/nixl
RUN cd /workspace/nixl
RUN source /opt/venv/bin/activate
RUN . /opt/venv/bin/activate && \
    uv pip install meson
RUN cd /workspace/nixl && \
    . /opt/venv/bin/activate && \
    rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl && \
    cd build && \
    ninja && \
    ninja install
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf
RUN ldconfig
RUN cd /workspace/nixl/ && \
    . /opt/venv/bin/activate && \
    uv build --wheel --out-dir /tmp/dist && \
    uv pip install /tmp/dist/nixl-0.3.0-cp312-cp312-linux_x86_64.whl
# install build and runtime dependencies
COPY ./requirements/common.txt common.txt
COPY ./requirements/cuda.txt cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install torch==2.7.0 torchvision==0.22.0 && \
    uv pip install -r cuda.txt
# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}
#################### BASE BUILD IMAGE ####################
#################### WHEEL BUILD IMAGE ####################
# FROM base AS build
# install build dependencies
COPY ./requirements/build.txt build.txt
# max jobs used by Ninja to build extensions
ARG max_jobs=32
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=16
ENV NVCC_THREADS=$nvcc_threads
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install -r build.txt
ARG LMCACHE_COMMIT_ID=1

RUN git clone https://github.com/prashant182/LMCache.git

WORKDIR /workspace/LMCache
RUN git checkout memb
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache

#################### vLLM installation IMAGE ####################
# vllm 0.8.5.post1 was benchmarked
FROM vllm/vllm-openai:v0.8.5.post1 AS vllm-openai
ENV MAX_JOBS=10
RUN pip install --no-cache-dir \
    boto3 \
    aws-embedded-metrics
RUN --mount=type=bind,from=build,src=/workspace/LMCache/dist_lmcache,target=/vllm-workspace/dist_lmcache \
--mount=type=cache,target=/root/.cache/pip \
pip install dist_lmcache/*.whl --verbose

# Copy lmc_connector patch into vllm
# COPY patch/factory.py \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/
# COPY patch/lmcache_connector.py \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/
# COPY patch/factory.py \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/
# COPY patch/lmcache_connector.py \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/

# Use diff if file is too large
# COPY patch/parallel_state.patch \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/
# COPY patch/parallel_state.patch \
#     /usr/local/lib/python3.12/dist-packages/vllm/distributed/

#RUN patch /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py \
#    /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.patch
#RUN patch /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py \
#    /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.patch


ENTRYPOINT ["vllm", "serve"]
