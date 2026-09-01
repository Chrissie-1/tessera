# CPU image: small and sufficient for correctness tests and CI.
# For GPU serving, swap the base for nvidia/cuda:12.4.1-runtime-ubuntu22.04 and
# drop the --index-url pin so the CUDA torch wheel is installed instead.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/tessera/.cache/huggingface

# HF_HOME is a volume mount point in docker-compose. Docker seeds a new named
# volume from the image, ownership included, but only if the path exists in the
# image; otherwise it creates the mount point root-owned and the unprivileged
# user cannot write the model cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 tessera \
    && mkdir -p /home/tessera/.cache/huggingface \
    && chown -R tessera:tessera /home/tessera

WORKDIR /app

# CPU-only torch keeps the image ~1 GB instead of ~6 GB.
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY worker/pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
    "transformers>=4.44" "grpcio>=1.60" "grpcio-tools>=1.60" "protobuf>=4.25" \
    "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.6"

COPY proto /app/proto
COPY scripts /app/scripts
COPY worker/tessera_worker /app/tessera_worker

# Generate stubs inside the image so they are never committed stale.
RUN mkdir -p /app/tessera_worker/generated \
    && touch /app/tessera_worker/generated/__init__.py \
    && python -m grpc_tools.protoc \
        --proto_path=/app/proto \
        --python_out=/app/tessera_worker/generated \
        --grpc_python_out=/app/tessera_worker/generated \
        /app/proto/inference.proto \
    && sed -i 's/^import inference_pb2/from . import inference_pb2/' \
        /app/tessera_worker/generated/inference_pb2_grpc.py \
    && chown -R tessera:tessera /app

USER tessera
EXPOSE 50051 8000

ENV TESSERA_MODEL=gpt2 \
    TESSERA_GRPC_PORT=50051 \
    TESSERA_DEVICE=auto

ENTRYPOINT ["python", "-m", "tessera_worker.server"]
