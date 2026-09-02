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
# Read the dependencies out of pyproject rather than repeating them here. A
# second copy of the list drifts silently -- adding prometheus-client to
# pyproject once left this image without it, and a missing runtime dependency
# does not surface until the container starts.
#
# torch is filtered out because it is already installed above from the CPU
# index; letting pip resolve it here would pull the multi-gigabyte CUDA wheel.
RUN python -c "import tomllib; d = tomllib.load(open('/app/pyproject.toml','rb')); print('\n'.join(r for r in d['project']['dependencies'] if not r.startswith('torch')))" > /tmp/requirements.txt \
    && cat /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt "grpcio-tools>=1.60"

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
