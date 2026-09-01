#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from proto/inference.proto.
# Rust stubs are generated at build time by gateway/build.rs, so this script
# only covers the Python side.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/worker/tessera_worker/generated"

mkdir -p "${OUT_DIR}"
touch "${OUT_DIR}/__init__.py"

python -m grpc_tools.protoc \
  --proto_path="${REPO_ROOT}/proto" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  "${REPO_ROOT}/proto/inference.proto"

# protoc emits `import inference_pb2`, which only resolves if the output
# directory is on sys.path. Rewrite it to a package-relative import so the
# stubs work as a normal submodule.
sed -i 's/^import inference_pb2/from . import inference_pb2/' \
  "${OUT_DIR}/inference_pb2_grpc.py"

echo "generated stubs in ${OUT_DIR}"
