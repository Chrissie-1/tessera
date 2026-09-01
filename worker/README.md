# tessera-worker

The Python inference worker for [Tessera](https://github.com/Chrissie-1/tessera):
a paged KV cache, a continuous-batching scheduler, and exact speculative
decoding, all validated against a deliberately slow dense reference engine that
defines correctness.

Install:

```bash
pip install -e ".[dev]"
```

Run the gRPC worker:

```bash
python -m tessera_worker.server
```

Configuration is read from the environment (`TESSERA_MODEL`, `TESSERA_BACKEND`,
`TESSERA_DEVICE`, …). See the
[project README](https://github.com/Chrissie-1/tessera#configuration) for the
full table, the HTTP/gRPC API reference, and the current limitations.

The gRPC stubs in `tessera_worker/generated/` are not committed — generate them
with `../scripts/gen_proto.sh` after cloning.

MIT licensed.
