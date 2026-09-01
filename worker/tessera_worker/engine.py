"""Engine selection and lifecycle.

A single indirection point so that later phases can swap the execution backend
(paged KV cache, speculative decoding) without touching the gRPC or HTTP
layers. "reference" is the dense engine that defines correctness; "paged"
adds the Phase 2 block cache and "speculative" the Phase 3 draft-and-verify
loop. Both are asserted to match the reference token for token.
"""

from __future__ import annotations

import logging
import os

from .config import WorkerConfig
from .model import ReferenceEngine
from .paged_engine import PagedEngine
from .speculative import SpeculativeEngine

logger = logging.getLogger(__name__)

BACKENDS = {
    "reference": ReferenceEngine,
    "paged": PagedEngine,
    "speculative": SpeculativeEngine,
}
DEFAULT_BACKEND = "reference"


class EngineHandle:
    """Owns the engine instance and reports readiness.

    Loading is explicit rather than lazy so that a model that fails to load
    kills the process at startup instead of failing the first request.
    """

    def __init__(self, config: WorkerConfig, backend: str | None = None) -> None:
        self.config = config
        self.backend = backend or os.getenv("TESSERA_BACKEND", DEFAULT_BACKEND)
        if self.backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {self.backend!r}; expected one of {sorted(BACKENDS)}"
            )
        self._engine: ReferenceEngine | None = None

    def load(self) -> None:
        if self._engine is not None:
            return
        logger.info("initialising backend=%s", self.backend)
        self._engine = BACKENDS[self.backend](self.config)

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def engine(self) -> ReferenceEngine:
        if self._engine is None:
            raise RuntimeError("engine accessed before load()")
        return self._engine
