# Build stage: needs protoc because tonic-build compiles the proto at build time.
FROM rust:1.83-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first so dependency compilation is cached independently of
# gateway source edits.
COPY Cargo.toml Cargo.lock ./
COPY gateway/Cargo.toml gateway/Cargo.toml
RUN mkdir -p gateway/src \
    && echo "fn main() {}" > gateway/src/main.rs \
    && cargo build --release --package tessera-gateway \
    && rm -rf gateway/src

COPY proto proto
COPY gateway gateway
# Touch so cargo does not reuse the stub main.rs fingerprint.
RUN touch gateway/src/main.rs && cargo build --release --package tessera-gateway

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 tessera

COPY --from=builder /build/target/release/tessera-gateway /usr/local/bin/tessera-gateway

USER tessera
EXPOSE 8080
ENV TESSERA_GATEWAY_PORT=8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/tessera-gateway"]
