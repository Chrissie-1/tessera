# Security Policy

## Reporting a vulnerability

Please report security issues through
[GitHub's private vulnerability reporting](https://github.com/Chrissie-1/tessera/security/advisories/new)
rather than opening a public issue.

This is a personal side project maintained in spare time. There is no SLA and
no guaranteed response window — please do not rely on one. If a report goes
unanswered and you believe the issue is being actively exploited, you are free
to disclose it publicly.

## Supported versions

Only the latest release receives fixes. There are no backports.

## Scope

Tessera is a learning and benchmarking project. It ships with **no
authentication, no TLS, and no rate limiting** beyond a fixed in-flight cap per
worker, and it has not been hardened or audited for untrusted input. Treat both
the gateway and the worker as trusted-network services: do not expose them
directly to the public internet.

Given that, the following are known and documented, not vulnerabilities:

- Unauthenticated access to the HTTP and gRPC endpoints.
- Plaintext gRPC between gateway and worker.
- Resource exhaustion from concurrent requests beyond the shedding threshold.
- Arbitrary code execution from loading an untrusted model — `TESSERA_MODEL` is
  passed to `transformers`, which will execute repository code for
  architectures that require it. Only point it at models you trust.

Reports that *are* in scope: a way to make the worker or gateway serve another
request's data, escape the request path, or crash the process from a
well-formed API request within the documented limits.
