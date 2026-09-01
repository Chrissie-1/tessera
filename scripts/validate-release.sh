#!/usr/bin/env bash
# Pre-release checks. Run before tagging: `make validate-release`.
#
# This is a fast sanity pass over metadata and tooling. It does NOT run the
# test suites -- that is `make test`, and the release workflow runs them again
# on a clean checkout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'

ERRORS=0
pass() { printf '%s✓%s %s\n' "$GREEN" "$NC" "$1"; }
fail() { printf '%s✗%s %s\n' "$RED" "$NC" "$1"; ERRORS=$((ERRORS + 1)); }
warn() { printf '%s⚠%s %s\n' "$YELLOW" "$NC" "$1"; }

echo "Tessera pre-release validation"
echo "============================="
echo

echo "1. Required files"
for file in README.md CHANGELOG.md LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md \
            SECURITY.md RELEASING.md worker/README.md; do
    if [ -f "$file" ]; then pass "$file"; else fail "$file is missing"; fi
done
echo

echo "2. Version consistency"
CARGO_VERSION=$(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)
PY_VERSION=$(grep -m1 '^version = ' worker/pyproject.toml | cut -d'"' -f2)
if [ -z "$CARGO_VERSION" ] || [ -z "$PY_VERSION" ]; then
    fail "could not parse a version from Cargo.toml and/or worker/pyproject.toml"
elif [ "$CARGO_VERSION" = "$PY_VERSION" ]; then
    pass "Cargo.toml and pyproject.toml agree: $CARGO_VERSION"
else
    fail "version mismatch: Cargo.toml=$CARGO_VERSION pyproject.toml=$PY_VERSION"
fi
echo

echo "3. Changelog"
if [ -n "$CARGO_VERSION" ] && grep -q "^## \[${CARGO_VERSION}\]" CHANGELOG.md; then
    pass "CHANGELOG.md has a section for $CARGO_VERSION"
else
    fail "CHANGELOG.md has no '## [$CARGO_VERSION]' section"
fi
echo

echo "4. Git state"
if [ -z "$(git status --porcelain)" ]; then
    pass "working tree is clean"
else
    warn "uncommitted changes:"
    git status --short | sed 's/^/    /'
fi

if git describe --tags --abbrev=0 >/dev/null 2>&1; then
    pass "previous tag: $(git describe --tags --abbrev=0)"
else
    warn "no previous release tags"
fi

if git rev-parse "v${CARGO_VERSION}" >/dev/null 2>&1; then
    fail "tag v${CARGO_VERSION} already exists"
else
    pass "tag v${CARGO_VERSION} is free"
fi
echo

echo "5. Toolchain"
# `command -v` is not enough on Windows, where an App Execution Alias shadows
# python3 with a stub that only advertises the Microsoft Store.
if python3 -c 'import sys; print(sys.version.split()[0])' >/dev/null 2>&1; then
    pass "python3: $(python3 -c 'import sys; print(sys.version.split()[0])')"
else
    fail "python3 not found (or not a working interpreter)"
fi

if command -v cargo >/dev/null 2>&1; then
    pass "cargo: $(cargo --version | awk '{print $2}')"
else
    fail "cargo not found"
fi

if command -v protoc >/dev/null 2>&1; then
    pass "protoc: $(protoc --version | awk '{print $2}')"
else
    warn "protoc not found; the gateway build compiles the proto and will fail"
fi
echo

echo "6. Generated stubs"
if [ -f worker/tessera_worker/generated/inference_pb2.py ]; then
    pass "Python gRPC stubs present"
else
    warn "gRPC stubs missing; run 'make proto'"
fi
echo

echo "7. Formatting and lint"
run_check() {  # name, command...
    local name=$1; shift
    if ! command -v "$1" >/dev/null 2>&1; then warn "$1 not installed; skipped $name"; return; fi
    if "$@" >/dev/null 2>&1; then pass "$name"; else fail "$name failed"; fi
}
run_check "black --check"  black --check worker
run_check "ruff check"     ruff check worker
run_check "cargo fmt"      cargo fmt --all -- --check
run_check "cargo clippy"   cargo clippy --all-targets -- -D warnings
echo

echo "8. Undeclared-dependency spot check"
# A dependency nobody imports is a promise the package does not keep.
missing=0
while read -r dep; do
    module=${dep//-/_}
    # uvicorn is invoked as a console script by `make run-api`, never imported.
    case "$dep" in uvicorn) continue ;; esac
    if ! grep -rqi --include='*.py' "\\b${module}\\b" worker/tessera_worker; then
        warn "declared but never imported: $dep"
        missing=1
    fi
done < <(sed -n '/^dependencies = \[/,/^\]/p' worker/pyproject.toml \
         | grep -o '"[a-zA-Z0-9_-]*' | tr -d '"' | grep -v '^$')
[ "$missing" -eq 0 ] && pass "every declared dependency is imported somewhere"
echo

echo "============================="
if [ "$ERRORS" -eq 0 ]; then
    printf '%sAll checks passed.%s\n\n' "$GREEN" "$NC"
    echo "Next: make test, then follow RELEASING.md to tag v${CARGO_VERSION}."
    exit 0
fi
printf '%s%d check(s) failed.%s\n' "$RED" "$ERRORS" "$NC"
exit 1
