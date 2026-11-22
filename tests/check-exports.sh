#!/usr/bin/env bash
# The library must export Streamline's sl* entry points and nothing else.
set -euo pipefail
lib=${1:-libsl_ngx.so}
bad=$(nm -D --defined-only "$lib" | awk '{print $3}' | grep -v '^sl' || true)
if [[ -n $bad ]]; then
    echo "exports beyond sl*:"; echo "$bad"; exit 1
fi
echo "exports: $(nm -D --defined-only "$lib" | awk '{print $3}' | grep -c '^sl') sl* symbols, nothing else"
