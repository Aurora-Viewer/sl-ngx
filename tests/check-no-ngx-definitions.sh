#!/usr/bin/env bash
# NGX's declarations come from the user's SDK at build time: this repository
# must never define an NVSDK_NGX_* type, constant, function or macro itself.
set -euo pipefail
cd "$(dirname "$0")/.."
hits=$(grep -nE '^\s*(#\s*define|typedef|struct|enum|class|using)\s+(NVSDK_NGX_|NGX_)' -r src include tests || true)
hits+=$(grep -nE '^\s*[A-Za-z_][A-Za-z0-9_ \*]*\bNVSDK_NGX_[A-Za-z0-9_]+\s*\([^;]*\)\s*\{' -r src || true)
if [[ -n $hits ]]; then
    echo "NVSDK_NGX_ definitions in the repository:"; echo "$hits"; exit 1
fi
echo "no NVSDK_NGX_ definitions"
