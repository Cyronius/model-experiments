#!/usr/bin/env bash
# Builds the C library and runs capi/tests/smoke.c against it. Run from app/.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release -p titler-capi
out=$(mktemp -d)
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    out=$(cygpath -m "$out")
    # Needs a Visual Studio developer shell (cl.exe and the Windows headers on the path).
    cl -nologo -Icapi/include -DTITLER_DLL_IMPORT capi/tests/smoke.c target/release/titler_c.dll.lib -Fe:"$out/smoke.exe" -Fo:"$out/"
    cp target/release/titler_c.dll "$out/"
    "$out/smoke.exe" model/titler-v2-q4_k_m.gguf ;;
  Darwin)
    cc -Icapi/include capi/tests/smoke.c -Ltarget/release -ltitler_c -o "$out/smoke"
    DYLD_LIBRARY_PATH=target/release "$out/smoke" model/titler-v2-q4_k_m.gguf ;;
  *)
    cc -Icapi/include capi/tests/smoke.c -Ltarget/release -ltitler_c -o "$out/smoke"
    LD_LIBRARY_PATH=target/release "$out/smoke" model/titler-v2-q4_k_m.gguf ;;
esac
