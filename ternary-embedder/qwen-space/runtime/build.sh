#!/bin/sh
# Build libtde.so: baseline x86-64-v3 (AVX2/FMA/F16C), AVX-512 VNNI kernel in its own translation unit,
# chosen at load time. No OpenMP; no libstdc++ dependency (plain C allocation); libgcc linked statically.
set -e
cd "$(dirname "$0")"
OUT=${1:-build}
mkdir -p "$OUT"
CXX=${CXX:-g++}
FLAGS="-O3 -march=x86-64-v3 -fPIC -fvisibility=hidden -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-math-errno -std=c++17 -Wall"
$CXX $FLAGS -c src/tde.cpp -o "$OUT/tde.o"
$CXX $FLAGS -c src/lin_avx2.cpp -o "$OUT/lin_avx2.o"
$CXX $FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -c src/lin_vnni.cpp -o "$OUT/lin_vnni.o"
${CC:-gcc} -shared -o "$OUT/libtde.so" "$OUT/tde.o" "$OUT/lin_avx2.o" "$OUT/lin_vnni.o" \
  -static-libgcc -Wl,--version-script=src/exports.map -Wl,-z,defs -lm
echo "built $OUT/libtde.so"
