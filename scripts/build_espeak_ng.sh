#!/usr/bin/env bash
# build_espeak_ng.sh: build espeak-ng 1.52.0 from source into the workspace (OMNI login node).
#
# Usage:   bash scripts/build_espeak_ng.sh        (setup_omni_env.sh runs it)
#
# Kokoro's text front end (misaki) gives words outside its lexicon to espeak-ng. The
# library bundled with espeakng-loader 0.2.4 was built on Ubuntu 20.04 and needs glibc
# 2.29, but OMNI runs Rocky Linux 8 with glibc 2.28, so it cannot load there. This script
# builds the same espeak-ng commit with the same CMake options as that package, using the
# system compilers, so that the library needs only OMNI's own C and C++ runtimes.
# listening_beyond_words.espeak points misaki at it.
#
# Needs   LBW_WORKSPACE; git, /usr/bin/gcc and /usr/bin/g++, the cmake/3.19.4 module, and
#         access to GitHub (the source, and the sonic library that it fetches at a pinned commit)
# Writes  $LBW_WORKSPACE/env/espeak-ng/                the library (lib/) and its data (share/)
#         $LBW_WORKSPACE/env/espeak-ng-build.<random>/ the source and build tree, kept
#
# Safe to re-run: it does nothing once the pinned commit is installed.
set -euo pipefail
umask 077

# Tag 1.52.0, the commit that espeakng-loader 0.2.4 bundles.
ESPEAK_NG_TAG="1.52.0"
ESPEAK_NG_COMMIT="4870adfa25b1a32b4361592f1be8a40337c58d6c"
: "${LBW_WORKSPACE:?LBW_WORKSPACE is not set; see docs/engineering_conventions.md}"
prefix="$LBW_WORKSPACE/env/espeak-ng"
stamp="$prefix/built-from-$ESPEAK_NG_COMMIT"

if [ -f "$stamp" ]; then
  echo "espeak-ng $ESPEAK_NG_TAG is already built in $prefix"
  exit 0
fi

build="$(mktemp -d "$LBW_WORKSPACE/env/espeak-ng-build.XXXXXX")"
echo "building espeak-ng $ESPEAK_NG_TAG in $build ..."
git clone --quiet --depth 1 --branch "$ESPEAK_NG_TAG" \
  https://github.com/espeak-ng/espeak-ng.git "$build/src"
if [ "$(git -C "$build/src" rev-parse HEAD)" != "$ESPEAK_NG_COMMIT" ]; then
  echo "tag $ESPEAK_NG_TAG no longer points to $ESPEAK_NG_COMMIT; stop and check" >&2
  exit 1
fi

set +u  # the module command reads unset variables
module load cmake/3.19.4
set -u
# The options of espeakng-loader 0.2.4 (build.sh and build.yml), except that the system
# compilers are named and intonations are compiled in the same build as the library.
cmake -S "$build/src" -B "$build/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS=-pthread -DCMAKE_EXE_LINKER_FLAGS=-pthread \
  -DBUILD_SHARED_LIBS=ON -DENABLE_TESTS=OFF -DCOMPILE_INTONATIONS=ON \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build "$build/build" --parallel 4
cmake --install "$build/build"

library="$prefix/lib/libespeak-ng.so"
echo "glibc and C++ runtime versions the library needs:" \
  "$(objdump -T "$library" | grep -oE 'GLIBC(XX)?_[0-9.]+' | sort -uV | tr '\n' ' ')"
# The installed program uses the new library; it must read an unknown word aloud.
sounds="$("$prefix/bin/espeak-ng" --path="$prefix/share" -q --ipa -v en-us "zorblefax")"
if [ -z "$sounds" ]; then
  echo "espeak-ng gave no sounds for a test word" >&2
  exit 1
fi
echo "espeak-ng $ESPEAK_NG_TAG reads 'zorblefax' as: $sounds"
touch "$stamp"
echo "OK: espeak-ng $ESPEAK_NG_TAG is in $prefix"
