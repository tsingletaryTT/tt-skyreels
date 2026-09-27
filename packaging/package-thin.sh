#!/usr/bin/env bash
# Stage the episod/tt-skyreels v6 thin bundle -- the ONE recipe a repackage uses.
#
# Everything the published bundle is built from is in this directory, so the recipe is
# reproducible from a checkout instead of from a bring-up log:
#   packaging/requirements.txt   the bundle's pip pins (ttnn, the two wheels, runtime deps)
#   packaging/kernel_patch/      3 fabric kernel sources the ttnn 0.78.0 wheel omits
#                                (byte-identical to tt-metal v0.78.0; see its README.md)
#   this script                  the exact package-thin flags
# tests/test_server_app.py parses this script and packaging/requirements.txt, so a flag or
# pin that drifts from the code (app target, mesh env, weights revision, wheel version)
# fails CI instead of shipping.
#
# Usage:
#   CLOSURE_WHEEL=/path/to/tt_skyreels_models_closure-0.78.0-py3-none-any.whl \
#     packaging/package-thin.sh <out-dir>
#
# This only STAGES. It does not push, because package-thin cannot ship an extra directory:
# a push straight from package-thin would publish a bundle without kernel_patch/, and
# `tt-model push` takes only v5.1 container packages. Verify the staged bundle on hardware,
# then publish <out-dir> as a whole (e.g. `hf upload episod/tt-skyreels <out-dir> .`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
OUT="${1:?usage: package-thin.sh <out-dir>}"
: "${CLOSURE_WHEEL:?set CLOSURE_WHEEL to the tt-skyreels-models-closure wheel (vendored tt_dit)}"
TT_MODEL="${TT_MODEL:-tt-model}"   # override to use a specific tt-model build

# Must match skyreels_ttnn.pipeline_skyreels.PINNED_WEIGHTS_REVISION (a test enforces it).
WEIGHTS_REVISION=958acd63685c7e632e4b194549f2a703e34bd98b

# 1. Build this repo's served-path wheel (skyreels-ttnn, version from setup.py).
WHEEL_DIR="$(mktemp -d)"
(cd "$REPO_ROOT" && uv build --wheel --out-dir "$WHEEL_DIR" >/dev/null)
SERVED_WHEEL="$(ls "$WHEEL_DIR"/skyreels_ttnn-*.whl)"

# 2. Stage the bundle.
"$TT_MODEL" package-thin \
  --out "$OUT" \
  --kind tt-dit-server \
  --app skyreels_ttnn.server.app:app \
  --model-py "$REPO_ROOT/skyreels_ttnn/session.py" \
  --requirements "$HERE/requirements.txt" \
  --models-wheel "$SERVED_WHEEL" \
  --models-wheel "$CLOSURE_WHEEL" \
  --no-vllm \
  --arch blackhole \
  --weights Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers \
  --weights-revision "$WEIGHTS_REVISION" \
  --tt-metal-version 0.78.0 \
  --mesh QB2 \
  --device-count 4 \
  --env SKYREELS_MESH_SHAPE=2x2 \
  --env 'TT_METAL_KERNEL_PATH=$HERE/kernel_patch' \
  --name tt-skyreels

# 3. Ship the kernel patch that TT_METAL_KERNEL_PATH points at.
cp -r "$HERE/kernel_patch" "$OUT/kernel_patch"
