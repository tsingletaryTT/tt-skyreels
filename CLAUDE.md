# tt-skyreels — project log

Packages SkyReels-V2-DF-1.3B-540P (text-to-video) as a tt-model-manager v5.1 CONTAINER,
following the `tt-dit-server` pattern established by `tenstorrent/tt-animatediff`.

## 2026-09-09 — initial bring-up and packaging

**Prompt:** "Use the agentic model bring up process to bring the SkyReels family of models
I ported to ~/code/tt-local-generator to models served and packaged and registered with
tt-model-manager. Use the v5.1 format." Scoped down to T2V only (SkyReels-V2-DF-1.3B-540P);
I2V (SkyReels-V2-I2V-14B-540P) is a follow-up. User explicitly asked for a dedicated repo
decoupled from tt-metal (not injected into a tt-metal checkout, unlike the flux2-dev-qb2
precedent on the Hub) — same shape as tt-animatediff.

### What existed already

SkyReels-V2-DF-1.3B-540P was already ported and working in `~/code/tt-local-generator`, but
wired into tt-inference-server's "media engine" runner architecture
(`patches/tt_dit/pipelines/skyreels_v2/pipeline_skyreels.py` +
`patches/media_server_config/tt_model_runners/skyreels_runner.py`), not the standalone ASGI
shape `kind: tt-dit-server` needs. This repo ports the pipeline out and adds the serving
glue tt-animatediff's `animatediff_ttnn/server/app.py` already solved.

### What this repo adds

- `skyreels_ttnn/pipeline_skyreels.py` — the ported pipeline, with `models.tt_dit`/`ttnn`
  imports moved out of module scope (see the bug below).
- `skyreels_ttnn/server/app.py` — new. FastAPI app, 2×2 QB2 mesh, MP4 output via imageio.
- `tt_model_package.yaml` — v5.1 manifest.
- `tests/test_server_app.py` — CPU-only, no hardware needed.

### Real bugs found only by actually serving on hardware

None of these were caught by `tt-model package`'s build-time `verify` step — it imports the
ASGI app and the pipeline module on a machine with no card, which never exercises weight
loading or a real denoise pass. Each was found by an actual `/v1/videos/generations` call:

1. **Module-scope `ttnn` import.** The original hotpatch imported `models.tt_dit.*` (and
   therefore `ttnn`) at module scope, for type hints alone. Broke the tt-dit-server
   contract, and an unleashed `import skyreels_ttnn.pipeline_skyreels` during MY OWN
   bring-up work crashed hitting ttnn's C extension on a box with real chips attached — a
   direct hit of the exact hazard `~/CLAUDE.md` warns about. Fixed by deferring every
   `models.tt_dit`/`ttnn` import to point-of-use.
2. **`ModuleNotFoundError: pytest`** — `models/common/utility_functions.py` imports
   `pytest` at module scope, reached transitively through `models.tt_dit.layers.linear`.
   A known tt-metal quirk (also called out by name in the tt-model-package-test skill's own
   troubleshooting notes). Fixed by adding `pytest` to `runtime.packages`.
3. **Missing `.device` on the TTNN transformer wrapper.** `diffusers.DiffusionPipeline`'s
   `_execution_device`/`.device` properties read `.device` off every registered
   `torch.nn.Module` component; plain `nn.Module` has none. Raised a confusing
   `AttributeError: '...Pipeline' object has no attribute '_execution_device'` — Python's
   property protocol converts an `AttributeError` raised INSIDE a getter into "not found",
   which then fell through to `ConfigMixin.__getattr__` and reported the wrong name
   entirely. Fixed by adding a `device` property returning `torch.device("cpu")`
   (a placeholder; real compute is on the TTNN mesh).
4. **`NameError: ftfy`** — `diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2`'s prompt
   cleaning calls `ftfy.fix_text()` with no `ImportError` guard in the installed diffusers
   version. Fixed by adding `ftfy` to `runtime.packages`.
5. **Timestep dtype mismatch.** `SkyReelsTTNNTransformer.forward()` forwarded diffusers'
   raw int64 timestep straight through; `WanTimestepsEmbedding` asserts its input dtype
   matches its own (`DataType.FLOAT32`). The docstring claimed the TTNN model wanted a bare
   PyTorch tensor — never exercised until this bring-up's first real denoise attempt. Fixed
   by mirroring `models.tt_dit.pipelines.wan.pipeline_wan`'s own call site: cast to
   float32, reshape `(B,) -> (B,1,1,1)`, convert via `float32_tensor(...)`.

### Also: rebuilt the box's tt-metal to the latest release

`/home/ttuser/tt-metal` (the checkout actually on `.tenstorrent-venv`'s PYTHONPATH — a
different, much older/stale checkout also exists at `~/code/tt-metal`, unused) was on
v0.77.0. Locked to `v0.78.0` (latest release tag on origin at the time) and rebuilt. Hit a
stale `BUILD_TT_TRAIN=ON` in the cached `CMakeCache.txt` from a previous build, which broke
against v0.78.0's tt-train example CMakeLists (`PRECOMPILE_HEADERS_REUSE_FROM` referencing a
target that no longer resolved). Fixed with `cmake -DBUILD_TT_TRAIN=OFF .` on the existing
build dir rather than a full clean rebuild.

### Verified, on hardware (P300×2, QB2 2×2 mesh)

- `tt-model package --container tt_model_package.yaml --out ~/tt-model-builds` — clean
  build, in-image `verify` passes.
- `tt-model serve` — mesh opens (4 chips), weights load from the existing HF cache
  (`~/.cache/huggingface`, bind-mounted into the container at `/hf`), server reaches
  `Application startup complete`.
- `curl .../v1/models` — reports the correct weights id.
- `curl .../v1/videos/generations` — TWO real generations, both HTTP 200 with valid H.264
  MP4 output confirmed via `ffprobe` (480×272 @ 24fps): a 9-frame/8-step debug shape, and
  the server's real default (33 frames, 8 steps).

### Not yet done

- I2V (SkyReels-V2-I2V-14B-540P) — separate follow-up, same pattern.
- Push to HF (`tsingletary/tt-skyreels`) — not yet done; needs explicit go-ahead before
  publishing anywhere.
- No GitHub remote yet — this repo is local-only. `tt_model_package.yaml`'s `extra_code`
  currently points at a local path; switch to a `{repo, ref}` pin once pushed.

## 2026-09-09 — Gradio app + discolike manifest

Added `app.py` (Gradio UI) and `.disco/app.yaml`, matching tt-animatediff's own pair of
files. Along the way, extracted the device/pipeline open-once-per-process logic that used
to live only inside `skyreels_ttnn/server/app.py`'s ASGI lifespan into
`skyreels_ttnn/session.py` (mirroring `animatediff_ttnn/session.py`), so the ASGI server
and the Gradio app share one implementation of "open the mesh, load the pipeline, keep it
open" instead of two copies that can drift. Re-verified the served path end-to-end after
this refactor (rebuild → serve → `/v1/videos/generations` → HTTP 200) since it changed
`server/app.py`'s lifespan, not just additive new files.

No cpu/sim mode selector, unlike tt-animatediff's `app.py` — this model has only ever run
on real Blackhole hardware (2×2 QB2 mesh), so offering modes that don't work would save
nobody anything. No per-step preview streaming either: `SkyReelsPipeline.__call__` has no
`on_step` hook the way `generate_frames_temporal` does, so there is nothing to stream
mid-denoise; the UI yields a "starting" status immediately instead so it doesn't look
frozen during first-request kernel compilation.

Verified: `app.py` imports cleanly with no card (no `ttnn` in `sys.modules`), the Gradio
`Blocks` graph builds, and `demo.launch()` actually serves an HTTP 200 on `:7861` (not
re-verified: an actual Generate click through the UI — the ASGI path's equivalent request
was re-verified instead, since both now go through the same `skyreels_ttnn.session`).

## 2026-09-09 — pushed to GitHub and HF; registered with tt-model-manager

- Code: [github.com/tsingletaryTT/tt-skyreels](https://github.com/tsingletaryTT/tt-skyreels)
  (private, `main` as default branch — the repo was initially created with `master` by an
  oversight in an earlier `git init`; renamed, re-pushed, default branch changed, old
  `master` deleted, per the house "default branch is always `main`" rule).
- Package: `episod/tt-skyreels` on HF (private). **Not** `tsingletary/tt-skyreels` as
  earlier drafts of this file and the manifest said — `hf auth whoami` resolves to
  `episod` (a Tenstorrent org member), not `tsingletary`; caught before the push would
  have failed on a permissions error, and fixed in `tt_model_package.yaml`'s `repo:`
  field before pushing.
- `extra_code.root` now pins `https://github.com/tsingletaryTT/tt-skyreels` at a real
  commit sha (reachable from `main`), replacing the local-path root used during bring-up.
  Rebuilt and confirmed the git-clone path works (`built.tt_metal.mode: "git"`, `dirty:
  false`) before pushing to HF.
- **Registered, confirmed**: `tt-model info episod/tt-skyreels` resolves the pushed
  manifest and reports "✓ compatible with the local environment". Pushed **private**
  (the tool's own default) — not `--public`, not `--publish` (community catalog) — per
  the tt-model-package-test skill's own caution not to add those on judgment alone.
- I2V (SkyReels-V2-I2V-14B-540P) is still a separate, not-yet-started follow-up.

## 2026-09-09 — README, LICENSE, and a proper model card

Added a real `README.md` + `LICENSE` (Apache-2.0, matching the SPDX headers already in
every source file) to the GitHub repo, and a `card.description`/`card.quickstart` block
to `tt_model_package.yaml` so the HF-generated README actually says what the model is
instead of just the bare pull/serve commands. Rebuilt and re-pushed — confirmed via a
fresh `hf_hub_download('episod/tt-skyreels', 'README.md')` that the new description and
quickstart are live on the Hub.
