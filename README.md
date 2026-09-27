# tt-skyreels

[SkyReels-V2-DF-1.3B-540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
(text-to-video) on Tenstorrent Blackhole via TTNN, packaged as a
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) v6 **thin** bundle
(a pip/venv install, not a container image):

```bash
tt-model pull episod/tt-skyreels --with-weights
tt-model serve episod/tt-skyreels
curl localhost:20000/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red bicycle leaning against a brick wall in soft morning light"}'
```

Package: [episod/tt-skyreels](https://huggingface.co/episod/tt-skyreels) on the Hub.

## What this is

SkyReelsV2's transformer is weight-compatible with `WanTransformer3DModel`
(`models.tt_dit` in [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)), so
this repo reuses that TTNN model wholesale rather than shipping its own kernels — see
[`skyreels_ttnn/pipeline_skyreels.py`](skyreels_ttnn/pipeline_skyreels.py)'s module
docstring for the exact weight-key compatibility mapping. What this repo *does* add is
the serving glue: a small ASGI app (`kind: tt-dit-server` in tt-model-manager's terms,
following the pattern [tenstorrent/tt-animatediff](https://github.com/tenstorrent/tt-animatediff)
established) that wraps the pipeline behind an HTTP surface, plus the fixes needed to
actually run a real generation end to end (see [CLAUDE.md](CLAUDE.md) for the specific
bugs found and fixed during bring-up — none of them were in the reused TTNN model itself).

- **Model**: SkyReels-V2-DF-1.3B-540P, text-to-video, 480×272 @ 24fps
- **Hardware**: Tenstorrent Blackhole, 4 chips as a 2×2 (`QB2`) mesh: two P300c boards,
  i.e. a QuietBox 2. That is the only mesh the published bundle ships and the only one run.
- **Weights**: [`Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
  (a pointer — never embedded in the package; downloaded to your own HF cache). Since
  `skyreels-ttnn` 0.1.1 every component loads from pinned revision `958acd6`; set
  `TT_MODEL_WEIGHTS_REVISION` to override. Licensed separately from this repo: see
  [License](#license).

### Known divergences from the upstream model

These are documented rather than fixed. Neither has been measured against a reference:
there is no PCC and no side-by-side comparison. The only quality evidence is visual
inspection of generated clips.

- **Plain T2V, not diffusion forcing.** The checkpoint is published for
  `SkyReelsV2DiffusionForcingPipeline` (the "DF": per-frame noise levels and
  autoregressive long-video extension). This port builds the plain `SkyReelsV2Pipeline`
  instead, so each request produces one fixed-length clip with a shared timestep and no
  AR extension.
- **No fps conditioning.** The checkpoint's transformer sets `inject_sample_info: true`,
  so the reference adds an fps-embedding term to its timestep projection. The reused TTNN
  WAN transformer has no such term: the 5 `fps_embedding.*` / `fps_projection.*` tensors
  are dropped at load (logged as "Ignored SkyReels-only keys"), and the `fps` argument is
  ignored.

## Repo layout

| Path | What |
| --- | --- |
| `skyreels_ttnn/pipeline_skyreels.py` | The ported TTNN pipeline (`SkyReelsPipeline`, `SkyReelsTTNNTransformer`) |
| `skyreels_ttnn/session.py` | Mesh-device + pipeline singleton, shared by the ASGI server and the Gradio app |
| `skyreels_ttnn/server/app.py` | The ASGI app tt-model-manager's `tt-dit-server` kind serves |
| `app.py` | A local Gradio UI (`pip install -e ".[ui]"` then `python app.py`, port 7861) |
| `.disco/app.yaml` | [tt-discolike](https://github.com/tsingletaryTT/tt-discolike) catalog manifest |
| `setup.py` | Builds `skyreels-ttnn`, the wheel `tt-model package-thin` ships as the served-path closure |
| `tests/` | CPU-only test suite — no hardware or tt-metal needed to run it |

## Running it

**Via tt-model-manager** (the packaged, hardware-verified path):

```bash
tt-model pull episod/tt-skyreels --with-weights
tt-model serve episod/tt-skyreels
```

**Gradio UI**, locally on a box with the 2×2 QB2 mesh free:

```bash
pip install -e ".[ui]"
python app.py    # http://localhost:7861
```

**Via [tt-discolike](https://github.com/tsingletaryTT/tt-discolike)**, if this repo is
checked out under a scanned root: it shows up in the catalog automatically via
`.disco/app.yaml`.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
pytest tests/ -q          # pure CPU, no card needed; never imports ttnn
```

To rebuild the actual v6 thin bundle (needs a tt-model-manager checkout, a tt-metal
source tree at v0.78.0 for the `models/tt_dit` closure, and real hardware to verify):
build `skyreels-ttnn` from this repo's own `setup.py`, vendor `models/tt_dit` +
`models/common/{utility_functions.py,device_utils.py,modules/tt_ccl.py}` into a second
wheel, then `tt-model package-thin --kind tt-dit-server --app skyreels_ttnn.server.app:app
--models-wheel <both wheels> ...`. See [CLAUDE.md](CLAUDE.md)'s 2026-09-15 entry for the
exact recipe, including the `kernel_patch/` workaround the published bundle needs for a
real ttnn 0.78.0 PyPI wheel gap (three missing fabric kernel source files).

See [CLAUDE.md](CLAUDE.md) for the full bring-up log, including every bug found only by
actually serving a generation on hardware (a module-scope `ttnn` import, missing
`pytest`/`ftfy` runtime deps, a missing `.device` shim for diffusers' generic pipeline
introspection, and a timestep dtype mismatch) — none of them caught by tt-model-manager's
build-time `verify` step, which never opens a device.

## Status

- ✅ Built, served, and verified on real hardware: `/v1/videos/generations` returns valid
  MP4 at a minimal debug shape (9 frames, 8 steps) and at the server's default clip
  length (33 frames), which bring-up ran with `num_inference_steps: 8` passed explicitly.
  The request default is **20 steps** (`DEFAULT_NUM_INFERENCE_STEPS` in
  `skyreels_ttnn/server/app.py`); it has never been 8. Measured 2026-09-27 on the
  published bundle: about 58 s per 33-frame clip at 20 steps and about 54 s at 8 steps.
  The TT denoise loop is only about 4 s of that; the rest is the CPU text encoder and VAE.
- ⏳ SkyReels-V2-I2V-14B-540P (image-to-video, the other member of the SkyReels family) —
  not yet started, same pattern expected to apply.

## License

Two different licenses apply, and they are not the same:

- **This repo's code** (the port, serving glue and tests) is Apache-2.0: see
  [`LICENSE`](LICENSE) and the SPDX headers. The vendored tt-metal `models/tt_dit` code
  in the published bundle's closure wheel is Apache-2.0 too.
- **The model weights** are **not** Apache-2.0. They are published under the **Skywork
  Community License** (HF metadata `license: other`, `license_name: skywork-license`):
  see the
  [upstream LICENSE](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers/blob/main/LICENSE).
  This package never embeds or redistributes the weights. They are downloaded from
  Skywork's repo, and using them is subject to Skywork's terms.

An earlier version of this section wrongly said the weights were Apache-2.0.
