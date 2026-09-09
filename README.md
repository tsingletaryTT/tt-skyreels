# tt-skyreels

[SkyReels-V2-DF-1.3B-540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
(text-to-video) on Tenstorrent Blackhole via TTNN, packaged as a
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) v5.1 **CONTAINER**
package. Pull it with just Docker + a TT card — no host tt-metal install required:

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
- **Hardware**: Tenstorrent Blackhole, P300×2 board as a 2×2 (`QB2`) mesh — 4 chips
- **Weights**: [`Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
  (a pointer — never embedded in the package; downloaded to your own HF cache)

## Repo layout

| Path | What |
| --- | --- |
| `skyreels_ttnn/pipeline_skyreels.py` | The ported TTNN pipeline (`SkyReelsPipeline`, `SkyReelsTTNNTransformer`) |
| `skyreels_ttnn/session.py` | Mesh-device + pipeline singleton, shared by the ASGI server and the Gradio app |
| `skyreels_ttnn/server/app.py` | The ASGI app tt-model-manager's `tt-dit-server` kind serves |
| `app.py` | A local Gradio UI (`pip install -e ".[ui]"` then `python app.py`, port 7861) |
| `.disco/app.yaml` | [tt-discolike](https://github.com/tsingletaryTT/tt-discolike) catalog manifest |
| `tt_model_package.yaml` | The tt-model-manager v5.1 manifest this package is built from |
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
pytest tests/ -q          # 48 tests, pure CPU, no card or tt-metal needed
```

To rebuild and re-verify the actual container package (needs a tt-model-manager
checkout and real hardware):

```bash
tt-model package --container tt_model_package.yaml --out ~/tt-model-builds
tt-model serve ~/tt-model-builds/tt-skyreels/tt_kernel_manifest.json
```

See [CLAUDE.md](CLAUDE.md) for the full bring-up log, including every bug found only by
actually serving a generation on hardware (a module-scope `ttnn` import, missing
`pytest`/`ftfy` runtime deps, a missing `.device` shim for diffusers' generic pipeline
introspection, and a timestep dtype mismatch) — none of them caught by tt-model-manager's
build-time `verify` step, which never opens a device.

## Status

- ✅ Built, served, and verified on real hardware: `/v1/videos/generations` returns valid
  MP4 at both a minimal debug shape (9 frames/8 steps) and the server's real default (33
  frames/8 steps).
- ⏳ SkyReels-V2-I2V-14B-540P (image-to-video, the other member of the SkyReels family) —
  not yet started, same pattern expected to apply.

## License

Apache 2.0 (matching the upstream `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers` weights'
license terms — see the weights repo for details).
