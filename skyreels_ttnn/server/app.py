# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""ASGI app serving tt-skyreels (SkyReels-V2-DF-1.3B-540P, T2V), for
tt-model-manager's ``tt-dit-server`` kind.

Modeled directly on tt-animatediff's ``animatediff_ttnn/server/app.py`` (the reference
implementation for this kind). A diffusion transformer has no tokens, no KV cache and no
continuous batching, so vLLM has nothing to do for it. ``tt-dit-server`` installs a small
HTTP stack instead of an engine and launches this module with uvicorn:

    python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on \\
        skyreels_ttnn.server.app:app

TWO CONTRACTS THIS FILE HAS TO HONOUR (same as tt-animatediff)
----------------------------------------------------------------
**1. Readiness is the lifespan.** tt-model-manager decides the server is up when uvicorn
logs ``Application startup complete``, which it prints *after* ASGI lifespan startup
returns. The device open, fabric configuration and weight load (15-30 min cold, 3.5B
params across 4 Blackhole chips over PCIe) belong in the lifespan and nowhere else.

**2. Importing this module must not touch hardware.** ``verify`` lines in the manifest
import the ASGI attribute at image-build time, on a machine with no card. Every ttnn /
models.tt_dit import is therefore inside a function, never at module scope.

ENVIRONMENT, set by the launcher
---------------------------------
``MESH_DEVICE``          the SKU, e.g. ``P300x2`` — informational here
``SKYREELS_MESH_SHAPE``  the resolved shape, e.g. ``2x2`` (manifest points
                         ``runtime.mesh_shape_env`` at this name — the kind's default is
                         FLUX.2's, which means nothing to this model)
``HF_MODEL``             the weights repo id, reported by ``/v1/models``

ONE REQUEST AT A TIME
----------------------
There is no continuous batching to hide behind: the pipeline owns the mesh, and two
concurrent denoise loops would interleave on it. Requests are serialised on a lock and
the blocking work runs in a worker thread so the event loop can still answer ``/health``
while a generation is in flight.
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from contextlib import asynccontextmanager
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#: The env var carrying the resolved mesh shape. Named for THIS model: the kind's default
#: is ``FLUX2_MESH_SHAPE``. A manifest serving this app sets ``runtime.mesh_shape_env`` to
#: the name below.
MESH_SHAPE_ENV = "SKYREELS_MESH_SHAPE"

#: The two mesh shapes SkyReelsPipeline.create_pipeline has actually been exercised on
#: (see pipeline_skyreels.py's module docstring). The pipeline's own tp/sp derivation
#: (``tp_factor = mesh_shape[1]; sp_factor = mesh_shape[0]``) is generic and would accept
#: others, but an unproven shape is exactly the kind of silent-wrong-answer risk
#: ANIMATEDIFF_MESH_SHAPE's validation exists to catch — refuse at the edge instead.
SUPPORTED_MESH_SHAPES = {(1, 4), (2, 2)}

#: Fixed output resolution. SkyReelsRunner (the tt-inference-server precedent this app
#: replaces) always calls the pipeline with these two values; nothing else has been
#: validated against the compiled TTNN graph.
DEFAULT_HEIGHT = 272
DEFAULT_WIDTH = 480

#: 33 frames ≈ 1.4s @ 24fps — the runner's default clip length.
DEFAULT_NUM_FRAMES = 33
DEFAULT_FPS = 24
DEFAULT_GUIDANCE_SCALE = 6.0  # SkyReels recommended: 5-7


class VideoGenerationRequest(BaseModel):
    """OpenAI-shaped video request, matching tt-animatediff's server contract and
    tt-inference-server's ``/v1/videos/generations`` field names, so the same client
    works against either surface.
    """

    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    model: Optional[str] = None
    # Valid counts satisfy (N-1) % 4 == 0: 9, 13, 17, 21, 25, 29, 33, 65, 97, ...
    # Not enforced here beyond the bound -- an invalid count fails inside the denoise
    # loop with a clear diffusers error, which is preferable to guessing at the full set
    # of legal values and rejecting some that would have worked.
    num_frames: int = Field(default=DEFAULT_NUM_FRAMES, ge=1, le=97)
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)
    seed: int = 0
    #: Only b64_json is offered -- a URL response would need a file store this
    #: container doesn't have.
    response_format: str = "b64_json"


class VideoData(BaseModel):
    b64_json: str


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[VideoData]


def mesh_shape_from_env(env: Optional[dict] = None) -> tuple:
    """Parse ``SKYREELS_MESH_SHAPE`` (``"RxC"``) into ``(rows, cols)``.

    Raises on anything not in ``SUPPORTED_MESH_SHAPES`` rather than silently opening
    whatever mesh the string describes: converting SkyReels' TP/SP split against an
    unvalidated shape is a wrong-frames failure, not a loud one, and a malformed or
    oversized value on a shared box would claim chips a neighbour is using.
    """
    raw = (env if env is not None else os.environ).get(MESH_SHAPE_ENV, "").strip()
    if not raw:
        raise ValueError(
            f"{MESH_SHAPE_ENV} must be set to one of "
            f"{sorted('x'.join(map(str, s)) for s in SUPPORTED_MESH_SHAPES)}"
        )
    try:
        rows, cols = (int(p) for p in raw.lower().split("x", 1))
    except ValueError as exc:
        raise ValueError(f"{MESH_SHAPE_ENV}={raw!r} is not a mesh shape like '2x2'") from exc
    if (rows, cols) not in SUPPORTED_MESH_SHAPES:
        raise ValueError(
            f"{MESH_SHAPE_ENV}={raw!r} is not a supported shape -- SkyReelsPipeline has "
            f"only been exercised on {sorted('x'.join(map(str, s)) for s in SUPPORTED_MESH_SHAPES)}"
        )
    return (rows, cols)


def _configure_fabric() -> None:
    """Set Blackhole fabric config before opening the mesh device.

    Mirrors TTSkyReelsRunner._configure_fabric / get_pipeline_device_params
    (tt-inference-server's precedent runner): FABRIC_1D, and on Blackhole a tensix MUX +
    ROW dispatch, which FABRIC_1D_MUX requires.
    """
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.MUX if ttnn.device.is_blackhole() else ttnn.FabricTensixConfig.DISABLED,
    )


def _open_device_and_pipeline(shape: tuple):
    """Claim the mesh, load the SkyReels pipeline. Blocking, hardware, lifespan-only."""
    import ttnn
    from skyreels_ttnn.pipeline_skyreels import SkyReelsPipeline

    _configure_fabric()
    rows, cols = shape
    dispatch_core_config = ttnn.DispatchCoreConfig(
        None,
        ttnn.device.DispatchCoreAxis.ROW if ttnn.device.is_blackhole() else None,
        ttnn.FabricTensixConfig.MUX if ttnn.device.is_blackhole() else None,
    )
    device = None
    try:
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            dispatch_core_config=dispatch_core_config,
        )
        pipeline = SkyReelsPipeline.create_pipeline(mesh_device=device)
    except Exception:
        if device is not None:
            _close_device(device)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
        raise
    return device, pipeline


def _close_device(device) -> None:
    """Release the mesh, swallowing any error -- mirrors animatediff_ttnn.server.app."""
    try:
        import ttnn

        ttnn.close_mesh_device(device)
    except Exception:
        pass


def _frames_to_mp4_b64(frames, fps: int = DEFAULT_FPS) -> str:
    """SkyReelsPipeline output (1, T, H, W, C) float32 in [0, 1] -> base64 MP4.

    Written through a temp file rather than an in-memory buffer: imageio's ffmpeg writer
    pipes to an ffmpeg subprocess that needs a real path to mux into, and this container
    has no volume to leave the file on afterward -- it is read back and removed in the
    same call.
    """
    import numpy as np
    import imageio.v2 as imageio

    frames_uint8 = (np.clip(frames[0], 0.0, 1.0) * 255).astype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        imageio.mimwrite(tmp.name, list(frames_uint8), fps=fps, format="mp4")
        tmp.seek(0)
        data = tmp.read()
    return base64.b64encode(data).decode("ascii")


def _generate(state: dict, req: VideoGenerationRequest):
    """Run the denoise loop. Blocking; called in a worker thread under the device lock."""
    pipeline = state["pipeline"]
    return pipeline(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=req.num_frames,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Claim the mesh and warm the pipeline BEFORE the server reports ready.

    A failure here is deliberately fatal: a server that starts without a device would
    answer /health cheerfully and fail every generation.
    """
    shape = mesh_shape_from_env()
    device, pipeline = await asyncio.to_thread(_open_device_and_pipeline, shape)
    app.state.engine = {
        "device": device,
        "pipeline": pipeline,
        "mesh_shape": shape,
        "hf_model": os.environ.get("HF_MODEL", "unknown"),
        "mesh_device": os.environ.get("MESH_DEVICE", "unknown"),
    }
    app.state.device_lock = asyncio.Lock()
    try:
        yield
    finally:
        _close_device(device)


app = FastAPI(title="tt-skyreels", lifespan=lifespan)


def _readiness() -> dict:
    engine = getattr(app.state, "engine", None)
    return {
        "model_ready": engine is not None,
        "model": (engine or {}).get("hf_model"),
        "mesh_device": (engine or {}).get("mesh_device"),
        "mesh_shape": "x".join(str(n) for n in (engine or {}).get("mesh_shape", ())),
    }


@app.get("/tt-liveness")
async def tt_liveness() -> dict:
    """LIVENESS probe: alive iff this process can respond at all. A model still warming
    is alive, not broken -- restarting it here would kill a load that is progressing
    normally. Lock-free and engine-free so it keeps answering while a generation holds
    the device.
    """
    try:
        return {"status": "alive", **_readiness()}
    except Exception as exc:  # noqa: BLE001 - the one case that IS unrecoverable
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {exc}")


@app.get("/health")
async def health() -> dict:
    """READINESS probe: 200 (empty body) when ready, 503 while the pipeline is still
    loading. Naming matches tt-media-server's convention (despite reading as the
    opposite of tt-liveness), kept for client compatibility.
    """
    if not _readiness()["model_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {}


@app.get("/v1/models")
async def models() -> dict:
    engine = getattr(app.state, "engine", None)
    name = (engine or {}).get("hf_model", "unknown")
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "tenstorrent"}]}


@app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
async def videos_generations(req: VideoGenerationRequest) -> VideoGenerationResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still starting")
    if req.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail=f"response_format {req.response_format!r} is not supported; this "
                   "server returns b64_json only",
        )
    # One denoise loop at a time: the pipeline owns the mesh.
    async with app.state.device_lock:
        frames = await asyncio.to_thread(_generate, engine, req)
    # MP4 muxing off the loop too: it is real CPU work over a multi-MB payload, and
    # inline here it would block /health and /tt-liveness during encode.
    b64 = await asyncio.to_thread(_frames_to_mp4_b64, frames)
    return VideoGenerationResponse(created=int(time.time()), data=[VideoData(b64_json=b64)])
