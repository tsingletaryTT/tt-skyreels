#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio UI for tt-skyreels (SkyReels-V2-DF-1.3B-540P, text-to-video).

Local (Blackhole hardware, P300x2/QB2 -- a 2x2, 4-chip mesh):
    pip install -e ".[ui]"
    python app.py

Unlike tt-animatediff's app.py, there is no cpu/sim mode here: SkyReels-V2-DF-1.3B-540P
has only ever been exercised on real Blackhole hardware (see
skyreels_ttnn.server.app.SUPPORTED_MESH_SHAPES), so a mode selector would offer choices
that don't work rather than save anyone anything.

Device/pipeline lifecycle lives in skyreels_ttnn.session (the same module
skyreels_ttnn/server/app.py's ASGI lifespan uses) -- opened lazily on the first Generate
click and held for the life of this process. Importing this module does NOT touch ttnn;
the session import happens inside the generate() callback.

No per-step preview streaming: unlike animatediff's generate_frames_temporal, SkyReels-
Pipeline.__call__ does not expose an on_step callback, so there is nothing to stream
mid-denoise. The Generate click still yields a "starting" status immediately so the UI
does not look frozen while kernels JIT-compile on a cold cache (first request, or any
new num_frames/num_inference_steps combination), before yielding the final video.
"""

import sys
import tempfile
from pathlib import Path

import gradio as gr

# Add repo root to path so skyreels_ttnn is importable when this file is run directly
# (python app.py) rather than through an installed package.
sys.path.insert(0, str(Path(__file__).parent))

#: The only mesh this app opens. Matches tt_model_package.yaml's serve.mesh_device: QB2.
MESH_SHAPE = (2, 2)

NEG_DEFAULT = "blurry, low quality, distorted, text, watermark, deformed"


def generate(
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Generator: yields a status update immediately, then the final video path.

    Blocking work (device open on first call, then denoise + VAE decode + MP4 encode)
    runs on this thread -- Gradio's queue already runs each callback off the main
    server thread, so there is no separate worker thread here the way the ASGI app
    needs one to keep /health answering during a request.
    """
    num_frames = int(num_frames)
    num_inference_steps = int(num_inference_steps)
    seed = int(seed)

    if not (prompt or "").strip():
        raise gr.Error("Prompt cannot be empty.")

    # Valid counts satisfy (N-1) % 4 == 0 (see skyreels_ttnn.server.app's own note) --
    # refused here, at the edge, rather than deep in the denoise loop with a device
    # already claimed for the attempt.
    if (num_frames - 1) % 4 != 0:
        raise gr.Error(
            f"num_frames must satisfy (N-1) % 4 == 0 (e.g. 9, 13, 17, ..., 33, 65, 97); "
            f"got {num_frames}."
        )

    yield None, "Opening the mesh device (first call only; loads weights + compiles kernels)…"

    from skyreels_ttnn.session import ensure_skyreels_pipeline

    try:
        _device, pipeline = ensure_skyreels_pipeline(MESH_SHAPE)
    except Exception as exc:
        raise gr.Error(f"Device/pipeline setup failed: {exc}") from exc

    yield None, "Generating (kernel compilation on a cold shape can take a couple minutes)…"

    from skyreels_ttnn.server.app import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_FPS

    try:
        frames = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc

    import numpy as np
    import imageio.v2 as imageio

    out_path = str(Path(tempfile.mkdtemp()) / "output.mp4")
    frames_uint8 = (np.clip(frames[0], 0.0, 1.0) * 255).astype(np.uint8)
    imageio.mimwrite(out_path, list(frames_uint8), fps=DEFAULT_FPS, format="mp4")

    yield out_path, "Done."


# ── UI layout ─────────────────────────────────────────────────────────────

_DESCRIPTION = """
**SkyReels-V2-DF-1.3B-540P on Tenstorrent Blackhole** — text-to-video, 480x272 @ 24fps.

Runs on a P300x2 / QB2 board (2x2, 4-chip mesh). The first generation in a fresh process
pays weight-load and TTNN kernel-compile cost; later ones with the same frame/step count
are much faster.
"""

with gr.Blocks(title="tt-skyreels") as demo:
    gr.Markdown("# tt-skyreels")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="a red bicycle leaning against a brick wall in soft morning light",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value=NEG_DEFAULT,
                lines=2,
            )
            with gr.Row():
                # 33 matches skyreels_ttnn.server.app.DEFAULT_NUM_FRAMES -- the same
                # serving default, not the bare pipeline default (9). See that module's
                # own comment on why the two deliberately differ.
                frames_num = gr.Number(
                    value=33, precision=0, label="Frames",
                    info="Must satisfy (N-1) % 4 == 0: 9, 13, 17, 21, 25, 29, 33, 65, 97, …",
                )
                steps_slider = gr.Slider(1, 50, value=20, step=1, label="Inference steps")
            with gr.Row():
                guidance_slider = gr.Slider(0.0, 20.0, value=6.0, step=0.5, label="Guidance scale")
                seed_num = gr.Number(value=0, precision=0, label="Seed")

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Output")
            status_label = gr.Textbox(label="Status", value="", interactive=False, lines=1)

    run_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, frames_num, steps_slider, guidance_slider, seed_num],
        outputs=[output_video, status_label],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
