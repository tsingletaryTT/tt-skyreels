# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SkyReels-V2-DF-1.3B-540P TTNN pipeline for Tenstorrent Blackhole.
#
# Architecture compatibility
# --------------------------
# SkyReelsV2Transformer3DModel (diffusers) is weight-compatible with
# WanTransformer3DModel (models.tt_dit) because they share:
#
#   • Per-block AdaLN-Zero: scale_shift_table (1, 6, D)
#   • Self-attention: to_q/k/v/out, norm_q/norm_k (RMSNorm via attn1.*)
#   • Cross-attention: attn2.*
#   • FFN: net.0.proj → ff1, net.2 → ff2  (renamed by WanTransformerBlock._prepare_torch_state)
#   • Norms: norm1/norm2/norm3
#   • Outer scale_shift_table (1, 2, D), norm_out, proj_out, patch_embedding, rope
#   • condition_embedder: time_embedder (silu), time_proj, text_embedder (gelu_tanh)
#     — WanTimeTextImageEmbedding uses the same activations as SkyReelsV2TimeTextImageEmbedding
#
# SkyReels-only keys (fps_embedding.*, fps_projection.*) are silently dropped
# via load_torch_state_dict(..., strict=False).
#
# Origin: ported from tt-local-generator's tt_dit hotpatch
#   (patches/tt_dit/pipelines/skyreels_v2/pipeline_skyreels.py), which bind-mounted this
#   file into ~/tt-metal/models/tt_dit/pipelines/skyreels_v2/ at container start. This repo
#   ships the same code as an ordinary importable package instead (tt-model-manager's
#   ``extra_code`` mechanism), so it is packaged like tt-animatediff rather than injected
#   into a tt-metal checkout. Its imports below (``models.tt_dit.*``) are unchanged: they
#   name real tt-metal modules, not hotpatch targets.
#
# Import path in this repo: skyreels_ttnn.pipeline_skyreels
#
# Hardware target: Tenstorrent Blackhole, 1×4 mesh (P150X4) or 2×2 mesh (P300X2).
#   sp_axis=0, tp_axis=1.  num_links=2 (same as WAN pipeline Blackhole config).
#
# Server integration:
#   skyreels_ttnn.server.app (this repo) calls SkyReelsPipeline.create_pipeline(mesh_device,
#   checkpoint_name) in its ASGI lifespan, then pipeline(prompt=..., num_frames=..., ...)
#   via SkyReelsPipeline.__call__ per request — see server/app.py for the tt-dit-server
#   contract (readiness-is-the-lifespan, import-must-not-touch-hardware).

# Deferred imports below (not at module scope): `models.tt_dit.*` pulls in `ttnn` as soon
# as it is imported, and this module is exactly what tt-model-manager's `verify` step
# imports at IMAGE BUILD time, on a machine with no card -- the same contract
# skyreels_ttnn/server/app.py is built around (see its module docstring). A bring-up
# mistake on THIS box found the failure mode directly: an unleased `import
# skyreels_ttnn.pipeline_skyreels` on a machine that DOES have real chips attached
# reached all the way into ttnn's C extension before anything explicitly asked for a
# device. Keeping `models.tt_dit`/`ttnn` out of module scope means importing this file
# is just class definitions -- no device touch -- on any machine, card or no card.
# `from __future__ import annotations` keeps the type hints below (`ttnn.MeshDevice`,
# `DiTParallelConfig`, ...) working as plain strings without needing the real imports
# just to satisfy them.
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import ttnn
    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager


# ---------------------------------------------------------------------------
# Minimal config stub (mirrors what diffusers reads from transformer.config)
# ---------------------------------------------------------------------------

class _TransformerConfig:
    """
    Minimal config object mimicking diffusers' model config.

    SkyReelsV2Pipeline reads these fields from transformer.config at runtime:
      - in_channels: allocate latent buffers
      - num_frame_per_block: causal AR attention (1 = no AR masking)
      - patch_size: compute frame/spatial dimensions
    """

    def __init__(self):
        self.in_channels = 16
        self.num_frame_per_block = 1
        self.patch_size = (1, 2, 2)


# ---------------------------------------------------------------------------
# SkyReelsTTNNTransformer — TTNN-backed drop-in for SkyReelsV2Transformer3DModel
# ---------------------------------------------------------------------------

class SkyReelsTTNNTransformer(torch.nn.Module):
    """
    TTNN-accelerated transformer for SkyReels-V2-DF-1.3B-540P.

    Internally wraps WanTransformer3DModel (models.tt_dit) configured with
    SkyReels dimensions.  Exposes a forward() compatible with
    SkyReelsV2Transformer3DModel so it plugs into SkyReelsV2Pipeline as-is.

    Weight loading:
      SkyReels state_dict keys map directly to WAN TTNN structure.
      FFN keys (net.0.proj → ff1, net.2 → ff2) are renamed by
      WanTransformerBlock._prepare_torch_state.
      Extra SkyReels-only keys (fps_embedding, fps_projection) are dropped
      via strict=False.
    """

    # SkyReels-V2-DF-1.3B-540P dimensions (from HF config.json)
    NUM_HEADS = 12
    DIM = 1536       # 12 heads × 128 head_dim
    FFN_DIM = 8960
    NUM_LAYERS = 30
    TEXT_DIM = 4096  # UMT5 hidden size
    FREQ_DIM = 256
    PATCH_SIZE = (1, 2, 2)

    def __init__(
        self,
        mesh_device: "ttnn.MeshDevice",
        parallel_config: "DiTParallelConfig",
        ccl_manager: "CCLManager",
        is_fsdp: bool = False,
    ):
        # Deferred: see the module-level comment on why models.tt_dit stays out of
        # module scope.
        from models.tt_dit.models.transformers.wan2_2.transformer_wan import (
            WanTransformer3DModel as TTNNWanTransformer3DModel,
        )

        super().__init__()
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config

        # Build WAN TTNN model with SkyReels dimensions.
        # model_type="t2v": T2V path (no image conditioning), in_channels=16.
        self.ttnn_model = TTNNWanTransformer3DModel(
            patch_size=self.PATCH_SIZE,
            num_heads=self.NUM_HEADS,
            dim=self.DIM,
            in_channels=16,
            out_channels=16,
            text_dim=self.TEXT_DIM,
            freq_dim=self.FREQ_DIM,
            ffn_dim=self.FFN_DIM,
            num_layers=self.NUM_LAYERS,
            cross_attn_norm=True,
            eps=1e-6,
            rope_max_seq_len=1024,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            model_type="t2v",
        )

        # Config stub so diffusers pipeline reads transformer.config.in_channels etc.
        self.config = _TransformerConfig()
        self._weights_loaded = False

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model (bfloat16 after weight loading)."""
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        """A placeholder torch device, for diffusers' generic pipeline introspection.

        THE BUG THIS EXISTS TO FIX: ``diffusers.DiffusionPipeline.device`` (and
        ``_execution_device``, which falls back to it) iterates every registered
        ``torch.nn.Module`` component and reads ``module.device``. Plain
        ``torch.nn.Module`` defines no such attribute -- it only exists on modules
        that hold real torch parameters/buffers or define it themselves. Without
        this property, the first real request (not the build-time ``verify``
        checks, which never call the pipeline) raised
        ``AttributeError: 'SkyReelsV2Pipeline' object has no attribute
        '_execution_device'`` -- a genuinely confusing message, because Python's
        property protocol silently converts an ``AttributeError`` raised INSIDE a
        property getter into "attribute not found on the instance", which then
        fell through to ``ConfigMixin.__getattr__``.

        This model's real compute happens on the TTNN mesh device, which has no
        ``torch.device`` representation -- ``cpu`` is a placeholder diffusers uses
        only for bookkeeping (e.g. where to build a ``torch.Generator``), not for
        dispatching any actual computation.
        """
        return torch.device("cpu")

    @contextmanager
    def cache_context(self, cache_name: str):
        """
        No-op context manager matching diffusers CacheMixin.cache_context().

        SkyReelsV2Pipeline wraps transformer calls in cache_context("cond") and
        cache_context("uncond") for KV-cache sharing between CFG passes.  Our
        TTNN backend does not implement KV caching, so we pass through.
        """
        yield

    def _set_ar_attention(self, causal_block_size: int):
        """Called by the DF pipeline to set the causal block size.  No-op here."""
        self.config.num_frame_per_block = causal_block_size

    def load_skyreels_weights(self, checkpoint_name: str):
        """
        Load weights from a SkyReels-V2-DF-1.3B-540P-Diffusers checkpoint.

        Loads the SkyReelsV2Transformer3DModel PyTorch state_dict then maps it
        onto the TTNN WanTransformer3DModel via load_torch_state_dict.

        Why this works:
          - Block keys are identical between SkyReels and WAN TTNN.
          - FFN rename (net.0.proj → ff1, net.2 → ff2) is handled inside the
            WanTransformerBlock._prepare_torch_state call chain.
          - condition_embedder keys match; time_proj weights are reshaped for TP
            by WanTimeTextImageEmbedding._prepare_torch_state.
          - SkyReels-only keys (fps_embedding, fps_projection) are silently
            dropped via strict=False.
        """
        from diffusers.models.transformers.transformer_skyreels_v2 import (
            SkyReelsV2Transformer3DModel,
        )

        print(f"[SkyReels] Loading transformer weights from {checkpoint_name} ...")
        torch_model = SkyReelsV2Transformer3DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        state_dict = torch_model.state_dict()
        del torch_model  # free CPU memory before TTNN loading

        print("[SkyReels] Loading weights into TTNN model ...")
        result = self.ttnn_model.load_torch_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"[SkyReels] WARNING: missing keys ({len(result.missing_keys)}): "
                  f"{result.missing_keys[:5]} ...")
        if result.unexpected_keys:
            print(f"[SkyReels] Ignored SkyReels-only keys ({len(result.unexpected_keys)}): "
                  f"{result.unexpected_keys[:5]} ...")

        self._weights_loaded = True
        print("[SkyReels] TTNN weights loaded.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image=None,
        enable_diffusion_forcing: bool = False,
        fps=None,
        return_dict: bool = True,
        attention_kwargs=None,
    ):
        """
        Drop-in for SkyReelsV2Transformer3DModel.forward().

        hidden_states:          (B, C, F, H, W) — video latents (PyTorch)
        timestep:               (B,)            — denoising timestep (PyTorch)
        encoder_hidden_states:  (B, L, 4096)   — UMT5 text encoder output (PyTorch)

        The TTNN model expects:
          spatial:  PyTorch tensor (B, C, F, H, W)
          prompt:   ttnn.Tensor  (1, B, L, 4096) — converted here from PyTorch
          timestep: ttnn.Tensor  (B, 1, 1, 1), dtype float32 — converted here from
                    PyTorch, mirroring models.tt_dit.pipelines.wan.pipeline_wan's own
                    call site.

        THE BUG THIS DOCSTRING USED TO PAPER OVER: it claimed the TTNN model wanted a
        bare PyTorch tensor for ``timestep``, unconverted. That was never true -- it
        was never exercised, because nothing had run a real denoise step through this
        code path before. diffusers' scheduler produces an int64 timestep (an index
        into the noise schedule); WanTimestepsEmbedding asserts its input dtype
        matches its own (``DataType.FLOAT32``), so a raw int64 torch tensor fails that
        assert immediately -- found on real hardware, one generation past the ftfy fix.
        """
        if not self._weights_loaded:
            raise RuntimeError("Call load_skyreels_weights() before forward().")

        from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor

        # Convert text encoder output to TTNN 4D format.
        # Unsqueeze: (B, L, D) → (1, B, L, D) — WAN TTNN's 4D convention.
        # bf16_tensor with no mesh_axis replicates across all devices.
        enc_4d = encoder_hidden_states.unsqueeze(0).to(torch.bfloat16)
        tt_prompt = bf16_tensor(enc_4d, device=self.mesh_device)

        # (B,) int64 -> (B, 1, 1, 1) float32 ttnn.Tensor, exactly as
        # pipeline_wan.py builds it before its own transformer call.
        tt_timestep = float32_tensor(
            timestep.to(torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1),
            device=self.mesh_device,
        )

        noise_pred = self.ttnn_model(
            spatial=hidden_states,
            prompt=tt_prompt,
            timestep=tt_timestep,
        )

        if return_dict:
            return {"sample": noise_pred}
        return (noise_pred,)


# ---------------------------------------------------------------------------
# SkyReelsPipeline — server-compatible wrapper with WanPipeline-like interface
# ---------------------------------------------------------------------------

class SkyReelsPipeline:
    """
    TT-hardware SkyReels-V2-DF-1.3B-540P pipeline.

    Wraps SkyReelsTTNNTransformer inside a diffusers SkyReelsV2Pipeline.
    Exposes create_pipeline(mesh_device, checkpoint_name) and __call__()
    matching the WanPipeline interface so TTSkyReelsRunner can follow the
    same pattern as TTWan22Runner.

    Attributes:
        mesh_device: the ttnn.MeshDevice (read by runner to pick resolution).
    """

    CHECKPOINT = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

    def __init__(self, diffusers_pipe, mesh_device: "ttnn.MeshDevice"):
        self._pipe = diffusers_pipe
        self.mesh_device = mesh_device

    @staticmethod
    def create_pipeline(
        mesh_device: "ttnn.MeshDevice",
        checkpoint_name: str = None,
        pipeline_class=None,
    ) -> "SkyReelsPipeline":
        """
        Create a SkyReelsPipeline backed by TTNN WAN transformer.

        Args:
            mesh_device:     Already-opened ttnn.MeshDevice (managed by
                             TTDiTRunner / BaseMetalDeviceRunner.set_device).
                             Fabric initialization has already been done by
                             TTDiTRunner._configure_fabric before this is called.
            checkpoint_name: HF repo ID or local path to the SkyReels checkpoint.
                             Defaults to MODEL_WEIGHTS_DIR env var, then the
                             canonical HF repo ID.
            pipeline_class:  Reserved for future subclasses (ignored currently).
        """
        # Deferred: see the module-level comment on why models.tt_dit/ttnn stay out of
        # module scope.
        import ttnn
        from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
        from models.tt_dit.parallel.manager import CCLManager

        if checkpoint_name is None:
            checkpoint_name = os.environ.get(
                "MODEL_WEIGHTS_DIR", SkyReelsPipeline.CHECKPOINT
            )

        mesh_shape = tuple(mesh_device.shape)  # e.g., (1, 4) or (2, 2)

        # Parallel config: TP on axis=1, SP on axis=0.
        # For (1, 4): tp=4, sp=1 — pure TP, no SP (same as standalone script).
        # For (2, 2): tp=2, sp=2 — 2-way TP + 2-way SP (QB2 configuration).
        tp_factor = mesh_shape[1]
        sp_factor = mesh_shape[0]

        parallel_config = DiTParallelConfig(
            cfg_parallel=None,
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=1),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=0),
        )

        # CCL manager: Linear topology for Blackhole 1D fabric.
        # num_links=2 matches the WAN pipeline Blackhole config.
        ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=2,
            topology=ttnn.Topology.Linear,
        )

        print(f"[SkyReels] Building TTNN transformer (mesh={mesh_shape}) ...")
        ttnn_transformer = SkyReelsTTNNTransformer(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            is_fsdp=False,
        )
        ttnn_transformer.load_skyreels_weights(checkpoint_name)

        print("[SkyReels] Loading diffusers pipeline components ...")
        diffusers_pipe = _build_diffusers_pipeline(
            checkpoint_name=checkpoint_name,
            ttnn_transformer=ttnn_transformer,
        )

        print("[SkyReels] Pipeline ready.")
        return SkyReelsPipeline(diffusers_pipe, mesh_device)

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 272,
        width: int = 480,
        num_frames: int = 9,
        num_inference_steps: int = 20,
        guidance_scale: float = 6.0,
        seed: int = 0,
        # Accept (and ignore) WAN-specific kwargs for API compatibility with the runner.
        guidance_scale_2: float = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Run SkyReels denoising and return video frames as a numpy array.

        Returns:
            numpy array of shape (1, T, H, W, C) with pixel values in [0, 1].
            This matches the format returned by WanPipeline so the video service
            layer can encode it as MP4 without modification.

        Args:
            prompt:               Text prompt describing the video.
            negative_prompt:      Negative guidance text (optional).
            height / width:       Output resolution in pixels.
            num_frames:           Number of output frames.
            num_inference_steps:  Number of denoising steps.
            guidance_scale:       Classifier-free guidance scale.
            seed:                 Random seed for the generator.
            guidance_scale_2:     Ignored — WAN API compat only (no dual-scale CFG).
            **kwargs:             Ignored — forward-compat guard.
        """
        generator = torch.Generator().manual_seed(seed)

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        # output.frames[0] is a list of T PIL.Image objects.
        # Convert to numpy (1, T, H, W, C) in [0, 1] matching WanPipeline format.
        frames_pil = output.frames[0]
        frames_np = np.stack(
            [np.array(f, dtype=np.float32) / 255.0 for f in frames_pil]
        )  # (T, H, W, C)
        return frames_np[np.newaxis]  # (1, T, H, W, C)


# ---------------------------------------------------------------------------
# Internal helper — build the diffusers SkyReelsV2Pipeline
# ---------------------------------------------------------------------------

def _build_diffusers_pipeline(checkpoint_name: str, ttnn_transformer):
    """
    Load VAE, scheduler, tokenizer, text encoder from the checkpoint and
    assemble a SkyReelsV2Pipeline with the TTNN transformer.

    This mirrors build_skyreels_ttnn_pipeline in skyreels_ttnn_pipeline.py but
    takes the transformer as an argument (device lifecycle is managed externally).
    """
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2 import SkyReelsV2Pipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")

    # Text encoder runs on CPU in bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_name,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    # VAE runs on CPU in float32 for numerical stability
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(
        checkpoint_name, subfolder="scheduler"
    )

    # Build pipeline directly rather than via from_pretrained so we can supply
    # our custom transformer without triggering ModelMixin type checks.
    pipe = SkyReelsV2Pipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=ttnn_transformer,
        vae=vae,
        scheduler=scheduler,
    )

    # Apply flow_shift (SkyReels recommended: 8.0 for 540P)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=8.0
    )

    return pipe
