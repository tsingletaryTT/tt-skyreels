# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The ASGI app, tested on CPU with no card and no tt-metal.

Modeled on tt-animatediff's tests/test_server_app.py -- the reference test suite for this
kind. Every assertion here is one a wrong answer makes expensive on hardware:

* importing the module must not touch ttnn -- tt-model-manager's ``verify_lines`` imports
  the ASGI attribute at IMAGE BUILD time, on a machine with no device, purely to prove the
  code allowlist shipped the server. A module-scope ``import ttnn`` turns that check into a
  build failure, and a stray one is invisible until then;
* the mesh shape must come from the environment or fail loudly, and only the two shapes
  SkyReelsPipeline has actually been exercised on may be accepted -- converting its TP/SP
  split against an unvalidated mesh produces bad frames, not an error;
* the readiness contract must hold: no device work outside the lifespan, because the
  supervisor calls the server ready the moment uvicorn logs "Application startup complete".

The lifespan is deliberately never entered: ``TestClient(app)`` as a plain object does not
run it, and entering it would open a device.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from skyreels_ttnn.server.app import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    MESH_SHAPE_ENV,
    SUPPORTED_MESH_SHAPES,
    VideoGenerationRequest,
    app,
    mesh_shape_from_env,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tt_model_package.yaml"


@pytest.fixture
def manifest():
    return yaml.safe_load(MANIFEST_PATH.read_text())


def test_importing_the_server_does_not_import_ttnn():
    """THE BUILD-TIME PROPERTY. Run in a subprocess so this process's own imports --
    which may include ttnn from another test -- cannot make it pass for the wrong
    reason."""
    code = textwrap.dedent(
        """
        import sys
        import skyreels_ttnn.server.app  # noqa: F401
        bad = sorted(m for m in sys.modules if m == "ttnn" or m.startswith("ttnn."))
        print(",".join(bad))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"importing the server pulled in: {out.stdout.strip()}"


def test_importing_the_pipeline_module_does_not_import_ttnn():
    """Same property, for skyreels_ttnn.pipeline_skyreels this time -- it is imported
    directly by the manifest's ``verify`` list (independent of the server app), so it
    needs its own check.

    THE BUG THIS GUARDS AGAINST: the first version of this file imported
    ``models.tt_dit.models.transformers.wan2_2.transformer_wan`` (and therefore ``ttnn``)
    at module scope, for type hints alone. On a machine with no card that is merely
    slow; on THIS box, which has real chips attached, an unleased `import
    skyreels_ttnn.pipeline_skyreels` run during bring-up reached all the way into
    ttnn's C extension and segfaulted -- before anything had explicitly asked for a
    device, and with no gozer lease held. Every ``models.tt_dit``/``ttnn`` import now
    lives inside the function that needs it (see the module's own top-of-file comment).
    """
    code = textwrap.dedent(
        """
        import sys
        import skyreels_ttnn.pipeline_skyreels  # noqa: F401
        bad = sorted(m for m in sys.modules if m == "ttnn" or m.startswith("ttnn."))
        print(",".join(bad))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"importing the pipeline pulled in: {out.stdout.strip()}"


def test_the_asgi_attribute_is_what_uvicorn_will_load():
    """``runtime.app: skyreels_ttnn.server.app:app`` names this attribute; uvicorn calls
    it with (scope, receive, send)."""
    assert callable(app)


def test_the_routes_are_the_ones_the_manifest_promises():
    paths = {r.path for r in app.routes}
    assert {"/health", "/tt-liveness", "/v1/models", "/v1/videos/generations"} <= paths


# ---- mesh shape --------------------------------------------------------------------------


def test_mesh_shape_is_required_because_this_model_has_no_safe_default():
    """Unlike tt-animatediff (1x1 default is always safe -- it is the only shape the
    model supports), SkyReels needs exactly ONE of two multi-chip shapes. Defaulting to
    either would be a guess about which four chips to claim on a shared box."""
    with pytest.raises(ValueError):
        mesh_shape_from_env({})


@pytest.mark.parametrize("raw,expected", [("1x4", (1, 4)), ("2X2", (2, 2)), ("2x2", (2, 2))])
def test_mesh_shape_is_read_from_the_environment(raw, expected):
    assert mesh_shape_from_env({MESH_SHAPE_ENV: raw}) == expected


@pytest.mark.parametrize("raw", ["four", "1x", "0x4", "-1x2", "1,4"])
def test_a_malformed_mesh_shape_raises_rather_than_defaulting(raw):
    with pytest.raises(ValueError):
        mesh_shape_from_env({MESH_SHAPE_ENV: raw})


@pytest.mark.parametrize("raw", ["1x1", "1x2", "4x1", "1x8"])
def test_a_mesh_shape_the_pipeline_has_not_been_exercised_on_is_refused(raw):
    """Parsing is not the question -- SkyReelsPipeline.create_pipeline's tp/sp derivation
    is generic and would accept these too, but nobody has proven the result correct."""
    with pytest.raises(ValueError):
        mesh_shape_from_env({MESH_SHAPE_ENV: raw})


def test_supported_mesh_shapes_matches_the_pipelines_own_documented_targets():
    assert SUPPORTED_MESH_SHAPES == {(1, 4), (2, 2)}


# ---- the request contract ----------------------------------------------------------------


def test_prompt_is_required_and_may_not_be_empty():
    with pytest.raises(Exception):
        VideoGenerationRequest(prompt="")


def test_defaults_match_the_pipelines_own_call_signature():
    """Derived from the library, not retyped as literals -- see
    pipeline_skyreels.py:SkyReelsPipeline.__call__ for the values being compared against."""
    import inspect

    from skyreels_ttnn.pipeline_skyreels import SkyReelsPipeline

    lib = inspect.signature(SkyReelsPipeline.__call__).parameters
    r = VideoGenerationRequest(prompt="a bicycle race")

    assert r.num_inference_steps == lib["num_inference_steps"].default
    assert r.guidance_scale == lib["guidance_scale"].default
    assert r.seed == lib["seed"].default
    assert DEFAULT_HEIGHT == lib["height"].default
    assert DEFAULT_WIDTH == lib["width"].default
    # DEFAULT_GUIDANCE_SCALE backs the request field default above; pin it too so a
    # future edit to one cannot drift from the other silently.
    assert DEFAULT_GUIDANCE_SCALE == lib["guidance_scale"].default

    # num_frames is the one value that deliberately does NOT track the library: 33 here
    # against the pipeline's bare 9. It matches TTSkyReelsRunner.DEFAULT_NUM_FRAMES (the
    # tt-inference-server precedent this app replaces) -- a serving choice about clip
    # length, not a calibrated constant -- so it is pinned as a literal on purpose, and
    # pinned here so that changing the API default for every HTTP client stays deliberate.
    assert DEFAULT_NUM_FRAMES == 33
    assert lib["num_frames"].default == 9


def test_the_transformer_has_a_device_property_diffusers_can_introspect():
    """THE BUG THIS GUARDS AGAINST: found on real hardware, on the first actual generation
    request (the build-time ``verify`` checks never call the pipeline, so this survived
    all the way to a served, weight-loaded container).

    ``diffusers.DiffusionPipeline.device`` -- and ``_execution_device``, which falls back
    to it -- iterates every registered ``torch.nn.Module`` component and reads
    ``module.device``. Plain ``torch.nn.Module`` has no such attribute. Without this
    property, the very first ``/v1/videos/generations`` call raised
    ``AttributeError: 'SkyReelsV2Pipeline' object has no attribute '_execution_device'``
    -- confusing because Python's property protocol converts an ``AttributeError`` raised
    INSIDE a property getter into "attribute not found", which then fell through to
    ``ConfigMixin.__getattr__`` and reported the wrong attribute name entirely.

    Constructed via ``object.__new__`` rather than the real ``__init__`` (which opens a
    TTNN mesh device) -- this test only needs the class to define the property, not a
    working instance.
    """
    import torch

    from skyreels_ttnn.pipeline_skyreels import SkyReelsTTNNTransformer

    bare = object.__new__(SkyReelsTTNNTransformer)
    assert bare.device == torch.device("cpu")


@pytest.mark.parametrize(
    "field,value",
    [("num_frames", 0), ("num_frames", 98), ("num_inference_steps", 0),
     ("num_inference_steps", 101), ("guidance_scale", -1.0), ("guidance_scale", 21.0)],
)
def test_out_of_range_parameters_are_refused_at_the_edge(field, value):
    """A device-side failure deep in a denoise loop is far more expensive to diagnose
    than a 422 from the request model."""
    with pytest.raises(Exception):
        VideoGenerationRequest(prompt="a bicycle race", **{field: value})


# ---- behaviour before the lifespan has run -----------------------------------------------


def test_health_is_a_READINESS_probe_and_refuses_traffic_before_the_model_is_warm():
    r = TestClient(app).get("/health")
    assert r.status_code == 503


def test_health_body_is_vllm_compatible_when_ready():
    app.state.engine = {"hf_model": "x", "mesh_device": "QB2", "mesh_shape": (2, 2)}
    try:
        r = TestClient(app).get("/health")
        assert r.status_code == 200 and r.json() == {}
    finally:
        del app.state.engine


def test_tt_liveness_reports_alive_while_the_model_is_still_warming():
    r = TestClient(app).get("/tt-liveness")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "alive"
    assert body["model_ready"] is False


def test_tt_liveness_carries_the_model_payload_once_warm():
    app.state.engine = {
        "hf_model": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        "mesh_device": "QB2",
        "mesh_shape": (2, 2),
    }
    try:
        body = TestClient(app).get("/tt-liveness").json()
        assert body == {
            "status": "alive",
            "model_ready": True,
            "model": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            "mesh_device": "QB2",
            "mesh_shape": "2x2",
        }
    finally:
        del app.state.engine


def test_the_two_probes_cannot_disagree_about_readiness():
    from skyreels_ttnn.server.app import _readiness

    c = TestClient(app)
    assert _readiness()["model_ready"] is False
    assert c.get("/tt-liveness").json()["model_ready"] is False
    assert c.get("/health").status_code == 503

    app.state.engine = {"hf_model": "x", "mesh_device": "QB2", "mesh_shape": (2, 2)}
    try:
        assert _readiness()["model_ready"] is True
        assert c.get("/tt-liveness").json()["model_ready"] is True
        assert c.get("/health").status_code == 200
    finally:
        del app.state.engine


def test_generation_is_refused_while_still_starting():
    r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a bicycle race"})
    assert r.status_code == 503


def test_an_unsupported_response_format_is_refused():
    """The server has no file store, so a URL response would point at nothing."""
    app.state.engine = {"hf_model": "x", "mesh_shape": (2, 2)}
    try:
        r = TestClient(app).post(
            "/v1/videos/generations",
            json={"prompt": "a bicycle race", "response_format": "url"},
        )
        assert r.status_code == 400
    finally:
        del app.state.engine


# ---- the manifest agrees with the code it describes ---------------------------------------


def test_the_manifest_points_at_the_asgi_attribute_that_exists(manifest):
    module_path, attr = manifest["runtime"]["app"].split(":", 1)
    assert module_path == "skyreels_ttnn.server.app"
    assert attr == "app"


def test_the_manifest_ships_the_package_that_holds_the_server(manifest):
    paths = [e["paths"] for e in manifest["source"]["extra_code"]]
    flat = [p for group in paths for p in group]
    assert "skyreels_ttnn" in flat


def test_the_manifest_mesh_shape_env_matches_the_name_the_server_reads(manifest):
    assert manifest["runtime"]["mesh_shape_env"] == MESH_SHAPE_ENV


def test_the_manifest_declares_the_diffusion_kind(manifest):
    assert manifest["kind"] == "tt-dit-server"


def test_the_manifest_declares_a_mesh_this_server_actually_supports(manifest):
    from skyreels_ttnn.server.app import mesh_shape_from_env as parse

    # Mirrors tt_kernel.container_manifest.MESH_DEVICE_PRESETS: "QB2" -> (2, 2).
    assert manifest["serve"]["mesh_device"] == "QB2"
    assert parse({MESH_SHAPE_ENV: "2x2"}) in SUPPORTED_MESH_SHAPES


def test_the_allowlist_ships_the_tt_dit_subtree_the_pipeline_imports(manifest):
    code = manifest["source"]["code"]
    assert "models/tt_dit" in code
    assert "models/common" in code


def test_the_manifest_declares_the_weights_the_served_path_actually_loads(manifest):
    from skyreels_ttnn.pipeline_skyreels import SkyReelsPipeline

    assert manifest["weights"] == SkyReelsPipeline.CHECKPOINT
