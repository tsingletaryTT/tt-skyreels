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


def test_importing_the_session_module_does_not_import_ttnn():
    """Same property, for skyreels_ttnn.session -- the shared device singleton the
    server app AND the Gradio app both delegate to. Its module scope is just a lock and
    three globals; every ttnn/pipeline import lives inside ensure_skyreels_pipeline()."""
    code = textwrap.dedent(
        """
        import sys
        import skyreels_ttnn.session  # noqa: F401
        bad = sorted(m for m in sys.modules if m == "ttnn" or m.startswith("ttnn."))
        print(",".join(bad))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"importing the session module pulled in: {out.stdout.strip()}"


def test_a_mesh_shape_mismatch_against_an_already_open_session_is_refused():
    """Reconfiguring means closing first -- silently reopening against a different
    shape than a caller asked for is the same wrong-mesh hazard mesh_shape_from_env
    guards against, one layer up."""
    import skyreels_ttnn.session as session

    session._device = object()
    session._pipeline = object()
    session._mesh_shape = (2, 2)
    try:
        with pytest.raises(RuntimeError):
            session.ensure_skyreels_pipeline((1, 4))
    finally:
        session._device = None
        session._pipeline = None
        session._mesh_shape = None


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


# ---- the packaging contract agrees with the code it describes ----------------------------
#
# These tests used to read tt_model_package.yaml, the v5.1 CONTAINER manifest. That file
# was deleted on 2026-09-15 when the package moved to a v6 thin bundle, which
# `tt-model package-thin` builds from CLI flags; nothing in this repo is the manifest
# any more. So the 8 tests that read it errored. Here is what happened to each property:
#
# * dropped, v5.1-only concepts with no v6 equivalent: `source.extra_code` shipping
#   skyreels_ttnn, and the `source.code` allowlist containing models/tt_dit. In v6 those
#   are the two wheels, which this repo doesn't build as a unit.
# * dropped: `kind == "tt-dit-server"`. It is a package-thin flag, and nothing in this
#   repo carries it, so asserting it here would compare a literal to itself.
# * kept, re-pointed at what v6 actually uses: the `--app` target (documented in
#   README.md) resolves to the ASGI app; the env the published bundle sets parses to a
#   supported mesh; the Gradio app opens that same mesh; the weights repo + pinned
#   revision reach every from_pretrained call.

#: What the published bundle's manifest.json sets (episod/tt-skyreels @ 69473ad,
#: `env` and `deps.app`). Recorded by hand, not fetched: tests stay offline. If a
#: repackage changes either one, update it here.
PUBLISHED_V6_ENV = {"SKYREELS_MESH_SHAPE": "2x2"}
PUBLISHED_V6_APP = "skyreels_ttnn.server.app:app"


def test_the_readme_documents_the_app_target_the_bundle_uses():
    """The README's package-thin recipe is how a repackage gets its --app flag, so it
    must name the same target the published bundle runs."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert f"--app {PUBLISHED_V6_APP}" in readme


def test_the_bundle_app_target_resolves_to_the_asgi_app():
    import importlib

    module_path, attr = PUBLISHED_V6_APP.split(":", 1)
    assert getattr(importlib.import_module(module_path), attr) is app


def test_the_bundle_env_names_the_variable_the_server_reads():
    assert MESH_SHAPE_ENV in PUBLISHED_V6_ENV


def test_the_bundle_env_mesh_is_one_this_server_supports():
    assert mesh_shape_from_env(PUBLISHED_V6_ENV) == (2, 2)
    assert (2, 2) in SUPPORTED_MESH_SHAPES


# ---- weights: one repo, one pinned revision, every component ------------------------------


def test_the_weights_pin_is_a_full_sha():
    from skyreels_ttnn.pipeline_skyreels import PINNED_WEIGHTS_REVISION

    assert len(PINNED_WEIGHTS_REVISION) == 40
    int(PINNED_WEIGHTS_REVISION, 16)


@pytest.mark.parametrize(
    "env_value, expected",
    [(None, "PIN"), ("", "PIN"), ("abc123" * 6 + "abcd", "abc123" * 6 + "abcd")],
)
def test_weights_revision_env_override(monkeypatch, env_value, expected):
    """Unset and exported-but-empty both mean "use the pin"; anything else wins."""
    from skyreels_ttnn import pipeline_skyreels as p

    if env_value is None:
        monkeypatch.delenv(p.WEIGHTS_REVISION_ENV, raising=False)
    else:
        monkeypatch.setenv(p.WEIGHTS_REVISION_ENV, env_value)
    want = p.PINNED_WEIGHTS_REVISION if expected == "PIN" else expected
    assert p.weights_revision() == want


def test_a_local_checkpoint_dir_gets_no_revision(tmp_path):
    from skyreels_ttnn.pipeline_skyreels import _revision_kwargs

    assert _revision_kwargs(str(tmp_path)) == {}


def test_a_foreign_hub_repo_does_not_get_this_repos_sha(monkeypatch):
    from skyreels_ttnn import pipeline_skyreels as p

    monkeypatch.delenv(p.WEIGHTS_REVISION_ENV, raising=False)
    assert p._revision_kwargs("someone/else") == {}


def test_every_component_loads_from_the_pinned_revision(monkeypatch):
    """THE WIRING, not the helper: stub every from_pretrained that
    _build_diffusers_pipeline and load_skyreels_weights reach, run them, and check each
    call carried the pinned repo AND revision. A pin that exists as a constant but
    never reaches a from_pretrained call is the failure this catches. The stubs keep
    this offline and CPU-only; no weights are downloaded and ttnn is never imported."""
    import diffusers
    import diffusers.models.transformers.transformer_skyreels_v2 as tsv2
    import diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2 as psv2
    import transformers

    from skyreels_ttnn import pipeline_skyreels as p

    monkeypatch.delenv(p.WEIGHTS_REVISION_ENV, raising=False)
    calls = []

    class _Loaded:
        config = {}

        def state_dict(self):
            return {}

    def recorder(name):
        def from_pretrained(checkpoint, **kwargs):
            calls.append((name, checkpoint, kwargs.get("subfolder"), kwargs.get("revision")))
            return _Loaded()

        return from_pretrained

    for name, owner in [
        ("tokenizer", transformers.AutoTokenizer),
        ("text_encoder", transformers.UMT5EncoderModel),
        ("vae", diffusers.AutoencoderKLWan),
        ("scheduler", diffusers.UniPCMultistepScheduler),
        ("transformer", tsv2.SkyReelsV2Transformer3DModel),
    ]:
        monkeypatch.setattr(owner, "from_pretrained", staticmethod(recorder(name)))
    # The pipeline constructor and the flow_shift re-config would type-check the
    # stubs; replace them too. Neither loads weights.
    monkeypatch.setattr(psv2, "SkyReelsV2Pipeline", lambda **kw: type("P", (), {"scheduler": _Loaded()})())
    monkeypatch.setattr(diffusers.UniPCMultistepScheduler, "from_config", staticmethod(lambda *a, **k: None))

    p._build_diffusers_pipeline(p.SkyReelsPipeline.CHECKPOINT, ttnn_transformer=object())

    # The transformer load is a method on SkyReelsTTNNTransformer; call it unbound
    # with a stand-in self whose ttnn_model accepts the (empty) state dict.
    class _FakeTTNN:
        def load_torch_state_dict(self, sd, strict):
            return type("R", (), {"missing_keys": [], "unexpected_keys": []})()

    fake_self = type("S", (), {"ttnn_model": _FakeTTNN()})()
    p.SkyReelsTTNNTransformer.load_skyreels_weights(fake_self, p.SkyReelsPipeline.CHECKPOINT)

    assert {c[0] for c in calls} == {"tokenizer", "text_encoder", "vae", "scheduler", "transformer"}
    for name, checkpoint, subfolder, revision in calls:
        assert checkpoint == "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers", name
        assert revision == p.PINNED_WEIGHTS_REVISION, name


def test_the_request_default_step_count_is_20_and_the_gradio_ui_agrees():
    """Docs once claimed an "8 step" default; the code default has always been 20.
    Pin the server constant, the request model and the UI slider together."""
    import app as gradio_app

    from skyreels_ttnn.server.app import DEFAULT_NUM_INFERENCE_STEPS

    assert DEFAULT_NUM_INFERENCE_STEPS == 20
    assert VideoGenerationRequest(prompt="x").num_inference_steps == DEFAULT_NUM_INFERENCE_STEPS
    assert gradio_app.steps_slider.value == DEFAULT_NUM_INFERENCE_STEPS


# ---- the Gradio app and its discolike manifest --------------------------------------------


def test_importing_the_gradio_app_does_not_import_ttnn():
    """Same property as the ASGI app: skyreels_ttnn.session (and therefore ttnn) is
    imported inside the generate() callback, not at module scope, so building the
    gr.Blocks graph works with no card."""
    code = textwrap.dedent(
        """
        import sys
        import app  # noqa: F401
        bad = sorted(m for m in sys.modules if m == "ttnn" or m.startswith("ttnn."))
        print(",".join(bad))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"importing app.py pulled in: {out.stdout.strip()}"


def test_the_gradio_apps_frame_count_default_matches_the_servers():
    """gr.Number(value=33, ...) is hand-typed in app.py rather than imported, because
    Gradio component defaults are literals, not references -- pin it against the
    server's own constant so the two cannot drift apart silently."""
    import app as gradio_app

    from skyreels_ttnn.server.app import DEFAULT_NUM_FRAMES

    assert gradio_app.frames_num.value == DEFAULT_NUM_FRAMES


def test_the_gradio_app_opens_the_same_mesh_the_bundle_declares():
    import app as gradio_app

    assert gradio_app.MESH_SHAPE == mesh_shape_from_env(PUBLISHED_V6_ENV)


def test_the_disco_manifest_parses_and_points_at_the_gradio_app():
    disco = yaml.safe_load((REPO_ROOT / ".disco" / "app.yaml").read_text())
    assert disco["name"] == "skyreels"
    assert disco["port"] == 7861
    assert disco["launch"] == ".venv/bin/python app.py"
    # tt-discolike's own manifest schema requires this pattern for `name`.
    import re

    assert re.fullmatch(r"[A-Za-z0-9_.-]+", disco["name"])


def test_the_disco_manifest_port_matches_the_gradio_apps_launch_port():
    disco = yaml.safe_load((REPO_ROOT / ".disco" / "app.yaml").read_text())
    server_port_line = (REPO_ROOT / "app.py").read_text()
    assert f"server_port={disco['port']}" in server_port_line
