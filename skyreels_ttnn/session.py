# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Module-level mesh-device singleton for library callers.

Modeled on tt-animatediff's ``animatediff_ttnn/session.py``. Manages the lifetime of the
TTNN mesh device and the loaded SkyReels pipeline so that callers (the ASGI server, the
Gradio app, a script) share one open device across repeated calls without paying the
weight-load + fabric-init cost more than once per process.

Usage::

    from skyreels_ttnn.session import ensure_skyreels_pipeline
    device, pipeline = ensure_skyreels_pipeline((2, 2))

Thread safety: the initialization lock guarantees concurrent first callers serialize
correctly. After the first call returns, subsequent calls (with the SAME mesh shape) are
lock-free reads. Publish order matters: ``_pipeline`` is set before ``_device`` is
readable via the fast path, mirroring animatediff's own comment on this -- a caller must
never observe a device with no pipeline behind it.

Process lifetime: the device is held open until close() is called or the process exits.
TTNN does not support opening the same device twice in one process, so only one caller in
a process may hold this open at a time -- the ASGI server (skyreels_ttnn/server/app.py)
and the Gradio app (app.py) each run in their OWN process and are unaffected by each
other.
"""

import threading
from typing import Optional, Tuple

_lock = threading.Lock()
_device = None
_pipeline = None
_mesh_shape: Optional[Tuple[int, int]] = None


def ensure_skyreels_pipeline(mesh_shape: Tuple[int, int]):
    """Open the mesh device and load the SkyReels pipeline (once per process).

    Args:
        mesh_shape: (rows, cols), e.g. (2, 2) for QB2. A second call with a DIFFERENT
            shape than the one already open raises -- reconfiguring means closing
            first, and silently reopening against a different shape than a caller
            asked for is exactly the kind of wrong-mesh failure this module exists to
            prevent (see skyreels_ttnn.server.app.mesh_shape_from_env's own reasoning).

    Returns:
        (device, pipeline)

    Raises:
        RuntimeError: device open or pipeline load failed, or a shape mismatch against
            an already-open device.
    """
    global _device, _pipeline, _mesh_shape
    if _device is not None and _pipeline is not None:
        if _mesh_shape != mesh_shape:
            raise RuntimeError(
                f"skyreels_ttnn.session already holds a {_mesh_shape} mesh; "
                f"call close() before requesting {mesh_shape}."
            )
        return _device, _pipeline

    with _lock:
        # Re-check inside the lock -- another thread may have initialized while we
        # were waiting.
        if _device is not None and _pipeline is not None:
            if _mesh_shape != mesh_shape:
                raise RuntimeError(
                    f"skyreels_ttnn.session already holds a {_mesh_shape} mesh; "
                    f"call close() before requesting {mesh_shape}."
                )
            return _device, _pipeline

        import ttnn

        from skyreels_ttnn.pipeline_skyreels import SkyReelsPipeline

        _configure_fabric()
        rows, cols = mesh_shape
        dispatch_core_config = ttnn.DispatchCoreConfig(
            None,
            ttnn.device.DispatchCoreAxis.ROW if ttnn.device.is_blackhole() else None,
            ttnn.FabricTensixConfig.MUX if ttnn.device.is_blackhole() else None,
        )

        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            dispatch_core_config=dispatch_core_config,
        )
        try:
            pipeline = SkyReelsPipeline.create_pipeline(mesh_device=device)
        except Exception:
            # Close the chip before re-raising -- TTNN cannot open the same device
            # twice in one process, so a leaked handle turns a transient load failure
            # into a permanent one, and on a shared box the chips stay claimed by a
            # workload that has already given up. Mirrors
            # skyreels_ttnn.server.app._open_device_and_pipeline's own rollback.
            _close_device(device)
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            raise

        # _device last: it is the gate the fast path reads first, so publishing it
        # after _pipeline means seeing it set implies the pipeline is there too.
        _pipeline = pipeline
        _mesh_shape = mesh_shape
        _device = device
        return _device, _pipeline


def close() -> None:
    """Close the TTNN mesh device and release the pipeline.

    Rarely needed -- process exit reclaims all TTNN resources automatically. Call this
    only to release hardware mid-process (e.g. to hand the chips back) without
    restarting.
    """
    global _device, _pipeline, _mesh_shape
    with _lock:
        if _device is None:
            return
        _close_device(_device)
        _device = None
        _pipeline = None
        _mesh_shape = None


def _close_device(device) -> None:
    """Release a TTNN mesh device, swallowing any error.

    Shared by close() and the failed-load path so the two cannot drift. Errors are
    swallowed deliberately in both: a failing close must not strand the globals, and on
    the failure path it must not mask the load error the caller actually needs to see.
    """
    try:
        import ttnn

        ttnn.close_mesh_device(device)
    except Exception:
        pass


def _configure_fabric() -> None:
    """Set Blackhole fabric config before opening the mesh device.

    Mirrors TTSkyReelsRunner._configure_fabric / get_pipeline_device_params
    (tt-inference-server's precedent runner): FABRIC_1D, and on Blackhole a tensix MUX
    + ROW dispatch, which FABRIC_1D_MUX requires.
    """
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.MUX if ttnn.device.is_blackhole() else ttnn.FabricTensixConfig.DISABLED,
    )
