# Kernel patch: ttnn 0.78.0 PyPI wheel packaging gap

The `ttnn` PyPI wheel (0.78.0) ships `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
but omits three sibling kernel source files that `FabricConfig::FABRIC_1D` needs to JIT-compile its
mux/relay routing kernels: `fabric_router_mux_extension.cpp`, `fabric_router_relay_extension.cpp`,
`fabric_router_udm_mux_extension.cpp`. Both are present in the real tt-metal v0.78.0 source tree
(github.com/tenstorrent/tt-metal, same tag this bundle's `ttnn` pin matches) -- this looks like an
upstream wheel-packaging omission, not anything about this model or tt-model-manager.

These three files are vendored here byte-for-byte from that source tree and made discoverable via
`TT_METAL_KERNEL_PATH` (an existing tt_metal env var -- see `tt_metal/llrt/rtoptions.cpp` -- that adds
an extra kernel search directory ahead of the installed `ttnn` package's own tree), set in `run.sh`.
Remove this directory and the env var once a `ttnn` release ships these files itself.
