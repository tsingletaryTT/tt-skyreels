#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Setup script for tt-skyreels (SkyReels-V2-DF-1.3B-540P on Tenstorrent Blackhole)."""

from setuptools import setup, find_packages

setup(
    name="skyreels-ttnn",
    version="0.1.0",
    author="Tenstorrent",
    description="SkyReels-V2-DF-1.3B-540P (T2V) on Tenstorrent Blackhole via TTNN",
    url="https://github.com/tenstorrent/tt-skyreels",
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy<3",
        "diffusers>=0.32.1,<0.41",
        "transformers>=4.30.0,<6",
        "accelerate>=0.20.0,<2",
        "safetensors>=0.4.0,<1",
        "sentencepiece",
        "protobuf",
        "imageio>=2.30,<3",
        "imageio-ffmpeg>=0.4,<1",
        # tt-metal and ttnn must be installed separately (not on PyPI)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "httpx>=0.24.0",  # fastapi's TestClient needs it; otherwise server tests
                               # fail at import rather than skipping
            "pyyaml",  # tests/test_server_app.py reads tt_model_package.yaml directly
        ],
        # The ASGI serving surface (skyreels_ttnn/server/). Matches the packages
        # tt-model-manager's tt-dit-server kind installs for this app, so a bundle and a
        # local `pip install -e .[serve]` run the same stack.
        "serve": [
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
            "pydantic>=2.0.0",
        ],
    },
)
