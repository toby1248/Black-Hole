# GPU Module Documentation

## Overview
This module implements CUDA-accelerated physics solvers for TDE-SPH.
It targets NVIDIA RTX 4090 hardware, utilizing massive parallelism.

## Architecture
- **Data Management**: `GPUManager` handles data transfer between Host (CPU/RAM) and Device (GPU/VRAM).
- **Kernels**: Implemented using `numba.cuda` for high performance and Python integration.
- **Arrays**: Uses `cupy` arrays for device-side storage and manipulation.

## Key Components
1. **Gravity**: Tiled N-body solver ($O(N^2)$) using shared memory.
2. **Neighbours**: Brute-force ($O(N^2)$) or Grid-based ($O(N)$) search.
3. **Hydro**: SPH force computation on GPU.

## Requirements
- `cupy-cuda12x`
- `numba`
- NVIDIA Driver supporting CUDA 12.x

## Usage
The `Simulation` class detects if `HAS_CUDA` is true and switches to GPU solvers.
Data is kept on the GPU as much as possible to avoid PCIe bottlenecks.
