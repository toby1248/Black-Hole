#!/usr/bin/env python3
"""
Convert TDE-SPH HDF5 snapshots into JSON consumable by the web visualizer.

The web app expects a JSON object per snapshot with the following fields:
- time: simulation time (float)
- step: integer snapshot index (best-effort inference)
- n_particles: number of particles
- positions: list of [x, y, z] triples
- density, temperature, internal_energy, velocity_magnitude, pressure, entropy:
  arrays (lists) of length n_particles

Derived quantities:
- velocity_magnitude is computed from velocities if not present
- pressure falls back to (gamma-1) * rho * u if missing
- entropy falls back to P / rho**gamma if missing
- temperature defaults to internal_energy if not present

Use the optional stride/limit arguments to downsample large snapshots for faster
web loading.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

DEFAULT_GAMMA = 5.0 / 3.0
DEFAULT_THREADS = 8
SIGNIFICANT_FIGURES = 5


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    """Expand glob patterns and return existing files."""
    paths: List[Path] = []
    for pattern in inputs:
        matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    if not paths:
        raise FileNotFoundError("No input HDF5 files found for given patterns.")
    return paths


def _read_dataset(
    group: h5py.Group,
    name: str,
    stride: int,
    limit: Optional[int],
) -> Optional[np.ndarray]:
    """Return dataset as float32 array with optional downsampling."""
    if name not in group:
        return None
    data = np.asarray(group[name], dtype=np.float32)
    if stride > 1:
        data = data[::stride]
    if limit is not None:
        data = data[:limit]
    return data


def _infer_step(path: Path, attrs: Dict[str, Any]) -> int:
    """Try to infer a step/index from metadata or filename."""
    for key in ("step", "iteration", "timestep", "step_index"):
        if key in attrs:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue

    match = re.search(r"(\d+)(?=\.h5$)", path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return 0


def _ensure_list(arr: np.ndarray, flatten_positions: bool = False) -> List[Any]:
    """Convert numpy array to a JSON-serializable list."""
    if arr.ndim == 2 and arr.shape[1] == 3 and not flatten_positions:
        # Preserve [x, y, z] triples for readability
        return arr.tolist()
    return arr.reshape(-1).tolist()


def _round_sig_array(arr: np.ndarray, sig: int = SIGNIFICANT_FIGURES) -> np.ndarray:
    """Round a numeric array to the requested significant figures."""
    if arr is None:
        return arr
    arr = np.asarray(arr, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mags = np.floor(np.log10(np.abs(arr)))
    mags[~np.isfinite(mags)] = 0.0
    decimals = (sig - mags - 1).astype(int)
    # Clamp decimals to avoid extreme scaling/overflow
    decimals = np.clip(decimals, -6, 12)
    scale = np.power(10.0, decimals, dtype=np.float32)
    # Element-wise rounding: round(x * scale) / scale
    return np.round(arr * scale) / scale


def _round_sig_scalar(val: float, sig: int = SIGNIFICANT_FIGURES) -> float:
    """Round a scalar to the requested significant figures."""
    if val == 0 or not math.isfinite(val):
        return float(val)
    magnitude = math.floor(math.log10(abs(val)))
    decimals = sig - magnitude - 1
    return round(val, decimals)


def load_hdf5_snapshot(
    path: Path,
    stride: int = 1,
    limit: Optional[int] = None,
    gamma: float = DEFAULT_GAMMA,
    flatten_positions: bool = False,
) -> Dict[str, Any]:
    """Load and convert a single HDF5 snapshot to the web JSON schema."""
    with h5py.File(path, "r") as f:
        if "particles" not in f:
            raise KeyError(f"Missing 'particles' group in {path}")
        particles = f["particles"]

        positions = _read_dataset(particles, "positions", stride, limit)
        if positions is None:
            raise KeyError(f"'positions' dataset missing in {path}")

        density = _read_dataset(particles, "density", stride, limit)
        if density is None:
            raise KeyError(f"'density' dataset missing in {path}")

        internal_energy = _read_dataset(particles, "internal_energy", stride, limit)
        if internal_energy is None:
            raise KeyError(f"'internal_energy' dataset missing in {path}")

        velocities = _read_dataset(particles, "velocities", stride, limit)
        velocity_magnitude = _read_dataset(
            particles, "velocity_magnitude", stride, limit
        )
        if velocity_magnitude is None and velocities is not None:
            velocity_magnitude = np.linalg.norm(velocities, axis=1).astype(np.float32)
        if velocity_magnitude is None:
            velocity_magnitude = np.zeros(positions.shape[0], dtype=np.float32)

        pressure = _read_dataset(particles, "pressure", stride, limit)
        if pressure is None and density is not None:
            pressure = (gamma - 1.0) * density * internal_energy

        temperature = _read_dataset(particles, "temperature", stride, limit)
        if temperature is None:
            temperature = internal_energy.copy()

        entropy = _read_dataset(particles, "entropy", stride, limit)
        if entropy is None and pressure is not None and density is not None:
            entropy = pressure / np.maximum(density, 1e-30) ** gamma
        if entropy is None:
            entropy = np.zeros(positions.shape[0], dtype=np.float32)

        # Time/metadata
        attrs: Dict[str, Any] = dict(f.attrs)
        if "metadata" in f:
            attrs.update(dict(f["metadata"].attrs))

        time_val = float(attrs.get("time", attrs.get("simulation_time", 0.0)))
        step = _infer_step(path, attrs)
        bh_mass = attrs.get("bh_mass", attrs.get("M_bh", None))
        bh_spin = attrs.get("bh_spin", attrs.get("a", None))
        bh_mass_out = _round_sig_scalar(float(bh_mass), sig=SIGNIFICANT_FIGURES) if bh_mass is not None else None
        bh_spin_out = _round_sig_scalar(float(bh_spin), sig=SIGNIFICANT_FIGURES) if bh_spin is not None else None

        # Precision reduction
        positions = _round_sig_array(positions)
        density = _round_sig_array(density)
        internal_energy = _round_sig_array(internal_energy)
        temperature = _round_sig_array(temperature)
        velocity_magnitude = _round_sig_array(velocity_magnitude)
        if pressure is not None:
            pressure = _round_sig_array(pressure)
        entropy = _round_sig_array(entropy)

        snapshot = {
            "time": _round_sig_scalar(time_val),
            "step": step,
            "n_particles": int(positions.shape[0]),
            "positions": _ensure_list(positions, flatten_positions=flatten_positions),
            "density": density.tolist(),
            "temperature": temperature.tolist(),
            "internal_energy": internal_energy.tolist(),
            "velocity_magnitude": velocity_magnitude.tolist(),
            "pressure": pressure.tolist() if pressure is not None else [],
            "entropy": entropy.tolist(),
            "bh_mass": bh_mass_out,
            "bh_spin": bh_spin_out,
        }

    return snapshot


def convert_file(
    input_path: Path,
    output_path: Path,
    stride: int,
    limit: Optional[int],
    gamma: float,
    flatten_positions: bool,
    indent: Optional[int],
) -> None:
    """Convert one snapshot and write JSON to disk."""
    snapshot = load_hdf5_snapshot(
        input_path,
        stride=stride,
        limit=limit,
        gamma=gamma,
        flatten_positions=flatten_positions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=indent)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TDE-SPH HDF5 snapshots into JSON that the web visualizer loads."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="HDF5 snapshot files or glob patterns (e.g. outputs/snapshot_*.h5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output file or directory. "
            "For multiple inputs, provide a directory (files keep their stem). "
            "Defaults to writing alongside each input with .json extension."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Adiabatic index used for pressure/entropy fallback (default: 5/3).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth particle to downsample (default: 1, keep all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard cap on particle count after striding.",
    )
    parser.add_argument(
        "--flatten-positions",
        action="store_true",
        help="Store positions as a flat list instead of [[x,y,z], ...].",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for JSON output (default: 2).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of worker threads for parallel conversion (default: {DEFAULT_THREADS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)

    multiple_inputs = len(input_paths) > 1
    output_arg = Path(args.output) if args.output else None

    def resolve_output_path(input_path: Path) -> Path:
        if output_arg is None:
            return input_path.with_suffix(".json")
        if multiple_inputs or (output_arg.exists() and output_arg.is_dir()):
            return output_arg / f"{input_path.stem}.json"
        if output_arg.suffix:
            return output_arg
        return output_arg / f"{input_path.stem}.json"

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = {
            executor.submit(
                convert_file,
                input_path=input_path,
                output_path=resolve_output_path(input_path),
                stride=max(1, args.stride),
                limit=args.limit,
                gamma=args.gamma,
                flatten_positions=args.flatten_positions,
                indent=args.indent,
            ): input_path
            for input_path in input_paths
        }

        for future in as_completed(futures):
            input_path = futures[future]
            try:
                output_path = future.result()
                print(f"Wrote {output_path}")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Failed to convert {input_path}: {exc}")


if __name__ == "__main__":
    main()
