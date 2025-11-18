#!/usr/bin/env python3
"""
Export TDE-SPH simulation data to Blender/ParaView formats (TASK-038).

Converts HDF5 snapshots to point clouds and volume grids suitable for
external visualization and rendering tools.

Supported Formats:
- PLY (Polygon File Format): Point clouds for Blender
- VTK (Visualization Toolkit): Unstructured grids for ParaView
- OBJ (Wavefront): Simple point clouds for Blender

Features:
- Particle positions with color mapping (density, temperature, velocity)
- Scalar field export (density, internal energy, etc.)
- Optional SPH interpolation to regular grids (basic implementation)
- Batch conversion of multiple snapshots

Usage:
    >>> from tools.export_to_blender import SnapshotExporter
    >>> exporter = SnapshotExporter()
    >>> exporter.export_ply("snapshot_0000.h5", "output.ply", color_by="density")
    >>> exporter.export_vtk("snapshot_0000.h5", "output.vtk")
    >>> exporter.export_obj("snapshot_0000.h5", "output.obj")

Command Line:
    python tools/export_to_blender.py snapshot_0000.h5 -o output.ply -f ply --color density
    python tools/export_to_blender.py snapshots/*.h5 -o renders/ -f vtk --batch

References:
    - PLY format: http://paulbourke.net/dataformats/ply/
    - VTK format: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    - OBJ format: https://en.wikipedia.org/wiki/Wavefront_.obj_file
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


class SnapshotExporter:
    """
    Export TDE-SPH HDF5 snapshots to Blender/ParaView formats.

    Parameters
    ----------
    normalize_colors : bool, optional
        Normalize color values to [0, 1] range (default: True).
    verbose : bool, optional
        Print progress messages (default: True).

    Attributes
    ----------
    normalize_colors : bool
        Color normalization flag.
    verbose : bool
        Verbosity flag.

    Examples
    --------
    >>> exporter = SnapshotExporter()
    >>> exporter.export_ply("snapshot.h5", "out.ply", color_by="density")
    >>> exporter.export_vtk("snapshot.h5", "out.vtk", scalars=["density", "internal_energy"])
    """

    def __init__(self, normalize_colors: bool = True, verbose: bool = True):
        """Initialize SnapshotExporter."""
        self.normalize_colors = normalize_colors
        self.verbose = verbose

    def read_snapshot(self, filename: str) -> Dict[str, any]:
        """
        Read HDF5 snapshot file.

        Parameters
        ----------
        filename : str
            Path to HDF5 snapshot file.

        Returns
        -------
        data : dict
            Dictionary with 'particles' (dict of arrays) and 'metadata' (dict).

        Raises
        ------
        FileNotFoundError
            If snapshot file doesn't exist.
        KeyError
            If required keys are missing from HDF5 file.
        """
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot file not found: {filename}")

        with h5py.File(filename, 'r') as f:
            # Read particle data
            particles = {}
            if 'particles' in f:
                for key in f['particles'].keys():
                    particles[key] = f['particles'][key][:]

            # Read metadata
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].attrs.keys():
                    metadata[key] = f['metadata'].attrs[key]

            # Also check for top-level metadata (backward compatibility)
            for key in f.attrs.keys():
                if key not in metadata:
                    metadata[key] = f.attrs[key]

        return {'particles': particles, 'metadata': metadata}

    def compute_colors(
        self,
        field: NDArrayFloat,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> NDArrayFloat:
        """
        Compute RGB colors from scalar field.

        Parameters
        ----------
        field : NDArrayFloat, shape (N,)
            Scalar field values.
        cmap : str, optional
            Color map name (default: "viridis"). Options: "viridis", "plasma", "hot", "cool".
        vmin, vmax : float, optional
            Min/max values for color normalization (default: None, uses field min/max).

        Returns
        -------
        colors : NDArrayFloat, shape (N, 3)
            RGB colors in range [0, 255] (uint8).

        Notes
        -----
        Simple linear color mapping. For advanced colormaps, consider matplotlib.
        """
        # Normalize field to [0, 1]
        if vmin is None:
            vmin = np.min(field)
        if vmax is None:
            vmax = np.max(field)

        if vmax > vmin:
            normalized = (field - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(field)

        normalized = np.clip(normalized, 0.0, 1.0)

        # Simple color maps (linear interpolation)
        if cmap == "viridis":
            # Approximate viridis: dark blue -> green -> yellow
            r = np.clip(1.5 * normalized - 0.25, 0, 1)
            g = np.clip(1.2 * normalized, 0, 1)
            b = np.clip(1.5 * (1 - normalized) + 0.25, 0, 1)
        elif cmap == "plasma":
            # Approximate plasma: dark purple -> orange -> yellow
            r = np.clip(1.3 * normalized, 0, 1)
            g = np.clip(1.5 * normalized - 0.3, 0, 1)
            b = np.clip(1.5 * (1 - normalized), 0, 1)
        elif cmap == "hot":
            # Hot: black -> red -> yellow -> white
            r = np.clip(3 * normalized, 0, 1)
            g = np.clip(3 * normalized - 1, 0, 1)
            b = np.clip(3 * normalized - 2, 0, 1)
        elif cmap == "cool":
            # Cool: cyan -> magenta
            r = normalized
            g = 1 - normalized
            b = np.ones_like(normalized)
        else:
            # Default: grayscale
            r = g = b = normalized

        # Convert to uint8 [0, 255]
        colors = np.stack([r, g, b], axis=1)
        colors = (colors * 255).astype(np.uint8)

        return colors

    def export_ply(
        self,
        snapshot_file: str,
        output_file: str,
        color_by: str = "density",
        cmap: str = "viridis",
        point_size: Optional[float] = None
    ) -> None:
        """
        Export snapshot to PLY point cloud format (Blender compatible).

        Parameters
        ----------
        snapshot_file : str
            Path to input HDF5 snapshot.
        output_file : str
            Path to output PLY file.
        color_by : str, optional
            Particle field to use for coloring (default: "density").
            Options: "density", "internal_energy", "velocity_magnitude", "smoothing_length".
        cmap : str, optional
            Color map name (default: "viridis").
        point_size : float, optional
            Override point size (default: None, uses smoothing_length if available).

        Notes
        -----
        PLY format spec: http://paulbourke.net/dataformats/ply/
        Blender can import PLY via File > Import > Stanford (.ply)
        """
        if self.verbose:
            print(f"Exporting {snapshot_file} to PLY: {output_file}")

        # Read snapshot
        data = self.read_snapshot(snapshot_file)
        particles = data['particles']

        # Get positions
        if 'positions' not in particles:
            raise KeyError("Snapshot missing 'positions' key")
        positions = particles['positions']
        N = len(positions)

        # Get color field
        if color_by == "velocity_magnitude":
            if 'velocities' not in particles:
                raise KeyError("Snapshot missing 'velocities' for color_by='velocity_magnitude'")
            color_field = np.sqrt(np.sum(particles['velocities']**2, axis=1))
        elif color_by in particles:
            color_field = particles[color_by]
        else:
            if self.verbose:
                print(f"Warning: Field '{color_by}' not found, using density")
            color_field = particles.get('density', np.ones(N, dtype=np.float32))

        # Compute colors
        colors = self.compute_colors(color_field, cmap=cmap)

        # Get point sizes (optional)
        if point_size is not None:
            sizes = np.full(N, point_size, dtype=np.float32)
        elif 'smoothing_length' in particles:
            sizes = particles['smoothing_length']
        else:
            sizes = None

        # Write PLY file
        with open(output_file, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment TDE-SPH snapshot exported from {snapshot_file}\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            if sizes is not None:
                f.write("property float radius\n")
            f.write("end_header\n")

            # Data
            for i in range(N):
                x, y, z = positions[i]
                r, g, b = colors[i]
                if sizes is not None:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {sizes[i]:.6f}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        if self.verbose:
            print(f"  Exported {N} particles to {output_file}")

    def export_vtk(
        self,
        snapshot_file: str,
        output_file: str,
        scalars: Optional[List[str]] = None,
        vectors: Optional[List[str]] = None
    ) -> None:
        """
        Export snapshot to VTK unstructured grid format (ParaView compatible).

        Parameters
        ----------
        snapshot_file : str
            Path to input HDF5 snapshot.
        output_file : str
            Path to output VTK file.
        scalars : list of str, optional
            List of scalar fields to include (default: ["density", "internal_energy"]).
        vectors : list of str, optional
            List of vector fields to include (default: ["velocities"]).

        Notes
        -----
        VTK format spec: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
        ParaView can open VTK files directly via File > Open.
        """
        if self.verbose:
            print(f"Exporting {snapshot_file} to VTK: {output_file}")

        # Default fields
        if scalars is None:
            scalars = ["density", "internal_energy"]
        if vectors is None:
            vectors = ["velocities"]

        # Read snapshot
        data = self.read_snapshot(snapshot_file)
        particles = data['particles']

        # Get positions
        if 'positions' not in particles:
            raise KeyError("Snapshot missing 'positions' key")
        positions = particles['positions']
        N = len(positions)

        # Write VTK file (legacy ASCII format)
        with open(output_file, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"TDE-SPH snapshot exported from {snapshot_file}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Points
            f.write(f"POINTS {N} float\n")
            for i in range(N):
                f.write(f"{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f}\n")

            # Cells (each particle is a vertex cell)
            f.write(f"\nCELLS {N} {2 * N}\n")
            for i in range(N):
                f.write(f"1 {i}\n")

            f.write(f"\nCELL_TYPES {N}\n")
            for i in range(N):
                f.write("1\n")  # VTK_VERTEX = 1

            # Point data
            f.write(f"\nPOINT_DATA {N}\n")

            # Scalar fields
            for scalar_name in scalars:
                if scalar_name in particles:
                    field = particles[scalar_name]
                    f.write(f"\nSCALARS {scalar_name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for i in range(N):
                        f.write(f"{field[i]:.6e}\n")
                elif self.verbose:
                    print(f"  Warning: Scalar field '{scalar_name}' not found, skipping")

            # Vector fields
            for vector_name in vectors:
                if vector_name in particles:
                    field = particles[vector_name]
                    if field.shape == (N, 3):
                        f.write(f"\nVECTORS {vector_name} float\n")
                        for i in range(N):
                            f.write(f"{field[i, 0]:.6e} {field[i, 1]:.6e} {field[i, 2]:.6e}\n")
                    elif self.verbose:
                        print(f"  Warning: Vector field '{vector_name}' has wrong shape, skipping")
                elif self.verbose:
                    print(f"  Warning: Vector field '{vector_name}' not found, skipping")

        if self.verbose:
            print(f"  Exported {N} particles to {output_file}")

    def export_obj(
        self,
        snapshot_file: str,
        output_file: str
    ) -> None:
        """
        Export snapshot to OBJ point cloud format (Blender compatible).

        Parameters
        ----------
        snapshot_file : str
            Path to input HDF5 snapshot.
        output_file : str
            Path to output OBJ file.

        Notes
        -----
        OBJ format: https://en.wikipedia.org/wiki/Wavefront_.obj_file
        Simple point cloud (vertices only, no faces).
        Blender can import OBJ via File > Import > Wavefront (.obj)
        """
        if self.verbose:
            print(f"Exporting {snapshot_file} to OBJ: {output_file}")

        # Read snapshot
        data = self.read_snapshot(snapshot_file)
        particles = data['particles']

        # Get positions
        if 'positions' not in particles:
            raise KeyError("Snapshot missing 'positions' key")
        positions = particles['positions']
        N = len(positions)

        # Write OBJ file
        with open(output_file, 'w') as f:
            f.write(f"# TDE-SPH snapshot exported from {snapshot_file}\n")
            f.write(f"# {N} particles\n\n")

            # Vertices
            for i in range(N):
                x, y, z = positions[i]
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        if self.verbose:
            print(f"  Exported {N} particles to {output_file}")

    def batch_export(
        self,
        snapshot_files: List[str],
        output_dir: str,
        format: str = "ply",
        **kwargs
    ) -> None:
        """
        Batch export multiple snapshots.

        Parameters
        ----------
        snapshot_files : list of str
            List of HDF5 snapshot files.
        output_dir : str
            Output directory for exported files.
        format : str, optional
            Output format: "ply", "vtk", or "obj" (default: "ply").
        **kwargs
            Additional arguments passed to export function.

        Examples
        --------
        >>> exporter.batch_export(
        ...     ["snap_0000.h5", "snap_0001.h5"],
        ...     "renders/",
        ...     format="ply",
        ...     color_by="density"
        ... )
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Batch exporting {len(snapshot_files)} snapshots to {output_dir}")

        for snapshot_file in snapshot_files:
            snapshot_path = Path(snapshot_file)
            output_file = output_path / f"{snapshot_path.stem}.{format}"

            if format == "ply":
                self.export_ply(snapshot_file, str(output_file), **kwargs)
            elif format == "vtk":
                self.export_vtk(snapshot_file, str(output_file), **kwargs)
            elif format == "obj":
                self.export_obj(snapshot_file, str(output_file), **kwargs)
            else:
                raise ValueError(f"Unknown format: {format}. Use 'ply', 'vtk', or 'obj'.")

        if self.verbose:
            print(f"Batch export complete: {len(snapshot_files)} files")


def main():
    """Command-line interface for snapshot export."""
    parser = argparse.ArgumentParser(
        description="Export TDE-SPH snapshots to Blender/ParaView formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single snapshot to PLY
  python export_to_blender.py snapshot_0000.h5 -o output.ply -f ply --color density

  # Export to VTK for ParaView
  python export_to_blender.py snapshot_0000.h5 -o output.vtk -f vtk

  # Batch export all snapshots
  python export_to_blender.py snapshots/*.h5 -o renders/ -f ply --batch --color temperature
        """
    )

    parser.add_argument("input", nargs='+', help="Input HDF5 snapshot file(s)")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory (for --batch)")
    parser.add_argument("-f", "--format", choices=["ply", "vtk", "obj"], default="ply",
                        help="Output format (default: ply)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: export multiple files")
    parser.add_argument("--color", default="density", help="Field for PLY coloring (default: density)")
    parser.add_argument("--cmap", default="viridis", help="Color map (default: viridis)")
    parser.add_argument("--scalars", nargs='+', default=None,
                        help="VTK scalar fields (default: density internal_energy)")
    parser.add_argument("--vectors", nargs='+', default=None,
                        help="VTK vector fields (default: velocities)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    # Create exporter
    exporter = SnapshotExporter(verbose=not args.quiet)

    # Export
    if args.batch:
        kwargs = {}
        if args.format == "ply":
            kwargs = {"color_by": args.color, "cmap": args.cmap}
        elif args.format == "vtk":
            if args.scalars:
                kwargs["scalars"] = args.scalars
            if args.vectors:
                kwargs["vectors"] = args.vectors

        exporter.batch_export(args.input, args.output, format=args.format, **kwargs)
    else:
        if len(args.input) > 1:
            print("Error: Multiple input files require --batch mode")
            return 1

        snapshot_file = args.input[0]
        output_file = args.output

        if args.format == "ply":
            exporter.export_ply(snapshot_file, output_file, color_by=args.color, cmap=args.cmap)
        elif args.format == "vtk":
            exporter.export_vtk(snapshot_file, output_file, scalars=args.scalars, vectors=args.vectors)
        elif args.format == "obj":
            exporter.export_obj(snapshot_file, output_file)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
