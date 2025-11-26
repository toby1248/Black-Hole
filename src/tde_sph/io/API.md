# IO Module API

## Class: `HDF5Writer`

### `write_snapshot(filename, particles, time, metadata=None)`
Writes full snapshot.
- **particles**: Dict of arrays (pos, vel, mass, u, rho, h).
- **metadata**: Dict of simulation parameters.

### `read_snapshot(filename, load_metadata=True) -> Dict`
Reads snapshot.
- Returns dict with `particles` (dict of arrays) and `metadata`.

## Functions
- `write_snapshot(...)`: Convenience wrapper.
- `read_snapshot(...)`: Convenience wrapper.
