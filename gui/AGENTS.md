# GUI Module Refactoring Instructions

## Status
- **Current State**: Basic PyQt GUI exists.
- **Goal**: "Greatly improve utility for real time diagnostics and debugging."

## Tasks

1.  **Real-time Diagnostics**:
    - Add **Energy Plot**: Kinetic, Potential, Internal, Total vs Time. (Critical for stability checks).
    - Add **Timestep Plot**: dt vs Time.
    - Add **Max Density/Temperature Plot**: Track extremes.

2.  **Performance Toggles**:
    - Make the 3D particle view toggleable (it's expensive).
    - Add "Update Rate" control (e.g., update GUI every N steps).

3.  **Simulation Control**:
    - Ensure "Pause", "Resume", "Step" buttons work reliably with the simulation thread.
    - Add "Soft Reset" (reset particles but keep config) and "Hard Reset".

4.  **Data Inspection**:
    - Add a "Particle Inspector": Click or select a particle ID to see its specific values (pos, vel, rho, P).
