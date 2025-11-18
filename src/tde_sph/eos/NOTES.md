# EOS Module — Implementation Notes

## Phase 1 & 2 Implementation

### IdealGas EOS (COMPLETE)
- ✅ Implements full EOS interface
- ✅ Arbitrary adiabatic index γ
- ✅ Physical constants in CGS for temperature calculations
- ✅ Safe handling of negative values
- ✅ FP32 optimized

## Phase 3 Implementation Notes

### RadiationGas EOS
**Status**: ✅ COMPLETE

**Key Design Decisions**:
- Temperature iteration: Newton-Raphson with damping factor 0.5 for stability
- Convergence tolerance: 1e-6 relative (50 iterations max)
- Initial guess: gas-only temperature formula
- Temperature floor: 1 K minimum enforced

**Implementation Details**:
- Combined pressure: P = P_gas + P_rad = (γ-1)ρu_gas + (1/3)aT⁴
- Total internal energy: u_total = u_gas + u_rad
- Sound speed: c_s² = (γP_gas + (4/3)P_rad)/ρ
- Energy partitioning methods: `gas_energy()`, `radiation_energy()`
- Beta parameter β = P_gas/P_total for regime diagnostic
- Radiation constant a = 7.5657e-15 erg/(cm³·K⁴) in CGS

**Temperature Solver**:
```python
# Newton-Raphson iteration
f(T) = u_total - u_gas(T) - u_rad(T, ρ)
f'(T) = -du_gas/dT - du_rad/dT
T_new = T - 0.5 * f(T) / f'(T)  # Damping for stability
```

**Regime Transitions**:
- Gas-dominated (β ≈ 1): T < 10⁶ K, behaves like IdealGas
- Transition (β ≈ 0.5): T ~ 10⁶-10⁷ K
- Radiation-dominated (β ≈ 0): T > 10⁷ K, P ∝ T⁴

## Testing Notes
- Ideal gas validation against analytic polytrope
- Radiation gas limit tests needed
- Performance benchmarking for iterative T solver
