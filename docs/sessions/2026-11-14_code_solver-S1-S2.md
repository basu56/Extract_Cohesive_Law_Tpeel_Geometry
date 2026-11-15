# 2026-11-14 — Claude Code — differentiable solver S1 (bulk element) and S2 (cohesive interface)

## Context / goal
The user pre-approved continuing through the solver steps without stopping. S1 and S2 of
`docs/solver/S0_S2_INSTRUCTIONS.md`.

## What was discussed or done
- S1: Q4/Q4IM elements, SVK and (added) Hencky energies, sparse AD Newton on tatva. ABAQUS CPE4I
  cantilever decks (Negishi 45856435); comparisons 45856728, 45857247 (commit d57cfca).
- S2: JAX α-family law, cohesive interface, single-element driver. New ABAQUS single-element decks
  (mixed, viscous, quadrature probe; Negishi 45857972) plus re-extraction of the 14 Phase 0 runs.

## Results
- [D] S1: Q4IM = CPE4I to ≤ 2e-8 (linear, ny 2–20); NLGEOM v_tip and moment ≤ 4e-5 to 0.1 rad.
  u_tip differs by 1.25 % (SVK) / 0.30 % (Hencky) of its own small value: constitutive, second order in strain.
  T-peel load-point rotation max 0.0325 rad (sw_s40).
- [D] S2: all inviscid COH2D4 runs ≤ 6.8e-8 σ_c; viscous backward Euler exact until D = 1; COH2D4 uses
  nodal quadrature; gradients verified. Cyclic-run dissipation 1.04e-5 vs G_c (path-integration error,
  same as ABAQUS).
- Details: `docs/SOLVER_LOG.md`, `results/solver/S1_bulk/README.md`, `results/solver/S2_interface/README.md`.

## Decisions
- D21 proposed (Q4IM + Hencky for S3).

## Failed / aborted / not done
- First S1 compare: u_tip threshold not met (kept, reported, explained). First two-block-bar runs failed
  to converge (residual scale at zero reaction): fixed in `solve.py`.
- Post-failure viscous tail rule of ABAQUS not reproduced.
- ABAQUS manual not accessible: citations in ELEMENTS.md / INTERFACE.md are unverified recollections.
- Laptop timings not measured.

## Next steps and open questions
- S3 instructions (T-peel mesh/loading, history scheme, COH2D4 frame under NLGEOM).
- User: D21, the u_tip flag, the manual citations.
