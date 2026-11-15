# Sensitivities of the discrete T-peel response: direct method and discrete adjoint

Step S5 (D28; plan `docs/solver/S5_INSTRUCTIONS.md`). Code: `src/tslid/fe/sensitivity.py` (direct,
S5a); the adjoint (`adjoint.py`) is S5b. Units: mm, N, MPa, s.

What is differentiated is the **discrete** response as computed:
- the realized increment schedule;
- the converged states;
- the monolithic incremental potential (D22).

It is not the derivative of the continuous problem. Derivatives with respect to the schedule, the mesh
or the geometry are out of scope.

## 1. The discrete problem, written for our code

Increments n = 1 … N with end loads λ_n (displacement control) and Δt_n. Neither depends on θ.

**Unknowns.** q (n_q) = node displacements, IM amplitudes, RP translations and RP rotations
(`TPeelModel`). There are two kinds of fixed dofs, and both have zero sensitivity:
- **Prescribed dofs** take values λ_n·(BC). These don't depend on θ.
- **Slave dofs of the rigid couplings** are eliminated. The element energies read slave
  displacements from (u_RP, θ_RP) via the lift, never from q's slave entries (`ElementSystem`).

The rest is free: f = `free`.

**History.** h = (dmax, Dv) at the cohesive integration points (nodal two-point rule). D is derived:
D = D(dmax; θ).

**Incremental potential** at frozen history (`interface.psi_incremental`). With a_n = μ/(μ+Δt_n) and
b_n = Δt_n/(μ+Δt_n) (inviscid: a = 0, b = 1):

    Π_n(q; h_{n-1}, θ) = Π_bulk(q) + W Σ_p w_p [ a_n (1 - Dv_{n-1}) K0/2 dm² + b_n Ψ_env(dm; dmax_{n-1}, θ)
                                                  + K0/2 <-δn>₊² ],

where:
- dm² = <δn>₊² + δs², and δ = δ(q) are the corotational separations;
- Ψ_env = (1 - D(dmax_{n-1})) K0/2 dm² when dm ≤ d* = max(dmax_{n-1}, d0);
- otherwise Ψ_env = (1 - D(dmax_{n-1})) K0/2 d*² + F(dm) - F(d*), where F is the envelope integral
  (closed form, `laws_jax.envelope_integral`).

The law parameters θ enter through:
- K0 (stiffness, compression penalty and d0 = σc/K0);
- σc, Gc and α (D, F, d0, df).

**Equilibrium.** R_n(q_n, h_{n-1}, θ) := ∇_q Π_n restricted to f = 0.
The tangent K_n = ∇²_qq Π_n (f × f) is a Hessian, so it is **symmetric**. It is indefinite in
softening.

**History update** (`interface.update_damage`):

    dmax_n = dm(q_n)      if dm(q_n) > dmax_{n-1}   (loading)
           = dmax_{n-1}   otherwise                  (unloading / no change; a tie counts here)
    D_n    = D(dmax_n; θ)
    Dv_n   = (Δt_n D_n + μ Dv_{n-1}) / (μ + Δt_n)    (= D_n when μ = 0)

So H_n depends on:
- q only through dm, and only on loading points;
- dmax_{n-1} only on non-loading points;
- Dv_{n-1} only when μ > 0;
- θ through D.

**Outputs.**
- P_n = RF2 at RP1 = [∇_q Π_n]_{RP1,U2}. This is the component of the full gradient at a prescribed
  dof (the reaction), so P depends on θ explicitly, not only through q.
- The mouth openings are components of q (checked not to be slave dofs).

So Y_n = Y(q_n, h_{n-1}, θ) with Y = ∇_q Π_n at an index, or q at an index.

## 2. Direct (forward) sensitivities

Differentiate R_n(q_n(θ), h_{n-1}(θ), θ) = 0 along a parameter direction e_j, with
S^q_n = dq_n/dθ_j (zero on fixed dofs) and S^h_n = dh_n/dθ_j:

    K_n S^q_{n,f} = -[ ∂R_n/∂h · S^h_{n-1} + ∂R_n/∂θ · e_j ]_f                       (D1)
    S^h_n         =  ∂H_n/∂q · S^q_n + ∂H_n/∂h · S^h_{n-1} + ∂H_n/∂θ · e_j          (D2)
    dy_n/dθ_j     =  ∂Y_n/∂q · S^q_n + ∂Y_n/∂h · S^h_{n-1} + ∂Y_n/∂θ · e_j          (D3)

with S^h_0 = 0 (zero initial history). These are the equations of S5_INSTRUCTIONS with the signs as
there [D: tests, section 5].

**Implementation.**
- **(D1).** The right-hand side is one `jax.jvp` of q ↦ ∇_q Π (the jitted element-assembly gradient)
  with respect to (h, θ), in the direction (S^h_{n-1}, e_j). The tangents of a_n and b_n are zero.
  The p directions are vmapped.
  - The solve uses the Newton system **at the converged q_n**: one Hessian (the same element
    assembly, IM condensation and PARDISO-sym as Newton) and one numerical factorization, with p
    right-hand sides (`ElementSystem.hessian_multi`, `recover_multi`).
  - Newton itself never factorizes at the converged point: it stops after evaluating the residual
    there. So this costs one extra Hessian and one extra factorization per increment; the price is
    measured in S5a.
- **(D3) and (D2).** One jvp of ∇_q Π (whole vector) and one of H, now with respect to (q, h, θ) in
  the direction (S^q_n, S^h_{n-1}, e_j).
  - The free part of the first jvp is the residual of (D1). It is reported as `res_free` (relative
    to the right-hand side), a consistency check of the linear solve.
  - Its component at RP1 U2 is dP/dθ_j.

**Parameter coordinates.** θ = (ln K0, ln σc, ln Gc, α). The law-space tangent of direction j is
diag(K0, σc, Gc, 1) e_j (`alpha_log_basis`), and FD perturbations use the same coordinates
(`perturb_alpha_log`).

**Storage.** Only S^h_{n-1}, which is p × (history size); no history of states is kept.

## 3. Discrete adjoint (S5b)

Objective J = Σ_n φ_n(y_n), with g_n = ∂φ_n/∂y_n. Introduce multipliers μ_n (on f) for R_n = 0 and
η_n for h_n − H_n = 0:

    L = Σ_n [ φ_n(Y_n(q_n, h_{n-1}, θ)) − μ_nᵀ R_n(q_n, h_{n-1}, θ) − η_nᵀ (h_n − H_n(q_n, h_{n-1}, θ)) ].

**Stationarity in q_{n,f}.** q_n appears only in increment n:

    (∂Y_n/∂q)ᵀ g_n − K_nᵀ μ_n + (∂H_n/∂q)ᵀ η_n = 0
    ⇒  K_n μ_n = (∂Y_n/∂q)_fᵀ g_n + (∂H_n/∂q)_fᵀ η_n                               (A1)

This uses K_nᵀ = K_n, so the adjoint system has the Newton matrix: same pattern, analysis and
factorization type.

**Stationarity in h_{n-1}.** It appears in the constraint of increment n−1 (as −η_{n−1}) and in R_n,
Y_n and H_n:

    η_{n-1} = (∂Y_n/∂h)ᵀ g_n + (∂H_n/∂h)ᵀ η_n − (∂R_n/∂h)ᵀ μ_n                      (A2)

with η_N = 0, since h_N is used by no later increment.

**Gradient.**

    dJ/dθ = Σ_n [ (∂Y_n/∂θ)ᵀ g_n + (∂H_n/∂θ)ᵀ η_n − (∂R_n/∂θ)ᵀ μ_n ]               (A3)

**Sign and consistency check.** Substitute (D1)–(D3) into Σ g_nᵀ dy_n and use (A1)–(A2), summing by
parts over n. Then Σ g_nᵀ dy_n = (A3). So the adjoint gradient equals Sᵀ g from the direct method
identically (the S5b gate, ≤ 1e-10).

(A1)–(A3) match S5_INSTRUCTIONS and Hao, Han & Bahmani (arXiv 2610.33711), Eqs. 57–62, up to their
notation. Each transposed partial is one `jax.vjp` of R, H or Y at the stored (q_n, h_{n-1}, a_n, b_n).
The storage per increment is measured in S5b.

## 4. Kinks, failure, tolerance

- **Loading switch.** The derivative of the active branch is used (tie → unloading). Each increment
  records the points within a relative 1e-8 of the switch: dm vs d* = max(dmax_{n-1}, d0), so
  initiation is included (`n_switch`). It also records the points with dmax within 1e-8 of df
  (`n_fail`; D(·) has a kink there).
  - A finite difference whose ± runs straddle such a switch is invalid at that increment. The step
    study reports those increments and the error with them excluded.
- **D = 1.** In the fully failed state (inviscid), the traction and its θ-derivative are exactly 0.
  The derivatives are finite there, since all branches use the double-where pattern [D: test_s5_partials].
  With viscosity, Dv < 1 still carries traction and a non-zero derivative.
- **Tolerance.** (D1)–(A3) assume R_n = 0 exactly. Two consequences:
  - Verification uses rtol 1e-10 with schedule replay.
  - The production tolerance (rtol 1e-6) changes S by ‖S(1e-6) − S(1e-10)‖ / ‖S(1e-10)‖, measured in
    S5a on the coarse mesh.
- **Frozen schedule.** Finite-difference runs replay the realized schedule (`incremental.run(schedule=…)`,
  driver `--schedule <base>_sens.npz` with exact times). A perturbed run that needs a cutback inside a
  scheduled increment adds sub-increments: a different discrete problem. This is reported (increment
  counts).
- **Determinism (D27).** FD differences at small steps are only meaningful when repeated runs agree to
  roundoff. FD runs use `scripts/fe/det_env.sh`:
  - one pinned thread;
  - MKL_CBWR=COMPATIBLE;
  - single-threaded XLA.

  The SuperLU solvers are a second, independent check. Bit-identity of repeated runs is shown in S5a.

## 5. Verification (S5.0, S5a)

| check | tolerance | where |
|---|---|---|
| partials ∂R/∂θ, ∂H/∂θ, ∂Y/∂θ vs central FD (Richardson), at frozen q, h; points in each regime (elastic, softening, unloading, failed, compressive), inviscid and viscous, α ∈ {−3, 0, 0.3, 5}; T-peel state | ≤ 1e-7 | `tests/fe/test_s5_partials.py` |
| single cohesive element, cyclic mixed-mode path with compression, μ ∈ {0, 2e-3}: recursion vs jax.jacfwd of the explicit secant-law history | ≤ 1e-10 | `tests/fe/test_s5a_sensitivity.py` |
| two-block bar (S2, monolithic), μ ∈ {0, 0.05}: recursion vs Richardson central FD, same load steps, Newton to roundoff | ≤ 1e-9 | same |
| coarse T-peel: condensed = uncondensed; res_free; first 20 increments vs FD with replay | ≤ 1e-9; ≤ 1e-8; ≤ 1e-5 | same |
| coarse T-peel, whole curve, step study 1e-3 … 1e-7, deterministic | ≤ 1e-5: ln K0 1.3e-6, ln σ_c 1.02e-5 (marginal), ln G_c 6.1e-7, α 4.0e-6 at the best steps; FD-limited (failure-event increments for h ≥ 1e-5, 1/h noise below 1e-6) | S5a (SOLVER_LOG) |
