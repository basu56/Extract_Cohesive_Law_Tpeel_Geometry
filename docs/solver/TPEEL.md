# T-peel model in `tslid.fe` (S3a, S3b)

Units: mm, N, MPa, N/mm, s. Code: `src/tslid/fe/inp_import.py`, `tpeel.py`, `incremental.py`,
`interface.py` (cohesive kinematics, incremental potential), `laws_jax.py` (envelope integrals).
Tests: `tests/fe/test_s3a_tpeel_model.py`, `tests/fe/test_s3b_monolithic.py`.
Results: `results/solver/S3_tpeel/`.

## 1. Mesh import (`inp_import.py`)

The mesh is read from the ABAQUS deck that `build_tpeel_inp.py` writes. It is not rebuilt.
The parser reads only the keyword subset that the builder uses. Any other keyword is kept in
`Deck.unparsed`, so nothing is dropped silently. It returns:
- nodes, as 0-based indices in label order;
- CPE4I connectivity, with a counter-clockwise check;
- the COH2D4 row as matched node pairs `(bot, top)`, checked to form a single chain;
- node sets and surfaces;
- the kinematic couplings;
- the solid section (W) and cohesive section (constitutive thickness, W, controls/viscosity);
- materials and the cohesive law, as α-law parameters recovered from the deck's own
  (8-digit) δ_f, which is what ABAQUS reads;
- the boundary conditions and the `*Static` line.

[D] For `sw_s40`: 27,414 nodes, 26,046 CPE4I, 582 COH2D4, bulk area 1111.0068 mm² (analytic
value), and a bond from x = 18.0 to 76.2. The only nodes not attached to any element are the
two reference points.

## 2. Unknowns and rigid couplings (`tpeel.py`)

The full vector U has the `BulkModel` layout: 2 dofs per deck node, then 4 incompatible-mode
amplitudes per element. The reference-point (RP) nodes are ordinary nodes that no element uses,
so their two U dofs are the RP translations. The reduced vector is

    q = [U, θ_1, θ_2],

with one rotation per `*Coupling, *Kinematic`.

**Coupling by elimination.** The slave nodes of a coupling follow the finite rigid motion of
their RP:

    u_s = u_RP + (R(θ) − I)(X_s − X_RP).

Their q entries are therefore dependent. They are listed with the eliminated dofs, never solved
for, and U = lift(q). With NLGEOM = NO the coupling is linearised:
u_s = u_RP + θ e₃ × (X_s − X_RP).

**Derivatives by the chain rule.** The tangent is K_U, the AD tangent of the full energy on the
plain mesh pattern (tatva `sparse.jacfwd`, 34 colours for the T-peel). Then

    r_q = Jᵀ r_U,    K_q = Jᵀ K_U J + G,

where J = dU/dq. J is the identity on the independent dofs. A slave row has 1 on its RP
translation and R′(θ)ΔX on θ. G is the θθ term, −Σ_s r_U,s · R(θ)ΔX.

**Why not tatva's `Lifter`?** It handles linear constraints, and this one is nonlinear in θ.
Also, putting the RP dofs into the element dof map would make the RP row couple to every
leg-top element. That would force about 150 extra colours (JVPs) on every tangent.

[D] Finite-difference checks of r_q and K_q (θ ≈ 0.05 rad): Kv to 1.9e-10, and both θ rows
including G. A rigid translation gives zero energy and zero force.

**Boundary conditions as in the deck.**
- RP2 is encastre: u1, u2 and θ fixed.
- RP1 has u2 prescribed (ramped linearly over the step); u1 and θ are free.
- P is the reduced residual at RP1-u2, i.e. RF2. It is a total force in N, because W is
  included in both sections.

## 3. Cohesive kinematics of COH2D4 with a geometric gap (S3a finding)

The T-peel bond faces sit at y = ±0.2 mm, a geometric gap G = (0, 0.4) mm. The constitutive
thickness is 1.

We determined the ABAQUS COH2D4 separation from single-element probes
(`scripts/fe/build_coh_probe.py`, Negishi job 45868344, `tslid_runs/solver_S3a_cohprobe/`). In
each probe every node's displacement is prescribed, and the element is elastic with K = 8500 in
both directions. [D]

| probe | ABAQUS S22, S12 (MPa) | jump-based ("global") prediction |
|---|---|---|
| gap 0 or 0.4, top translated (10⁻³, 0), NLGEOM NO / YES | 0, 8.5 | 0, 8.5 |
| gap 0.4, rigid rotation 0.1 rad, NLGEOM YES | 0, 0 | −17.0, −339 |
| gap 0.4, rigid rotation 0.1 rad, NLGEOM NO | −16.99, **0** | −17.0, −339 |
| gap 0.4, open 10⁻³ along the normal, then rotate 0.1 rad, NLGEOM YES | 8.5, 0 | −8.5, −340 |

COH2D4 therefore measures the current gap vector in the deformed mid-surface frame minus the
original gap. It does not measure the displacement jump:

    NLGEOM:     δ_n = d·n − G·n₀,  δ_s = d·t − G·t₀,   d = x_top − x_bot (current gap vector)
    NLGEOM=NO:  δ = R₀ᵀ([[u]] − ω J G),  ω = (Δu_mid·n₀)/L  (linearised mid-surface rotation)

Here (t, n) is the deformed mid-surface frame of the element and (t₀, n₀) the reference frame.
`CohesiveInterface(frame="corotational")` and `"corotational_linear"` implement these two forms.
`TPeelModel` picks the one matching the deck's NLGEOM flag. All seven probes are reproduced to
roundoff (`test_cohesive_kinematics_match_coh2d4_probes`). The S2 default, `"global"`, measures
the displacement jump and is correct only with zero gap or zero rotation.

[D] Effect in the T-peel (S3a). In the NLGEOM = NO run, the bonded laminate rotates rigidly by
ω ≈ 0.066 rad. COH2D4 lets the upper adherend slide by ω·0.4 = 0.0263 mm with no shear traction,
whereas `"global"` resists that slide. The reaction is unaffected, but the displacement field
differs by 0.026 mm. Under NLGEOM the shear tractions differed by 0.4–1.5 % of the peak normal
traction.

[I] With damage, a rotation φ of 0.01–0.03 rad across the 0.4 mm gap gives 0.004–0.012 mm of
spurious shear separation. That is the same size as δ₀ = 0.0047 mm at σ_c = 40, so the frame
choice is not cosmetic.

## 4. Incremental potential: damage and displacement in one Newton solve (S3b)

Freeze the history at increment n: δ_max,n and D_v,n per quadrature point. Let r = Δt/μ,
a = μ/(μ+Δt) = 1/(1+r) and b = Δt/(μ+Δt) = r/(1+r). (Inviscid: a = 0, b = 1.)

**Derivation (verified [D]).** The ABAQUS update (`update_damage`, S2) gives

    δ_max,n+1 = max(δ_max,n, δ_m),  D_n+1 = D(δ_max,n+1),  D_v,n+1 = a D_v,n + b D_n+1.

So 1 − D_v,n+1 = a(1 − D_v,n) + b(1 − D(max(δ_max,n, δ_m))). This depends on the separation only
through δ_m. The positive-part traction (1 − D_v,n+1)K0 δ⁺ is therefore radial: it equals
f′(δ_m) δ⁺/δ_m with f′(δ_m) = (1 − D_v,n+1(δ_m)) K0 δ_m. Integrating from 0 gives

    Φ_n = a (1 − D_v,n) K0/2 δ_m² + b Ψ_env(δ_m) + K0/2 ⟨−δ_n⟩₊²,
    Ψ_env = (1 − D_n) K0/2 δ_m²                              (δ_m ≤ δ*)
          = (1 − D_n) K0/2 δ*² + F(δ_m) − F(δ*)               (δ_m > δ*),

with δ* = max(δ_max,n, δ₀), D_n = D(δ_max,n), and F(δ) = ∫_{δ₀}^{δ} T(s) ds the envelope
integral. As μ → 0 this reduces to the Ortiz–Pandolfi (1999) irreversible potential, as the
instructions anticipated. f′ is continuous at δ*. If δ_max,n < δ₀, then D_n = 0 and T(s) = K0 s
up to δ₀.

**Closed-form envelopes (`laws_jax.s_integral`, `envelope_integral`).**
- α law: F = σ_c (δ_f − δ₀) S(x), with S(x) = g(α) − I(1 − x) and
  I(y) = (expm1(αy) − αy)/(α expm1(α)). For |α| < 0.5 a series ratio is used, smooth through
  α = 0. Under AD, dS/dx is exactly the s(x) formula of the same branch.
- TABULAR: T is quadratic in s on each table segment, so F is exact (cumulative sums plus the
  partial segment).

**Gradient safety.** The quadratic branch is written in δ_m², so it is smooth at δ_m = 0. The
other branch takes √ only where δ_m² > δ*² > 0 (double-`where`). There is a clip at x = 1.
First and second derivatives in (δ_n, δ_s) and first derivatives in (K0, σ_c, G_c, α) are finite
at δ_m = 0, δ₀, δ*, δ_f, in compression and at α = 0 [D, `test_potential_gradient_safety`].

**Use.** `TPeelModel(history="monolithic")` (the default) minimises bulk + Φ_n at frozen
history, with one Newton solve per increment, then calls `update_damage` once.
`history="staggered"` keeps the S2 scheme (solve with D_stiff frozen → update → repeat) for
cross-checks.

## 5. Incremental driver (`incremental.py`)

- **Time stepping as in ABAQUS/Standard.** Start at dt0 from `*Static`, cap at dtmax. After two
  consecutive increments converge in ≤ 4 iterations (I_G), grow dt by 1.5. If an increment does
  not converge within 16 iterations, halve dt. Abort below dtmin. The last increment lands on T.
- **Initial guess.** Linear extrapolation of the previous increment on the free dofs (the
  ABAQUS default, `extrapolation=LINEAR`). The prescribed dofs are set to their new values.
- **Newton.** Backtracking line search on ‖r_free‖₂. The Newton direction is a descent
  direction of ‖r‖² even when the softening tangent is indefinite. A non-finite trial residual
  also triggers backtracking.
- **Convergence.** ‖r_free‖∞ ≤ rtol · max(‖r_fixed‖∞, f_floor), with rtol = 1e-10 by default
  (S3c will tune it), or a roundoff-level correction.
- **Logging.** Every increment records its iterations, staggered passes, cutbacks before it,
  line-search halvings, and wall / tangent / solve / history time.
