# Research plan — identifiability of cohesive laws from load–displacement data

Status labels used throughout: **[D]** demonstrated (by a cited paper or our own
verified runs), **[H]** hypothesis to be tested, **[I]** interpretation / proposal.

---

## 1. Motivation and objectives

Cohesive laws are usually calibrated by assuming a shape (most often bilinear) and
tuning its parameters until a simulation matches a measured load–displacement curve.
Matching the curve does not establish that the law is correct, because the global
response integrates tractions over the whole process zone.

Objectives:
1. **Prediction** — infer the TSL from a load–displacement response.
2. **Uniqueness / identifiability** — determine whether the inferred law is unique, or
   which of its features the data leave unconstrained.
3. **Shape determination** — recover the shape in a physically meaningful, reliable way.

Ultimate goal: replace assumed cohesive laws with laws whose features are justified by
the data, with the unconstrained features explicitly reported.

## 2. Literature basis (project knowledge)

- **Basu, Singh, Yoon, Tomar (AIAA SciTech 2027 manuscript)** [D] T-peel, Al/cyanoacrylate,
  h ≈ 400 µm, H/h ≈ 15.9, ℓ0 = 17 mm, L = 76 mm, E = 70 GPa, ν = 0.33. FE-calibrated
  compliance; constant G_c fails to reproduce the propagation plateau; R-curve G_c(ℓ)
  reproduces it. Cohesive simulations: near-tip bilinear law (σ_c = 20 MPa,
  δ_c = 0.04 mm, G_c = 0.4 kJ/m²) insufficient; tip + tail + bridging bilinear
  contributions reproduce P–δ, ℓ(δ), G∞.
- **Basu, AAE 590 proposal** [H] R-curve makes the shape identifiable; neural law
  trained on P–δ + R-curve through a differentiable reduced-order model.
  *Caveat [I]:* if the R-curve is computed from the same P–δ data with a
  compliance-inferred crack length, it adds no information — only independent
  measurements (e.g. DIC crack length) can.
- **Gavris & Sun (CMAME 2026)** [D] differentiable explicit MPM with interface elements;
  MLP cohesive law trained from synthetic displacement fields; Mode I DCB (cubic law,
  NRMSE 4.4e-2 noise-free) and mixed-mode circular interface. Non-uniqueness for a
  single loading path acknowledged (Remark 4). No experiments. [I] Loading is strongly
  dynamic (≈4 µs histories), which may carry information absent in quasi-static tests.
- **Hao, Han & Bahmani (arXiv 2610.33711, Oct 2026)** [D] differentiable quasi-static FE
  + discrete adjoint; GSM-structured neural cohesive law (PICNN energy, monotone
  inverse resistance). DCB: small-scale bridging (ℓ_cz/H ≈ 0.88) force-only gives
  G_c error < 0.2% but median traction error ≈ 18%; DIC strains reduce it to < 3%;
  large-scale bridging (ℓ_cz/H ≈ 6) recovers shape from force alone in 4/5 runs.
  Appendix B: at fixed G_c and stiffness, peak traction barely affects P–δ under SSB.
  Learned priors and DIC for heterogeneous interfaces. Synthetic, noise-free, same mesh.

Further prior work to review before claiming novelty (TODO, not yet read by us):
Alfano (2006) on shape influence; Jensen et al. (2019) n-segmented multilinear inverse
identification; Bao & Suo (1992); Sills & Thouless (2015) cohesive length scales;
Sørensen & Jacobsen (2003) J-integral approach; and a dedicated search for prior
identifiability analyses of cohesive laws from global data.

## 3. Problem statement (Phase 1)

[I] Proposed clean statement for the paper:

> For a Mode I bonded-beam specimen with known elastic adherends and geometry, and a
> rate-independent, irreversible cohesive law, which functionals of the law are
> identifiable from the global load–displacement record within a given noise level,
> and how does this depend on the dimensionless cohesive-zone length?

Deliverables that would answer it:
1. A dimensional analysis defining the governing groups.
2. A forward sensitivity / identifiability map over those groups.
3. Local (Fisher / Jacobian SVD, profile likelihood) and global (ensembles, "twin"
   laws) identifiability analyses.
4. An inversion method that recovers the identifiable content with quantified
   uncertainty and reports what is not identifiable.

Novelty risk [I]: insensitivity of P–δ to shape under small-scale bridging is known
qualitatively (Bao & Suo; Alfano; Hao et al.). The contribution must be quantitative
and systematic (maps, identifiable subspaces, thresholds vs noise) plus a method,
not a re-demonstration of the qualitative effect.

## 4. Forward problem and law family

**Admissible set A:** Mode I, rate-independent, σ(δ) ≥ 0, σ(0) = 0, fixed initial
stiffness K, finite failure separation δ_f, irreversible damage with secant unloading,
G_c = ∫σ dδ.

**Forward map:** F: σ ∈ A ↦ y = (P(Δ_1), …, P(Δ_N)); observations
y_obs = F(σ_true) + ε, noise model TBD (e.g. Gaussian, std 0.5–1% of P_max).

**Phase 1 shape family (own definition, independent of ABAQUS conventions):**
linear elastic loading to σ_c at δ_0 = σ_c/K, then

    σ(δ) = σ_c · s(x; α),   x = (δ − δ_0)/(δ_f − δ_0) ∈ [0, 1]
    s(x; α) = (e^{−αx} − e^{−α}) / (1 − e^{−α}),   s(x; 0) = 1 − x

- α > 0: convex, exponential-type softening (more brittle as α grows)
- α = 0: linear softening (bilinear law)
- α < 0: concave, toward a trapezoidal / Dugdale-like plateau

Energy: G_c = σ_c δ_0 / 2 + σ_c (δ_f − δ_0) g(α), with g(α) = 1/α − 1/(e^α − 1),
g(0) = 1/2, g ∈ (0, 1). (Verified numerically against quadrature, see DECISIONS.md.)
Given (K, σ_c, G_c, α): δ_f = δ_0 + (G_c − σ_c δ_0/2)/(σ_c g(α)); requires
G_c > σ_c δ_0 / 2.

This family lets us vary **shape at fixed G_c and σ_c**, which isolates the shape
question. Parameters to identify in Phase 1B: θ = (σ_c, G_c, α), K fixed.

### 4a. Parameter bounds (DECISIONS.md D9)

Bounds come from existing experimental evidence, so the synthetic problem stays
physically grounded:

| quantity | range | source / status |
|---|---|---|
| G_c | 0.1 – 0.4 N/mm (reference 0.40) | experimental evidence, user [agreed] |
| K | 8500 MPa/mm | E_a/t, E_a = 3400 MPa assumed [needs source] |
| σ_c | lower/upper TBD; hard cap √(2 G_c K) = 41–82 MPa | adhesive strength data TBD |
| α | TBD (≥ 0 at least) | single-element test |

Admissibility worked example (K = 8500, E′ = 78.6 GPa, H = 6.324 mm) [D, arithmetic]:

| G_c (N/mm) | σ_c (MPa) | ℓ_ch/H | δ_f (linear) mm | φ_e = σ_c²/(2KG_c) |
|---|---|---|---|---|
| 0.1 | 10 | 12.4 | 0.020 | 0.06 |
| 0.1 | 40 | 0.78 | 0.005 | 0.94 |
| 0.4 | 20 | 12.4 | 0.040 | 0.06 |
| 0.4 | 60 | 1.38 | 0.013 | 0.53 |

### 4b. Energy constraints (DECISIONS.md D10 — guiding principle)

**Law level** (every generated or inferred law):
- E1. Exact energy: ∫₀^δ_f σ dδ = G_c, checked by closed form and by quadrature of the
  tabulated/ABAQUS law (relative tolerance 1e-4) [proposed tolerance].
- E2. Energy partition: φ_e = σ_c δ_0/(2G_c) = σ_c²/(2 K G_c) is the fraction of G_c
  stored elastically before damage onset. Require φ_e ≤ φ_max (proposed 0.5) so that the
  law has a real softening branch rather than being elastic-brittle. Report
  φ_e for every case.
- E3. Admissibility: σ ≥ 0, finite δ_f, monotone damage, secant unloading (set A).

**Simulation level** (every ABAQUS run; histories already extracted by
`extract_tpeel_odb.py`):
- E4. Global balance: W_ext = ∫P dΔ = ALLSE + ALLDMD + ALLVD (+ ALLPD = 0 for elastic
  adherends), residual < 1 % of W_ext [proposed].
- E5. Regularization: ALLVD / ALLDMD < 1 % [proposed]; otherwise the run is not part of
  the dataset, and it is reported as failed (rule 3).
- E6. Dissipation vs fracture energy: during steady propagation,
  dALLDMD/d(a) = G_c W (W = 15 mm); deviation < 2 % [proposed]. This links the law to
  the structural response directly and checks the crack-length measure.
- E7. Kinetic energy ≈ 0 (quasi-static), and no snap-back or abort; record the
  stability status for every run.

**Inference level** (Phase 1B onward):
- E8. Parameterize laws as G_c × (normalized shape of unit area), so that E1 holds by
  construction; infer G_c and shape separately.
- E9. Report identifiability separately for (i) G_c, (ii) the energy partition
  φ_e / σ_c, and (iii) the shape at fixed energy. Hypothesis [H]: P–Δ constrains (i)
  tightly and (iii) weakly (H1, H3).
- E10. Inferred laws that violate E1–E3 are rejected, never projected silently.

**Dimensional analysis [I, to be confirmed]:**

    P̂ = f(Δ̂; ℓ_ch/H, K H/E′, α, ℓ0/H, L/H, ν)
    P̂ = P/(W √(E′ G_c H)),   Δ̂ = Δ √(E′/(G_c H)),   ℓ_ch = E′ G_c / σ_c²

In the LEFM limit (ℓ_ch/H → 0), Irwin–Kies with C = f_C(ℓ/H)/(E′W) gives a master
curve P̂(Δ̂) depending only on geometry ratios — i.e. independent of the law. Shape and
strength information lives only in the deviation from this master curve. Reference
value: the SciTech near-tip law gives ℓ_ch/H ≈ 12 (E′ = 78.6 GPa, H = 6.36 mm inferred
from H/h). The actual process-zone length must be measured from the FE damage field.

## 5. Identifiability definitions

- Structural: F injective on A (noise-free).
- Practical: S_ε = {σ ∈ A : ‖F(σ) − y_obs‖ ≤ ε} is small in a law-space metric
  (normalized L2 traction error, G_c error, peak error). ε = noise + numerical error.
- Local tools: sensitivity Jacobian J = ∂y/∂θ (finite differences, job arrays),
  Fisher information JᵀΣ⁻¹J, SVD → identifiable/unidentifiable directions plotted
  as functions of δ; profile likelihood for parametric cases.
- Global tools: multi-start ensembles; explicit search for maximally different laws
  that fit the same data within ε ("twin laws").

Hypotheses:
- **H1** P–Δ fixes G_c accurately regardless of shape.
- **H2** The number of identifiable shape features grows with ℓ_ch/H (or ℓ_cz/H).
- **H3** α (shape) is the least identifiable of (σ_c, G_c, α) at small ℓ_ch/H.

## 6. Phases

- **Phase 0 — infrastructure & verification.** Repo; parameterized template from the
  user's T-peel `.inp`; law tabulation; single cohesive-element test vs input law;
  mesh, increment, viscous-regularization convergence → numerical error floor ε_num;
  regression against the SciTech near-tip bilinear case.
- **Phase 1A — forward sensitivity map.** Grid over (ℓ_ch/H, α) at fixed G_c and K;
  max |ΔP| between shapes relative to ε. Also record crack length, δ*, opening
  profile, J for later use (not used for identification in Phase 1).
- **Phase 1B — parametric identifiability** of θ = (σ_c, G_c, α).
- **Phase 1C — nonparametric inversion** (piecewise-linear and/or neural law) against
  family truths generated on a finer mesh; ensembles and twin-law search.
- **Phase 2 — method design** (AI): differentiable surrogate of ABAQUS, own
  differentiable solver validated against ABAQUS, or simulation-based inference;
  chosen based on Phase 1 results.
- **Phase 3 — extensions:** richer laws (multi-stage, bridging), additional
  observables (crack length, δ*, J), noise robustness.
- **Phase 4 — experiments:** T-peel data, J-integral (J = pΘ + 12M²/(E′H³) with
  calibrated κp² correction) as independent validation.

## 7. Open items (TBD)

- Geometry details from the existing `.inp` (H, W, mesh, element types, units).
- Cohesive layer representation (zero-thickness vs 0.4 mm COH2D4) and value of K.
- ABAQUS/Standard settings: viscous regularization, increment control.
- Law implementation route: built-in tabular damage evolution vs UMAT.
- Noise model and number of samples N along Δ.
- User's plotting style file and manuscript template.
