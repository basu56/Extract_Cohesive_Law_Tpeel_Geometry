# Inverse phase — Claude Code instructions for I1 (parametric inversion) and I2 (identifiability map)

Written by Claude Web, 2026-11-15. Session log: `docs/sessions/2026-11-15_web_inverse-plan.md`.
Decision: D31 (inverse phase I1–I2 design).

Read first: `CLAUDE.md`, `docs/STATUS.md`, `docs/RESEARCH_PLAN.md` §5 (identifiability
definitions, H1–H3), `docs/solver/ADJOINT.md`, the S5 entries of `docs/SOLVER_LOG.md`, and
D18, D24 and D28–D30.

The solver rules still apply:
- units mm–N–MPa;
- x64 precision;
- no ABAQUS-side changes except the one new run requested below;
- [D]/[H]/[I] labels;
- tatva pinned and unmodified;
- heavy jobs via `sbatch`;
- deterministic environment (`scripts/fe/det_env.sh`) for anything compared bitwise or by
  finite differences.

**Start after the S5 close-out (D30) is reviewed.**

## Why this phase

S5 made the T-peel response differentiable and gave a local identifiability result *within
the α family*: G_c ≈ 0.01 %, σ_c ≈ 0.4–0.7 % and α ± 0.05–0.15, at an assumed noise of
1e-3 P_peak. Two questions remain before any neural law:

- **I1:** Does an actual inversion, using data from another solver and fitted with our solver,
  recover the parameters with the accuracy the Fisher matrix predicts? And what happens with
  the conventional practice of fitting a bilinear law to data from a different shape?
- **I2:** How much of an *arbitrary* law shape does the P–δ curve constrain? Which features of
  σ(δ) are fixed, which are free, and how does that change with ℓ_cz/H (H2) and with an
  added observable?

I2 is the core of the planned contribution. I1 validates the tools I2 relies on.

## Logging

- Create `docs/INVERSE_LOG.md` with the same entry template and rules as
  `docs/SOLVER_LOG.md`: Built / Verification / Runtime / Gain / Cost and limitations / Open
  items, a running summary table, and dated corrections only.
- Write one entry per sub-step. "Gain" states what we now know that we did not know before.
- Results go to `results/inverse/I1_parametric/` and `results/inverse/I2_map/`, each with a
  README. Code goes to `src/tslid/inverse/`, kept separate from `tslid.fe`, and imports it.

## Common definitions (both parts)

**Data.** The synthetic data are ABAQUS P–δ records, which come from a different solver than
the model.

- The mesh is the same node for node. So this avoids the "same code" inverse crime, but not
  the "same mesh" one.
- I1c adds one ABAQUS truth on a finer mesh to address the second.

**Observation operator (D29).** The binned average of P over fixed δ bins of 0.003 mm, applied
identically to data, model and sensitivities. Two windows:

- **pre-peak:** 0 ≤ δ ≤ δ_peak;
- **full:** the D26 window, until the σ_c front reaches L − 15 mm.

**Noise.** Independent Gaussian noise on the binned load, standard deviation
s ∈ {1e-3, 1e-4} P_peak. These are **assumed** levels; state this in every table. Noise is added
to the binned ABAQUS record, with a recorded seed per realization.

**Misfit.** J(θ) = ½ Σ_b (P̄_b(θ) − P̄_b^data)² / s², over the bins of the chosen window.

**Parameters (I1).** θ = (ln σ_c, ln G_c, α). K0 stays at the deck value of 8500 MPa/mm (a
numerical penalty, D10). Bounds follow D9: α ∈ [−5, 10] and σ_c within D9's range. Never leave
the D10 admissible set: reject, or step back from, any iterate that violates it.

**Schedules.** In an inversion every forward run uses its own adaptive schedule. Unlike the
finite-difference checks, nothing is replayed. This is the realistic setting. The
schedule-dependent post-peak part is why the full window is binned (D29, D30).

## I1 — Parametric inversion baseline

### I1a — Optimizer and noise-free cross-solver recovery

1. **`src/tslid/inverse/gauss_newton.py`.** A Levenberg–Marquardt step using the S5a direct
   Jacobian of the binned load: one forward run plus sensitivities per iteration.
   - Use a damping update that follows the gain ratio, with box bounds by projection or a
     log/logit map. Document the choice.
   - Stop when the relative parameter step is below 1e-6, or the misfit reduction is below 1e-8
     of J over two iterations, or after 30 iterations.
   - Log every iterate: θ, J, damping, run time and any cutbacks.
2. **Truth cases.** Four ABAQUS production runs (confirm the job names in `scripts/phase1/`):

   | case | σ_c | α | why |
   |---|---|---|---|
   | `sw_s40` | 40 | 0 | reference |
   | `as_s40_a7` | 40 | 7 | convex long tail; on the D18 weak line |
   | `sw_s21` | 21 | 0 | long zone (ℓ_cz/H ≈ 1.0) |
   | `as_s21_am3` | 21 | −3 | concave |

3. **Noise-free inversion, multi-start.** For each case, run 8 starts: Latin hypercube over the
   bounds, the same seeds for every case. Use both windows.
   - Report the converged θ̂ against the truth, the final J, and whether all starts reach the
     same minimum. If two minima differ by more than 1e-3 in any parameter, report both; that
     would be a sign of non-uniqueness *within* the α family [H: not expected].
4. **Bias from the solver difference, predicted and measured.** Without noise, θ̂ differs from
   the truth only because ABAQUS and our solver differ (≤ 5.5e-4 P_peak, S4a). Predict the bias
   from the linearization,
   Δθ_pred = (Sᵀ S)⁻¹ Sᵀ (P̄_abq − P̄_ours(θ_true)),
   with S the binned Jacobian at the truth. Compare with Δθ measured.
   - **Gate:** agreement within 20 % per parameter, or within 1e-4 absolute where the bias is
     below that.
   - This also gives a number for the paper: the parameter error caused by solver mismatch
     alone.

**Stop for review after I1a**, with an INVERSE_LOG entry.

### I1b — Monte Carlo: does the Fisher matrix predict the scatter?

1. For `sw_s40` and `as_s40_a7`: 50 noise realizations at s = 1e-3 P_peak and 50 at 1e-4,
   pre-peak and full windows. Start from the noise-free θ̂ of I1a.
   - Run as an array job, one thread per realization. There is no licence limit, so up to the
     group cap of 64 cores. This is the throughput advantage over ABAQUS; record wall time and
     core-hours.
2. Compare the empirical covariance of θ̂ with the Cramér–Rao bound F⁻¹ (S5c), evaluated at the
   noise-free θ̂.
   - **Gate:** the empirical standard deviation of each parameter lies within the 95 %
     confidence interval of a standard deviation estimated from 50 samples (about ±20 %) around
     the Cramér–Rao value.
   - Also check coverage: the fraction of realizations whose θ̂ falls inside the 1σ and 2σ
     ellipses, expected about 20 % and 74 % in 3D [H: valid if the problem is close to linear
     at this noise].
   - Also report the mean of θ̂ minus the noise-free θ̂, to check that the noise adds no bias.
3. If the gate fails at s = 1e-3 but passes at 1e-4, report it as nonlinearity at the larger
   noise. It is a result, not a failure.

### I1c — Finer-mesh truth (mesh mismatch)

1. One new ABAQUS run: `sw_s40` at h_coh = 0.05 mm, with every other setting at D16 production
   values. The D16 convergence runs at h = 0.05 used viscosity 1e-3, so they do not qualify.
   Use the existing builder and runner; do not modify them.
2. Invert it noise-free with our solver on the production mesh (h_coh 0.1), with 8 starts as
   in I1a. Report θ̂ minus the truth and the predicted bias, as in I1a.4.
   - This is the parameter error from discretization mismatch alone. It is the honest error
     floor for any inversion of real data with this mesh.

### I1d — The conventional calibration and its model-form error

1. Fit a **bilinear law (α fixed at 0)** to the `as_s40_a7` data (truth α = 7), noise-free,
   both windows, with 8 starts.
   - Report the fitted (σ_c, G_c) against the truth, the final misfit relative to the noise
     levels, and where along δ the residual sits.
2. Repeat with truth `as_s21_am3` (α = −3).
3. The question this answers: would a practitioner fitting a bilinear law notice the wrong
   shape from the P–δ fit alone, at s = 1e-3 and at 1e-4? Report the χ² per bin of the best
   bilinear fit at each noise level.
   - [H] At 1e-3 the misfit may be acceptable while G_c and σ_c are biased. That would be the
     paper's motivating example, measured in our own setting.

**Stop for review after I1b–I1d**, with INVERSE_LOG entries.

## I2 — Piecewise-law identifiability map

### I2.0 — Law parameterization for a function-space analysis

1. **Support as a parameter.** `PiecewiseLawParams` currently fixes the knot offsets u_k, and
   with them δ_f. Add a differentiable support scale ℓ, with u_k = ℓ ξ_k for fixed
   0 = ξ_0 < … < ξ_m = 1, so that δ_f = δ_0 + ℓ. The parameters become (ln σ_c, ln ℓ, t_1 … t_{m−1}).
   - Check the S5 forward gate and the per-increment partials (≤ 1e-7 against finite
     differences) for the new parameter.
   - Re-run the S5b piecewise adjoint-vs-direct check (target as accepted in D30).
2. **Metric.** To compare different m, express the traction parameters so that the Euclidean
   norm approximates the L² norm of the traction perturbation over the support:
   t̃_k = t_k √(w_k / ℓ), where w_k are the trapezoid weights of knot k. Do the Fisher analysis
   in (ln σ_c, ln ℓ, t̃). Plot eigenvectors as traction perturbations δT(δ) on the physical δ
   axis. Document the transformation.
3. **Truth laws.** `PiecewiseLawParams.from_alpha(α-law, m)` of each truth α law, so the model is
   exact at the truth. Check that the piecewise forward run reproduces the α-law run to within
   the piecewise approximation (report the P difference for m = 15, 30, 60).

### I2a — The map at the reference case

1. Use the direct method (S5a; p = m + 1, which is cheap at this size) on **production**
   `sw_s40`, for m ∈ {15, 30, 60}. Compute the binned Jacobian and F for both windows.
   Report the cost.
2. **Report:**
   - **The spectrum.** Eigenvalues of F at s = 1e-3, 1e-4 and 1e-5 P_peak (they scale as 1/s²;
     plot once with the noise thresholds marked).
   - **Identifiable modes.** Call a mode identifiable at noise s if its Cramér–Rao standard
     deviation, as an RMS traction perturbation over the support, is below 0.05 σ_c. Report the
     number of identifiable modes against s and the window. The threshold is a choice; state it
     and show how the count changes at 0.02 and 0.1 σ_c.
   - **Eigenvectors as δT(δ),** overlaid on the truth law: the leading 4 (constrained) and the
     weakest 4 (free).
   - **Physically meaningful functionals.** Cramér–Rao standard deviations of linear functionals
     of the law, computed from F⁻¹ (pseudo-inverse with a documented cutoff) and the functional's
     gradient:
     - G_c;
     - σ_c;
     - δ_f;
     - the first moment δ̄ = ∫δ T dδ / G_c;
     - the energy fraction beyond δ_0 + ℓ/2.

     [H1, H3] G_c should be the best-determined functional regardless of shape detail.
3. **Convergence in m.** The leading eigenvalues and the δT(δ) of the leading modes should agree
   between m = 30 and 60 (eigenvalues within 10 %, eigenvector correlation > 0.95). If not,
   report it; the map would then depend on the discretization, not only the data.

### I2b — Nonlinear check: twin laws

The Fisher analysis is linear. Test it with the full forward model.

1. For the 3 weakest modes at m = 30, perturb the truth law along each mode by amplitudes a
   chosen so that:
   - the linear prediction gives Δχ² = 1 (a 1σ change at s = 1e-3);
   - the traction change is visibly large: max |δT| = 0.2 σ_c. Choose the sign so tractions stay
     ≥ 0 and the law stays D10-admissible; if impossible, reduce the amplitude and report.
2. Run each perturbed law forward (production) and compute the actual binned misfit to the truth
   run.
   - **Gate:** at the Δχ² = 1 amplitude, the actual Δχ² is within a factor of 2 of the
     prediction. At the large amplitude, report the actual misfit and P–δ difference.
3. **Figure `i2b.png`:** the truth law and twin laws (left), their P–δ curves and differences in
   units of s (right). If twin laws whose tractions differ by 20 % of σ_c produce P–δ
   differences below the noise, this is the paper's non-uniqueness figure, made with the full
   model rather than a linear argument.
4. **Optional, if time permits:** search directly for maximally different twin laws. Maximize
   ‖δT‖ subject to Δχ² ≤ 1, positivity and D10, using the adjoint for gradients. Report it as an
   open item if not done.

### I2c — The map across ℓ_cz/H (H2)

1. Repeat I2a (m = 30, both windows) for `sw_s58` (ℓ_cz/H 0.17) and `sw_s21` (1.00), alongside
   `sw_s40` (0.42).
2. Report the number of identifiable modes against ℓ_cz/H at each noise level, and the leading
   eigenvectors per case.
   - [H2] The count grows with ℓ_cz/H. Report what is found, including if it doesn't.
3. This is the first version of the identifiability map: modes against (ℓ_cz/H, noise, window).

### I2d — Does an extra observable help? (preview)

1. The mouth openings (MOUTH_U, MOUTH_L) already have sensitivities (S5a). Add them as a second
   observable with assumed noise s_u ∈ {1e-3, 1e-4} mm (DIC-like; state as assumed). Bin them
   on the same δ bins and treat them as independent of the load noise.
2. Recompute F = F_P + F_u for `sw_s40` and `sw_s21`, and report the change in the identifiable
   mode count and in the weakest eigenvectors.
3. This previews the experiment-design contribution. Crack length is not differentiable in the
   current outputs, because the crack tip moves element by element; list it as an open item.
   Do not add it now.

**Stop for review after I2a–I2d**, with INVERSE_LOG entries and a short synthesis in the I2
README:
- what the map shows;
- what H1–H3 look like now;
- what that implies for the neural-law step (Hao et al.).

## Exit criteria

- **I1:**
  - noise-free cross-solver recovery with the predicted bias confirmed (I1a);
  - the Monte Carlo scatter compared with Cramér–Rao, including coverage (I1b);
  - the mesh-mismatch bias measured (I1c);
  - the bilinear model-form experiment reported (I1d).
- **I2:**
  - the support parameter and L² metric verified (I2.0);
  - the spectrum, identifiable modes, functionals and m-convergence at `sw_s40` (I2a);
  - twin laws with the full model (I2b);
  - the mode count against ℓ_cz/H (I2c);
  - the observable preview (I2d).
- INVERSE_LOG entries for every sub-step, STATUS and session logs per the context protocol.

## Out of scope

- Neural laws and Hao et al.'s reproduction (next phase).
- Real experimental data.
- Correlated or non-Gaussian noise models.
- Global search beyond the multi-starts and the optional twin-law maximization.
- Batching or GPU, and the increment strategy (S4b).
- Any change to production settings D16, D24 and D25.
