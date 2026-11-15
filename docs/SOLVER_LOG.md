# Solver log — differentiable T-peel solver on tatva

One entry per step (S0, S1, …). Write an entry when the step's exit criteria are met, or when
a step is abandoned or changed. **Never edit an old entry.** If something later turns out to be
wrong, add a dated correction under that entry. This file records what each step gained us,
both for the manuscript and for the short contribution on the tatva module (D19, D20).

Labels: [D] demonstrated by a test or run in this repo, [H] hypothesis, [I] interpretation.

## Running summary

Update this table at each entry.

| Step | Status | Key verification result | Runtime / throughput | Gain (one line) |
|---|---|---|---|---|
| S0 environment | **done** 2026-11-14 | 16/16 tests on Bell, Negishi node, GitHub CI; `test_laws` OK on Py 3.6 | suite 14–15 s (Bell login, Negishi node) | one locked JAX/tatva env on every machine, with CI |
| S1 bulk element | **done**, 1 flagged item 2026-11-14 | Q4IM = CPE4I to 1e-8 (linear); NLGEOM v, M ≤ 4e-5; u_tip 1.25 % SVK / 0.30 % Hencky (own value) | ny=20 linear 1.8 s/solve; NL 2.6 s/incr (ABAQUS 50 s total) | locking-free bulk element matching CPE4I; constitutive gap quantified |
| S2 cohesive interface | not started | | | |
| S3 T-peel forward vs ABAQUS | not started | | | |
| S4 throughput (batched / GPU) | not started | | | |
| S5 gradients (adjoint) | not started | | | |

Reference ABAQUS costs:
- [D] ~20–25 min per T-peel run on 1 CPU (26k CPE4I, full output; `docs/STATUS.md`).
- [D] Licence-limited to ~12 concurrent jobs, i.e. ≈ 30 runs/h.
- [D] No gradients: finite differences need ≥ 2 runs per parameter.

## Entry template

Copy this template for each step.

```markdown
## Sx — <title> — <date> — commit <hash>

**Built.** What code and tests were added (files).

**Verification.** Each test with its threshold and measured value, [D].
Failed or partial tests are listed here too.

**Runtime.** Wall time of the relevant solve(s); hardware (laptop CPU / Negishi CPU / GPU);
for comparisons, the ABAQUS number on the same problem. Measured, not estimated.

**Gain.** What this step makes possible that was not possible before. Be concrete:
"we now know the Quad4 locking error at ny = 20 is X %", not "better accuracy".

**Cost / limitations.** What it does not do yet; known differences from ABAQUS; tatva
workarounds (see docs/solver/TATVA_NOTES.md).

**Open items.** Carried to the next step.
```

<!-- Entries start below this line. Newest at the bottom. -->

## S0 — Environment and skeleton — 2026-11-14 — commit 3c3fd07 (code), log in the following commit

**Built.**
- `pyproject.toml` (package `tslid`, src layout, extra `fe`: jax, tatva==0.11.6, jax-autovmap,
  numpy, scipy, matplotlib, pytest; `slow` marker), `uv.lock`, `.python-version` = 3.12.12.
- `src/tslid/fe/` with `__init__.py` (x64 on, `TATVA_VERSION`) and docstring stubs for
  `elements`, `kinematics`, `laws_jax`, `interface`, `meshing`, `solve`.
- `tests/fe/test_s0_smoke.py` (5 tests), `.github/workflows/tests.yml` + README badge,
  `scripts/slurm/run_fe_tests.sh`, `docs/solver/ENVIRONMENT.md`.
- Locked: Python 3.12.12, jax 0.12.0.1, tatva 0.11.6; uv 0.9.5.

**Verification.** [D]
- tatva version == 0.11.6: pass. x64 active (`jnp.zeros(1).dtype == float64`): pass.
- Quad4 `Operator.integrate(1.0)` on the unit square, 2×2 and 5×5 node meshes: |I − 1| ≤ 1e-14, pass.
- Extra test (not in the instructions): `import tslid, tslid.laws` does not load jax: pass.
- Full suite `pytest tests/test_laws.py tests/fe -m "not slow"`, 16 tests:
  - Bell login node (AMD EPYC 7502): 16 passed.
  - Negishi compute node a023 (AMD EPYC 7763), job 45855196: 16 passed.
  - GitHub Actions (ubuntu-latest), commits 9962f73, 6b39bea, 3c3fd07: success.
- `tests/test_laws.py` under the old interpreter (system Python 3.6.8, unittest), Bell and Negishi: 11/11 OK.
- Partial / deviation: the "laptop" leg was replaced by the Bell login node (user's choice
  this session); the laptop has not been tested. First Negishi job 45855040 also passed 16/16
  but ran Python 3.12.13 (`.python-version` was "3.12"); the patch was then pinned and re-run.

**Runtime.** Test suite wall time: 14.3 s (Bell login), 14.8 s (Negishi node, job 45855196;
26 s job elapsed). The first Negishi job took 67.7 s, 25.6 s of it in the first Quad4 test —
first-use JIT/import cost on a fresh environment [I]. No solver timings yet (S1).

**Gain.** Every machine (Bell, Negishi, CI) now runs the same locked JAX + tatva stack in
float64, and every push is tested automatically. The ABAQUS-side code is unaffected:
`tslid.laws` still runs on Python 3.6 and does not import JAX (guarded by a test).

**Cost / limitations.**
- `requires-python = ">=3.11"` applies to the whole package (not only the `fe` extra, which
  pyproject cannot express). Harmless because the ABAQUS side never installs the package;
  it uses `sys.path`.
- The lock forks by Python (3.11 → jax 0.10.2; 3.12 → jax 0.12.0.1), hence the exact Python pin.
- No tatva workarounds needed; `docs/solver/TATVA_NOTES.md` not created.
- Bell login node: one transient `os error 11` during the first `uv sync` (re-run worked).

**Open items.**
- GPU: Negishi partition `gpu` (g000–g004, 3 GPUs each) exists; GPU model and whether
  account `tomar` may use it are TBD (needed for S4).
- Laptop environment not yet verified.
- S1 waits for the user's go-ahead.

## S1 — Bulk element and kinematics — 2026-11-14 — code d57cfca, log in the following commit

**Built.**
- `kinematics.py`: plane-strain energies `linear`, `svk`, and `hencky` (log strain, power series
  of ln C, NaN-free at H = 0; added to test the SVK-vs-hypoelastic hypothesis).
- `elements.py`: `BulkModel` with Q4 (tatva Quad4, 2 × 2) and Q4IM (4 Wilson/Taylor incompatible
  modes as extra global unknowns, enhanced displacement gradient with j₀/j scaling; Simo–Armero
  F = F_c + H_enh under finite rotation).
- `solve.py`: `SparseNewton`, a Python-loop Newton with the tangent from `tatva.sparse.jacfwd`
  (pattern from the element dof map, distance-2 colouring: 18 colours for Q4, 34 for Q4IM) and a
  SciPy SuperLU solve.
- `meshing.py` (beam using the T-peel `graded()`, edge loads, 5-element patch); `benchmarks.py` (cantilever).
- ABAQUS side: `scripts/fe/build_cantilever_inp.py`, `extract_cantilever_odb.py`, `run_s1_abaqus.sh`;
  comparison `s1_compare.py`, `s1_plot.py`, `run_s1_compare.sh`.
- `docs/solver/ELEMENTS.md`; tests `tests/fe/test_s1_patch.py` (6), `test_s1_cantilever.py` (6 + 2 slow).

**Verification.** [D] Details in `results/solver/S1_bulk/README.md`.
1. Patch test (distorted 5-element patch, Q4 and Q4IM × linear and SVK): energy error ≤ 2e-16,
   nodal error ≤ 1e-18 mm, mode amplitudes 0. Threshold 1e-12: **pass**. SVK rigid rotation 0.3 rad
   gives zero energy.
2. Cantilever vs Timoshenko (reported only): Q4IM −0.12 … −0.48 %, ABAQUS identical; Q4 −19 % … −0.67 %.
3. Linear vs ABAQUS CPE4I (job 45856435): Q4IM tip deflection and reaction within ≤ 2.1e-8 at every
   ny (threshold 5e-4 at ny = 20): **pass**. Q4 locking: −19.3 / −11.1 / −2.29 / −0.195 % at
   ny = 2 / 4 / 8 / 20. Distorted mesh: Q4IM 4.7e-6 vs CPE4I (information).
4. NLGEOM to θ = 0.101 rad (P = 1400 N, 20 increments, tight ABAQUS tolerances):
   v_tip ≤ 3.6e-5, clamp moment ≤ 3.0e-5 (SVK); **pass**.
   **Flag:** u_tip relative to its own value is −1.25 % (SVK) and +0.30 % (Hencky), so it **fails 0.1 %**
   if the threshold is applied to that component alone. Relative to |v_tip| it is 4.5e-4 / 1.1e-4.
   Cause [D]: constitutive (changing the strain measure alone changes it 4× and flips its sign;
   it grows ∝ load²). Residual cause [H]: hypoelastic Cauchy vs hyperelastic Kirchhoff stress.
   Measured T-peel load-point rotation max = 0.0325 rad (sw_s40 UR3). At that rotation all
   differences are ≤ 1.3e-4 of |v| (SVK) and ≤ 4e-5 (Hencky).
5. Timing: see Runtime.
- Fast suite: 17 passed on Bell (2 slow deselected; the slow NLGEOM tests replicate the job above).

**Runtime.** Negishi node a001 (EPYC 7763, 4 cores): linear ny = 20 Q4IM (73 242 dofs) 1.8 s per
solve after JIT, 3.1 s first call; ABAQUS CPE4I 6.2 s wall (includes pre-processing and licence).
NLGEOM 20 increments: SVK 54 s including JIT (2.6 s per increment, 3 Newton iterations each);
ABAQUS 49.7 s. Hencky 202 s (≈ 4× SVK per tangent). Laptop not measured.

**Gain.** We now have a bulk element that reproduces ABAQUS CPE4I to 1e-8 in the linear regime and
to ≤ 4e-5 in P-relevant quantities under NLGEOM up to 3× the T-peel rotation. So any later P–Δ
mismatch in S3 cannot come from the bulk discretisation. We also know that plain Q4 at the
production mesh is 0.195 % too stiff (5× the floor), and that the remaining NLGEOM gap is
constitutive and second order in strain.

**Cost / limitations.**
- SVK and Hencky are hyperelastic; ABAQUS is hypoelastic. The gap is second order in strain and
  visible only in the axial tip displacement.
- The Hencky series is valid for strains ≲ 5 % (elastic Al only) and costs about 4× SVK.
- ABAQUS manual citations in `ELEMENTS.md` are from memory and unverified (manual not reachable).
- Each solve recompiles when the mesh changes (JIT); per-solve times exclude this.
- No tatva workarounds needed.

**Open items.**
- User: accept the u_tip flag (proposed: judge NLGEOM by v, M and |Δu|/|v|), and choose the S3 bulk
  energy. Proposed D21: Q4IM + Hencky, with SVK as a cross-check.
- User: confirm the ABAQUS manual sections cited in `ELEMENTS.md`.
- Laptop timing not done.
