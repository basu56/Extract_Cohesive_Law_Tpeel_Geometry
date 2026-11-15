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
| S1 bulk element | not started | | | |
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
