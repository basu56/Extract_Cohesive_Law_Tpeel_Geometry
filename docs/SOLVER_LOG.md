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
| S0 environment | not started | | | |
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
