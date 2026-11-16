# Inverse log — parametric inversion and identifiability maps

One entry per sub-step (I1a, I1b, …), in the same format as `docs/SOLVER_LOG.md`. Write it when
the sub-step's exit criteria are met, or when it is abandoned or changed. **Never edit an old
entry**; add dated corrections under it.

Labels: [D] demonstrated, [H] hypothesis, [I] interpretation. Noise levels are **assumed**
unless stated otherwise.

## Running summary

| Step | Status | Key result | Runtime / core-hours | Gain (one line) |
|---|---|---|---|---|
| I1a noise-free recovery, bias prediction | not started | | | |
| I1b Monte Carlo vs Cramér–Rao | not started | | | |
| I1c finer-mesh truth | not started | | | |
| I1d bilinear model-form error | not started | | | |
| I2.0 support parameter, L² metric | not started | | | |
| I2a map at `sw_s40` | not started | | | |
| I2b twin laws (nonlinear) | not started | | | |
| I2c map vs ℓ_cz/H | not started | | | |
| I2d extra observable | not started | | | |

## Entry template

```markdown
## Ix — <title> — <date> — commit <hash>

**Built.** Code, scripts, tests (files).

**Verification.** Each gate with its threshold and measured value, [D]. Failed or partial gates
are listed here too.

**Runtime.** Wall time and core-hours; hardware; number of forward runs.

**Gain.** What we now know that we did not know before, stated concretely.

**Cost / limitations.** Assumptions (noise model, window, threshold choices), what is local or
linear, what is not yet tested.

**Open items.** Carried to the next step.
```

<!-- Entries start below this line. Newest at the bottom. -->
