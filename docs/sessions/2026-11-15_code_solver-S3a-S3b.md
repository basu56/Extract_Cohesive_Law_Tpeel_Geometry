# 2026-11-15 — Claude Code — solver S3a and S3b

## Context / goal
- Apply Claude Web's S3 instructions (`apply_solver_S3.sh`, user-provided). Commit 462be90: the
  S3_INSTRUCTIONS, D21 update, D22 and the manual-citation note.
- Carry out S3a (T-peel import, couplings, elastic gate) and S3b (monolithic history,
  robustness), then stop for review.

## What was done
- Confirmed the five reference runs on Negishi; copied their small files to
  `/scratch/bell/basu56/tslid_s3/ref`.
- Wrote the deck importer and the T-peel model with rigid-coupling elimination (chain-rule
  reduction, FD-verified).
- Ran the elastic gate (Negishi jobs 45864387, 45864578, 45866503, 45868517). It missed the gate:
  6e-4 at first, then the following diagnosis.
  - NLGEOM = NO: exact to 3e-8 in RF2, but the upper adherend sat 0.026 mm away from ABAQUS,
    with ABAQUS reporting zero shear.
  - COH2D4 probes (job 45868344) showed the gapped element is corotational. The new kinematics
    were implemented and the gate re-run: linear exact everywhere; NLGEOM 2–3e-5 at T-peel loads.
  - The power-series Hencky diverged; replaced by a closed form.
- S3b:
  - closed-form envelope integrals and the incremental potential;
  - derivation checked against `update_damage` to 1e-14;
  - all 20 S2 runs and the two-block test reproduced;
  - incremental driver with line search, cutback and ABAQUS-like growth;
  - coarse damaged T-peel run with ABAQUS on the same deck (jobs 45868159, 45868519).
- Docs: `docs/solver/TPEEL.md` (new), INTERFACE.md update, SOLVER_LOG S3a/S3b, STATUS,
  `results/solver/S3_tpeel/` (README, CSVs, figure).

## Results
See SOLVER_LOG S3a/S3b; all [D].
- Linear T-peel matches ABAQUS to 3e-8.
- NLGEOM RF2 2.2e-5 / UR3 5.2e-5 at P = 331 N.
- COH2D4 gap kinematics found and implemented.
- Monolithic scheme equals the S2 scheme to 1e-12 σ_c; two-block 43 vs 183 iterations.
- Coarse damaged T-peel: P_peak −2.2e-5; 670 s vs ABAQUS 256 s.
- ABAQUS reference runs take 36–56 min, not 20–25.

## Failed / aborted / not done
- S3a gate (1e-5) not met under NLGEOM; cause not isolated beyond "not bulk energy, not interface".
- ABAQUS 1 %-load run (`s3a_small_tight`) aborted: too many attempts under the 1e-9 controls. It
  was replaced by the 16-increment run.
- The first Hencky T-peel runs (series form) gave garbage while reporting "converged", and the
  first closed-form run gave NaN with plain Newton. Both were superseded.
- Staggered coarse T-peel (job 45868159) was still running at session end. A partial staggered
  run on Bell (150 increments) was used only for a qualitative comparison.

## Decisions
None new (D21 update and D22 came from Claude Web). The cohesive frame default became
"corotational", per the probes. This is not a design choice: it is what ABAQUS does.

## Next steps and open questions
- User:
  - accept the 2–3e-5 NLGEOM floor at T-peel loads, or ask for an investigation of CPE4I NLGEOM
    first;
  - say whether to continue to S3c.
- Add the staggered coarse numbers when job 45868159 finishes.
