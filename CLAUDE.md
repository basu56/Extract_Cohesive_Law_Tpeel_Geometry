# CLAUDE.md — Identifiability of cohesive traction–separation laws from load–displacement data

Read this file at the start of every session. Detailed formulation lives in
`docs/RESEARCH_PLAN.md`; every settled choice is logged in `docs/DECISIONS.md`.
Read both before proposing methods, writing analysis code, or changing a study design.

**This repository is the single source of truth shared by Claude Code (cluster) and
Claude Web.** Follow `docs/CONTEXT_PROTOCOL.md`: at session start read `docs/STATUS.md`
and the newest files in `docs/sessions/`; at session end write a session log and update
`docs/STATUS.md`. Background on the existing ABAQUS model: `docs/EXISTING_MODEL.md`.

## What this project is

PhD research (Dibya Jyoti Basu, Purdue AAE, advisor V. Tomar). Target journal:
Computer Methods in Applied Mechanics and Engineering (CMAME). The work must be
defensible to a strong computational-mechanics audience; a clean, well-posed problem
statement is the main selling point.

Core question (Phase 1, synthetic only):
> Given only a simulated global load–displacement record of a Mode I bonded-beam
> (T-peel / DCB-type) specimen, which features of the cohesive traction–separation law
> (TSL) — fracture energy, strength, softening shape — can be recovered, and how does
> this depend on the cohesive-zone length relative to the structure?

Three research objectives: (1) prediction of the TSL from P–Δ, (2) uniqueness and
identifiability, (3) reliable, physically meaningful shape determination.
Objective 2 is studied first; any AI/inversion method is designed after the
identifiability results exist.

## Current phase

**Phase 0 — infrastructure and verification** (see `docs/RESEARCH_PLAN.md` §6).
Experimental data are explicitly out of scope until Phase 4. Do not add crack
bridging or multi-stage laws yet; Phase 1 uses the one-parameter shape family in
`docs/RESEARCH_PLAN.md` §4.

## Non-negotiable scientific rules

1. Never invent numerical results, equations, references, or conclusions. If a value
   is unknown, write `TBD` and ask.
2. Label every claim as **demonstrated** (by our runs or a cited paper),
   **hypothesis**, or **interpretation**. Keep these separate in notes and figures.
3. Report failed, non-converged, or aborted runs; never silently drop them.
4. Avoid the inverse crime: synthetic "truth" data are generated on a finer mesh
   (and with a different law representation) than the one used for inversion.
5. No identifiability claim is valid unless differences exceed the numerical error
   floor (mesh + increment + viscous regularization) established in Phase 0.
6. Challenge the user's assumptions when evidence warrants; do not just agree.
7. Explain the reasoning behind a new method before implementing it.
8. **Energy first.** Every law, simulation and inferred law must satisfy the energy
   constraints in `docs/RESEARCH_PLAN.md` §4b (exact G_c, energy partition, global energy
   balance, dissipation = G_c × area, small viscous dissipation). Parameter ranges come
   from experimental evidence (§4a), not from convenience.

## Cluster rules (Purdue HPC, SLURM)

- Do NOT run ABAQUS or heavy Python on login nodes. Submit with `sbatch`
  (job arrays for sweeps). Quick syntax checks and small post-processing are fine.
- Never edit the user's original `.inp` files in place. Copy them into
  `abaqus/templates/` and parameterize the copy.
- Raw ABAQUS outputs (`.odb`, `.dat`, `.msg`, `.res`, ...) go to scratch, never git.
- Cluster specifics (source: TPeel_cohesive_zone campaign03 Stage F, Nov 2026):
  - Cluster name: **Negishi** (Bell compute was decommissioned 2026-10-31; the Bell
    login node and `/scratch/bell` were still reachable on 2026-11-14 but cannot run jobs)
  - ABAQUS module / version: `module load intel/19.1.3.304 abaqus/2026`; batch jobs
    need `bash -lc` so the module system loads. Abaqus 2025 (used on Bell) is not
    installed on Negishi; an unchanged re-run gave identical results on 2025 and 2026.
  - SLURM account / partition: `--account=tomar --partition=cpu` (64 cores)
  - Scratch root: `/scratch/negishi/basu56/`; runs in `tslid_runs/<study>/`; Negishi clone
    `/scratch/negishi/basu56/Extract_Cohesive_Law_Tpeel_Geometry` (pulls from the bare repo
    `repo/Extract_Cohesive_Law_Tpeel_Geometry.git`). GitHub-connected clone:
    `/scratch/bell/basu56/Extract_Cohesive_Law_Tpeel_Geometry` (remotes `origin` = GitHub,
    `negishi` = bare repo). Workflow: commit on Bell → push origin + negishi → `git pull` on Negishi.
  - Submit from the Negishi repo root (scripts find the repo via `SLURM_SUBMIT_DIR`); batch
    scripts use `#!/bin/bash -l` so `module` works.
  - GitHub access: SSH only (`git@github.com:...`). HTTPS has no credentials, and Negishi
    has no GitHub key: push from Bell, and sync Negishi through a bare repo on Negishi
    scratch (same pattern as `TPeel_cohesive_zone`)
  - Licence: shared ECN FlexNet; in practice ~12 concurrent 1-cpu Standard jobs, the rest queue ("license request queued")
  - Cost reference: ~15 min per fracture run (campaign03 decks)

## Repository layout (target)

```
abaqus/templates/     parameterized .inp templates (copied from originals)
abaqus/umat/          user subroutines, if needed
src/tslid/laws.py     TSL definitions (shape family, tabulation, Abaqus tables)
src/tslid/inpgen.py   case generation from templates + metadata JSON
src/tslid/extract.py  abaqus-python ODB extraction -> .npz
src/tslid/analysis/   sensitivity, Jacobian/SVD, profile likelihood, inversion
scripts/slurm/        submission scripts
tests/                unit tests (law integrals, tabulation, single-element check)
docs/                 RESEARCH_PLAN.md, DECISIONS.md
results/              small derived data + figures only (raw data on scratch)
```

## Coding conventions

- Python 3, modular, documented functions with units in docstrings (SI internally;
  state the unit system of every `.inp` template explicitly).
- Every simulated case has an ID and a JSON metadata file: law parameters,
  nondimensional groups, mesh, solver settings, ABAQUS version, git commit hash.
- Note that `abaqus python` may be a different interpreter from the analysis
  environment; keep ODB-extraction scripts dependency-free.
- Tests before sweeps: a new law implementation must pass the single-element
  traction–separation check before it is used in any structural run.
- Figures: follow the user's existing plotting format (style file TBD — ask).

## When unsure

Stop and ask rather than guess, especially about geometry, units, solver settings,
or anything that would change a logged decision.
