# 2026-11-14 — Claude Code — repo setup and review of the existing model

## Context / goal
Start the project "AI-based prediction of fracture constitutive laws in adhesives".
Review `add_project_context.sh` (from the Claude Web session), set up the shared repo,
review the existing T-peel ABAQUS model, and define how Code and Web stay in sync.

## What was done
- Cloned the repo over SSH to `/scratch/bell/basu56/Extract_Cohesive_Law_Tpeel_Geometry`
  and wrote the four files from the script's heredocs unchanged, except:
  - CLAUDE.md: filled the cluster TBDs from TPeel_cohesive_zone Stage F records
    (Negishi, `abaqus/2026`, account `tomar`, partition `cpu`, SSH-only GitHub);
    added a pointer to the context protocol.
- Added `docs/CONTEXT_PROTOCOL.md`, `docs/STATUS.md`, `docs/EXISTING_MODEL.md`,
  `docs/sessions/` (this log and a reconstructed log of the Web session), and D6–D8 in DECISIONS.
- Reviewed `basu56/TPeel_cohesive_zone` (branch `campaign03-bridging-czm` @ `742e1ae`).
  The user referred to its folder as "TPeel_cohesive_zone".

## Results
- [D] Two model generations exist. The legacy CAE decks are superseded. The campaign03
  generator `build_tpeel_inp.py` is parametric and verified (Bell 2025 ≡ Negishi 2026).
- [D] Stage F (constant G_tip = 0.40, σ_tip 20→100 MPa): plateau and R-curve unchanged,
  peak +2 %→+13 %, ℓ_pz/H 0.68→0.16. This is prior evidence on our own geometry for H1/H3.
- [D] Bilinear admissibility σ < √(2GK) caps σ ≈ 81 MPa at K = 8500, G = 0.4.
- [H] The α-family equals ABAQUS's built-in exponential softening; needs a
  single-element check, including whether α < 0 is allowed.

## Decisions
- D6 (proposed): context-sync workflow (CONTEXT_PROTOCOL.md).
- D7 (proposed): reuse the campaign03 generator as the forward model.
- D8 (proposed): truth with ABAQUS EXPONENTIAL, inversion with a different representation.

## Not done
- No ABAQUS runs. The forward-model code has not been copied into this repo yet.

## Next steps
See `docs/STATUS.md`.
