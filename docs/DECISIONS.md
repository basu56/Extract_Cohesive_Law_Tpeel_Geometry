# Decisions log

Append-only. Each entry: date, decision, rationale, status. Changing a decision
requires a new entry that references the old one and is agreed with the user.

## 2026-11-14 — D1: Phase 1 is synthetic-only
Decision: identification uses ABAQUS-simulated data on the project's own T-peel
geometry. Experimental data and the J-integral analysis are deferred to Phase 4.
Rationale: establish identifiability under controlled, known ground truth first.
Status: agreed.

## 2026-11-14 — D2: Phase 1 observable is the global load–displacement record only
Decision: identification in Phase 1 uses P(Δ) only. Crack length, δ*, opening
profile and J are still extracted and stored for later phases.
Rationale: answers the cleanest version of the question; richer observables are
added later and compared against this baseline.
Status: agreed.

## 2026-11-14 — D3: Start with simple single-stage laws; no bridging
Decision: Phase 1 uses the one-parameter softening-shape family s(x; α)
(RESEARCH_PLAN.md §4), with α = 0 linear and α > 0 exponential-type softening.
Rationale: user preference to start simple; the family varies shape at fixed G_c
and σ_c, so "shape recovery" is a well-defined question rather than parameter
fitting within a single fixed shape.
Check performed: g(α) = 1/α − 1/(e^α − 1) agreed with numerical quadrature to
machine precision for α ∈ {−5, −2, ~0, 0, 1, 3, 7}.
Status: proposed — awaiting user confirmation.

## 2026-11-14 — D4: Project context lives in CLAUDE.md + docs/, skills later
Decision: CLAUDE.md (root, loaded every Claude Code session) holds rules and
current phase; docs/ holds the full plan and this log. Claude Code skills will be
written later for repeatable procedures (e.g. case generation, SLURM sweeps, ODB
extraction) once those scripts exist and are verified.
Status: proposed.

## 2026-11-14 — D5: Target journal CMAME
Decision: work is designed for a CMAME manuscript; emphasis on a clean problem
statement, rigorous identifiability analysis, and honest scope.
Status: agreed.

## 2026-11-14 — D6: Context-sync workflow between Claude Code and Claude Web
Decision: GitHub is the single source of truth. Session start/end procedure, the
`docs/STATUS.md` handoff page and per-session logs in `docs/sessions/` as specified in
`docs/CONTEXT_PROTOCOL.md`. Claude Web contributes through heredoc scripts using the
SSH remote; Claude Code commits and pushes directly. Extends D4.
Status: proposed (Claude Code) — awaiting user confirmation.

## 2026-11-14 — D7: Forward model = campaign03 generator, not a legacy CAE deck
Decision: generate T-peel decks with `build_tpeel_inp.py` from
`basu56/TPeel_cohesive_zone` (campaign03), extended with the α-family law, rather than
copying a legacy `input_files/*.inp` into `abaqus/templates/`.
Rationale: the generator is parametric, writes JSON metadata with the git revision, is
paired with an ODB extractor, matches the user-confirmed calibrated geometry
(H = 6.324, t = 0.4, clamp/load), and was verified identical on Abaqus 2025 and 2026.
The legacy decks use a superseded geometry and K. Open: copy the code into this repo
(self-contained) or depend on the other repo (recommended: copy, recording the source
commit).
Status: proposed (Claude Code) — awaiting user confirmation.

## 2026-11-14 — D8: Truth and inversion use different law representations
Decision: synthetic truth uses ABAQUS built-in exponential softening
(`type=DISPLACEMENT, softening=EXPONENTIAL`). Our algebra says it is identical to
s(x; α), to be confirmed by the single-element test. Truth also uses a finer mesh.
Inversion uses TABULAR or piecewise-linear representations on the working mesh.
Rationale: satisfies the inverse-crime rule (CLAUDE.md rule 4) without extra code for
the truth law. If ABAQUS rejects α < 0, truth for concave shapes uses TABULAR on the
fine mesh.
Status: proposed (Claude Code) — depends on the Phase 0 single-element test.
