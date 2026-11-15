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

## 2026-11-14 — D3 update: α shape family confirmed
User agreed: start with the one-parameter family s(x; α); add complexity (multi-stage,
bridging) later. D3 status → **agreed**.

## 2026-11-14 — D4/D6 update: context workflow confirmed
User agreed. Claude Code maintains notes, decisions and progress in this repo;
the user has Claude Web review the repository periodically. D4 and D6 status → **agreed**.

## 2026-11-14 — D7 update: generator reused, calibration not reproduced
User agreed to build on the campaign03 generator, with a change: synthetic studies do
**not** need to reproduce the SG24 calibration or match experiments. The generator is a
starting point and may be simplified or changed. Experimental evidence is used only to
**bound the input parameters** (D9). D7 status → **agreed (modified)**.

## 2026-11-14 — D9: Parameter space bounded by experimental evidence
Decision: synthetic laws are sampled from physically grounded bounds, not chosen
arbitrarily.
- G_c ∈ [0.1, 0.4] N/mm (= kJ/m²), user-specified from existing experimental
  evidence; reference value G_c = 0.40 N/mm (approximate, not to be matched exactly).
- K = 8500 MPa/mm (E_a/t with t = 0.4 mm; E_a = 3400 MPa is an **assumed** value
  inherited from campaign03 and still needs a source).
- σ_c: bounded above by energy admissibility, σ_c < √(2 G_c K) (41–82 MPa over the
  G_c range), and further by the elastic-energy fraction limit in D10. Lower and upper
  physical bounds from adhesive strength data: **TBD — user to supply** (campaign03
  calibrated tip strengths were 15–20 MPa).
- α: range TBD after the single-element test (whether α < 0 is supported).
Implied ℓ_ch/H = E′G_c/(σ_c² H) ranges from ≈ 0.35 to ≈ 50 (E′ = 78.6 GPa, H = 6.324),
which spans the small- to large-scale process-zone regimes needed for H2.
Status: G_c range **agreed**; σ_c bounds proposed/TBD.

## 2026-11-14 — D10: Energy constraints are a guiding principle of the project
Decision (user's emphasis): every generated law, every simulation and every inferred
law must satisfy explicit, checkable energy constraints. Fracture energy is constrained
by evidence, and the energy budget is reported throughout. Concrete constraints are
listed in RESEARCH_PLAN.md §4b (law level, simulation level, inference level).
Rationale: keeps the inverse problem physically grounded, not purely mathematical;
continues the user's prior work on fracture energy budgets in adhesives.
Status: principle **agreed**; the specific checks and tolerances are proposed.

## 2026-11-14 — D11: Elastic adherends for Phase 1 (proposed)
Decision: Phase 1 uses linear-elastic adherends (`--no-plastic`).
Rationale: then the only dissipation is cohesive (ALLPD = 0), so the energy balance
closes cleanly and the dissipated energy can be checked directly against G_c × area
(D10). Campaign01 found adherend plasticity irrelevant for this specimen. Geometry
default: H = 6.324, t = 0.4, W = 15, clamp/load, a0 = 18 mm (the user's preferred
1 mm precrack).
Status: proposed — awaiting user confirmation.

## 2026-11-14 — D9 update: K confirmed
User confirmed K = 8500 MPa/mm. D9 K entry → **agreed**.

## 2026-11-14 — D10 update: tolerances accepted
User accepted the proposed tolerances: E1 1e-4 (relative), E2 φ_e ≤ 0.5, E4 1 %,
E5 1 %, E6 2 %. Status → **agreed**.

## 2026-11-14 — D11 update: elastic adherends, a0 = 18 mm
User agreed. Implemented as generator defaults (H = 6.324, a0 = 18, elastic).
Status → **agreed**.

## 2026-11-14 — D12: σ_c lower bound from the physical process-zone size
Decision (user): σ_c has no well-established lower bound from strength data. Instead,
bound it physically: lower σ_c lengthens the process zone, and a process zone much
longer than the adherend thickness H makes the problem physically not meaningful.
Lower bound: σ_c,min(G_c, α) is the strength at which the FE-measured steady-state
process-zone length reaches ℓ_pz = c·H. Proposed c = 1, to be revisited once the pilot
map exists. Note that ℓ_ch = E′G_c/σ_c² is not ℓ_pz (Stage F: ℓ_ch/H ≈ 12 gave
ℓ_pz/H ≈ 0.87), so the bound is set from FE, not from ℓ_ch.
Upper bound: E2 gives σ_c ≤ √(K G_c) (29 / 46 / 58 MPa at G_c = 0.1 / 0.25 / 0.4).
A practical limit also comes from mesh resolution (≥ 3–5 cohesive elements in ℓ_pz).
Status: principle **agreed**; c = 1 proposed; values from the pilot map
(`scripts/phase0/run_pz_map.sh`).

## 2026-11-14 — D13: Compute on Negishi only
Bell compute was stopped by Purdue. All ABAQUS and heavy Python runs on Negishi
(`--account=tomar --partition=cpu`). The repo is mirrored to a bare repo on Negishi
(`/scratch/negishi/basu56/repo/Extract_Cohesive_Law_Tpeel_Geometry.git`, remote
`negishi`), cloned at `/scratch/negishi/basu56/Extract_Cohesive_Law_Tpeel_Geometry`.
Runs go in `/scratch/negishi/basu56/tslid_runs/<study>/`. Status: **agreed**.

## 2026-11-14 — D8 update: hypothesis confirmed by the single-element test
ABAQUS EXPONENTIAL softening reproduces s(x; α) to 1e-7 σ_c, including α = −2, and
TABULAR (1000 rows) to 1.6e-5 σ_c; dissipation = G_c within 1e-5
(`results/phase0/single_element/`). Truth: EXPONENTIAL (LINEAR for α = 0) on a finer
mesh; inversion: TABULAR / piecewise-linear. D8 status → **agreed in principle,
verified**; the α range can include α < 0. Final confirmation by the user pending.

## 2026-11-14 — D14: Zone-length definitions (partly agreed)
- **ℓ_cz (agreed):** distance from the point where the traction equals σ_c (δ = δ_0) to
  the traction-free point (δ = δ_f) in the regularized field. It is measured from the
  nodal opening profile during steady propagation. This is the length the user calls
  "process zone / ℓ_c" in the manuscript figure (σ_yy along the crack line, Griffith vs
  cohesive). In this repo it is always written ℓ_cz.
- **ℓ_pz (open):** an energy-based dissipation length was proposed (onset to w(δ) = q G_c).
  The user is reviewing the literature; the definition and q are not agreed. Do not use
  ℓ_pz until settled.
- SDEG-threshold fronts (campaign03 extractor) are **not** used for any zone length (bias up to ~25 %).

## 2026-11-14 — D12 update: σ_c lower bound via ℓ_cz or ℓ_ch
User: set the lower bound from ℓ_cz or from ℓ_ch. Pilot result [D, α = 0]: ℓ_cz/H is a
single function of ℓ_ch/H (runs at different G_c with the same ℓ_ch/H give the same ℓ_cz/H
to 2 digits), but ℓ_cz ≠ ℓ_ch: ℓ_cz/ℓ_ch = 0.02–0.14. Since the map is one-to-one, bounding
ℓ_cz ≤ c H is equivalent to bounding ℓ_ch/H ≤ f⁻¹(c). With c = 1 this gives ℓ_ch/H ≈ 11.6 and
σ_c,min ≈ 10.3 / 16.4 / 20.7 MPa at G_c = 0.1 / 0.25 / 0.4 [preliminary; c not yet agreed].
Status: approach agreed; c and the α-dependence pending.

## 2026-11-14 — D12 update: pilot complete (α = 0)
All 21 pilot runs complete. ℓ_cz/H = f(ℓ_ch/H) holds to ≤ 0.4 % across G_c. With c = 1:
ℓ_ch/H ≤ 11.3, σ_c,min = 10.5 / 16.6 / 21.0 MPa at G_c = 0.1 / 0.25 / 0.4
(`results/phase0/pz_map/`). Status: c and the α-dependence still pending the user.

## 2026-11-14 — D12 update: c = 1 adopted
User: take c = 1 for now (ℓ_cz ≤ H). At G_c = 0.4, α = 0: σ_c ∈ [21.0, 58.3] MPa
(lower bound from ℓ_cz = H, upper bound from E2). Status: **agreed (provisional)**.

## 2026-11-14 — D15: One-parameter studies — vary σ_c at fixed G_c
Decision (user): change one parameter at a time. The first study keeps G_c = 0.40 N/mm
and α = 0 fixed and varies σ_c only, within the D12 bounds.
Rationale: attributes any change in P–Δ to σ_c alone. Goal: show that laws with the same
G_c but different σ_c (different cohesive laws, the same energy) give nearly the same P–Δ,
with a small change in peak load. This is the first, controlled demonstration of
non-uniqueness. It must be measured against the numerical error floor (rule 5).
Status: **agreed**.

## 2026-11-14 — D16: Production numerical settings (from the convergence study)
h_coh 0.1 mm (≥ 11 elements per ℓ_cz at σ_c = 58), viscosity 1e-4 s, n_inc 800, Δ = 1.6 mm.
Evidence: `results/phase0/convergence/`. Mesh/increment error on P ≤ 0.04 % of P_peak;
viscosity 1e-3 biases P by ≈ 0.1 % after the peak; 1e-4 vs 1e-5 differ by < 0.01 %.
Status: **adopted by Claude Code** (numerical; within the agreed tolerances D10).

## 2026-11-14 — D17: Shape (α) sweep at fixed G_c and σ_c
Decision (user, on Claude Web's recommendation): vary α only, at G_c = 0.40 N/mm, with
α ∈ {−3, −1.5, 0, 1.5, 3, 5, 7} at two fixed σ_c: 40 MPa (ℓ_cz/H ≈ 0.42 at α = 0) and
21 MPa (ℓ_cz/H ≈ 1.0 at α = 0). Production settings D16; reference α = 0 for each σ_c.
Tests H3 (shape is the least identifiable parameter) and H2 (identifiability grows with ℓ_cz/H).
All laws are admissible (φ_e = 0.235 / 0.065, independent of α). δ_f − δ0 grows with α
(σ_c 40: 0.0106 / 0.0153 / 0.0539 mm at α −3 / 0 / 7).
**Caveat:** the c = 1 bound (D12) was set at α = 0, so the σ_c = 21 MPa runs with α > 0 will
likely give ℓ_cz > H. They are run anyway, ℓ_cz is measured, and the out-of-bound runs are flagged.
How ℓ_cz (to the traction-free point) should treat the long low-traction tail of high-α laws
is an open point.
Status: **agreed** (sweep design); runner `scripts/phase1/run_alpha_sweep.sh`.

## 2026-11-14 — D18: Interpretation of the σ_c and α sweeps — peak set mainly by ℓ_cz/H
Evidence: all 21 runs at G_c = 0.40 (σ_c sweep, α = 0; α sweeps at σ_c 40 and 21).
A quadratic fit of P_peak vs ln(ℓ_cz/H) using the σ_c sweep only predicts the α runs within
≈ 1.2 % for α ∈ [−3, 3]. The residuals grow to 1.6–5 % at α = 5 and 7 [D, Claude Web
re-analysis of the summary CSVs]. An energy-weighted mean separation of the law collapses
the data worse (residuals up to 4.5 %).
Statement [I]: at fixed G_c, P_peak ≈ f(ℓ_cz/H), with a second-order shape correction that
grows with tail length. This is why matched-peak (σ_c, α) pairs exist. The process-zone length,
not the initial softening slope, is the better descriptor: higher α steepens initial softening
yet lowers P_peak.
Correction: Claude Web had predicted a shape effect on P_peak smaller than the σ_c effect;
it is comparable (8.8 % / 13.6 % vs 12.2 %).
Status: **proposed** (to be tested with a dedicated check, e.g. matched-ℓ_cz pairs).

## 2026-11-14 — D19: Differentiable forward solver on tatva
Decision (user): build a differentiable T-peel solver as our own layer (`src/tslid/fe/`) on
**tatva 0.11.6, pinned** (JAX, LGPL-3.0, used as a dependency, not modified). Do not contact
the tatva authors for now; the cohesive/T-peel layer may become a short contribution.
ABAQUS stays the reference and truth generator. The development loop is local (laptop,
Claude Code); production runs are on the cluster.
The verification ladder is S0 environment → S1 bulk element (CPE4I-equivalent, NLGEOM) →
S2 cohesive interface (14 single-element runs) → S3 T-peel vs ABAQUS → S4 throughput →
S5 adjoint gradients. Each step stops for user review.
Speed is a **hypothesis**. It is measured as throughput (runs/h vs ABAQUS ≈ 30/h,
licence-limited) and time per gradient, not assumed. The ABAQUS cost is attributed to mesh
size and output, not to ℓ_cz (STATUS runtime note).
Then: reproduce Hao, Han & Bahmani (arXiv 2610.33711) as the benchmark.
Instructions: `docs/solver/S0_S2_INSTRUCTIONS.md`.
Status: **agreed**.

## 2026-11-14 — D20: Solver log of gains
Every solver step writes a fixed-format entry in `docs/SOLVER_LOG.md` (built, verification,
runtime, gain, limitations, open items) and updates its summary table.
Status: **agreed**.

## 2026-11-14 — D21: Bulk element and energy for the T-peel adherends in tslid.fe
Evidence: S1 (`results/solver/S1_bulk/README.md`). Q4IM reproduces ABAQUS CPE4I to ≤ 2e-8 (linear).
Plain Q4 is 0.195 % too stiff at the production mesh. Under NLGEOM, Hencky is 4× closer to CPE4I
than SVK in the axial tip displacement and the clamp moment, at 4× the cost.
Proposal: use Q4IM with the Hencky energy in S3, and keep SVK as a cross-check. Judge NLGEOM agreement
by v_tip, the reaction moment and |Δu|/|v|, not by u_tip relative to its own value.
Status: **proposed** (needs the user's agreement; it also closes the S1 u_tip flag).
