# Existing ABAQUS T-peel model — review for this project

Reviewed 2026-11-14 by Claude Code. Source repo: `git@github.com:basu56/TPeel_cohesive_zone`
(private), clone at `/scratch/bell/basu56/TPeel_cohesive_zone`, branch
`campaign03-bridging-czm` @ `742e1ae`. Labels: **[D]** demonstrated, **[H]** hypothesis,
**[I]** interpretation.

## 1. Two generations of model exist

| | Legacy decks (`input_files/*.inp`, 12 files) | Campaign03 generator (`scripts/build_tpeel_inp.py`) |
|---|---|---|
| How produced | Exported from CAE (`geometry/*.cae`), mesh inline | Python 3.6 stdlib script; writes `.inp` + JSON metadata with git rev |
| Elements | 11 387 CPE4R adherend + 291 COH2D4, 0.2 mm cohesive elems | CPE4I adherend (default), COH2D4 bond, `h_coh` 0.1 mm default, 20 elems through H |
| Geometry | H = 6.32, L = 76.2, legs 15.4; precrack cohesive x = 18–30 | Parametric: H, t (bondline), L, W, leg, groove, notch, a0 |
| Bondline | constitutive thickness 1, t not represented | t = 0.4 mm gap between faces; K0 = E_a/t = 8500 MPa/mm |
| Law | single bilinear (MAXS + ENERGY, linear), K = 6000 | `elastic`, `bilinear`, `bridging` (2–3 superposed bilinear layers, Dávila–Rose–Camanho) |
| BCs | RP2 pinned, RP1 U2 = 5 mm | `--bc-bottom pin/clamp`, `--bc-top load/pin/clamp`; calibrated = clamp/load |
| Solver | Standard, `*Static`, nlgeom, viscosity 1e-3 | same; viscosity 1e-3 s, step time from crosshead rate (1 mm/min default) |
| Status | superseded; the baseline README's elastic-plastic claim is wrong (decks are elastic) | **current, verified** (Bell 2025 ≡ Negishi 2026 to printed digits) [D] |

Related scripts in the old repo: `scripts/extract_tpeel_odb.py` (abaqus-python; per-frame
P, Δ, several crack-length measures, energies; per-element traction/SDEG profiles →
`.npz`), `scripts/rebuild_from_json.py` (rebuild a deck from its JSON with overrides),
`scripts/run_abaqus_dir.sh` and `scripts/stageF/run_abaqus_negishi.sh` (SLURM runners),
`scripts/compare_experiment.py` (the experiment's LEFM pipeline).

**Recommendation [I]:** use the campaign03 generator as the forward model, not a CAE
legacy `.inp`. It already provides parameterization, metadata and extraction, and the
CLAUDE.md "copy the `.inp` into `abaqus/templates/`" rule is only needed for the legacy
route. See DECISIONS.md D7 (proposed).

## 2. Calibrated configuration (user-confirmed in campaign03)

- Adherends Al 6063-T6, E = 70 GPa, ν = 0.33; campaign01 concluded adherend plasticity
  does not matter for this specimen [D, campaign01 FINAL_REPORT].
- H = 6.324 mm (user's compliance FE matches this, not the nominal 6.0), W = 15 mm,
  t = 0.4 mm, notch at x = 17 mm; a0 = 17 (Stages A–D) or 18 (user prefers a 1–2 mm
  precrack, Stage E). BCs: lower leg clamped, upper leg point-loaded (clamp/load).
- Experimental record SG24 (`experiment/SG24_load_disp_ell.csv`): P, δ, compliance and
  DIC crack lengths, LEFM G. Phase 4 only — not used here before then (D1).

## 3. Prior results directly relevant to non-uniqueness

- **Stage F (2026-11-07)** [D]: G_tip fixed at 0.40 N/mm, σ_tip swept 20→100 MPa.
  ℓ_pz/H fell from 0.68 to 0.16 (≈ σ^-0.9); **plateau, crack speed and R-curve were
  independent of σ_tip**; peak load rose +2 % → +13 %. This is our own evidence for
  H1/H3 in the propagation regime, and it shows that the information about strength
  sits in the initiation/peak part of P–Δ.
- **Admissibility** [D]: a bilinear layer needs σ < √(2 G K). With K0 = 8500 and
  G = 0.4 this caps σ ≈ 81 MPa; σ = 80 snapped back (151 negative-eigenvalue warnings).
  The Phase 1 grid over ℓ_ch/H must respect this or use a stiffer K (Stage F used
  K0 = 85 000 as a second series).
- **Stages A–E** [D]: a single bilinear law cannot produce the SG24 plateau. A
  tip + tail + bridging-hump law reproduces peak and plateau within ~2 %. Several
  distinct multi-layer laws fit about equally well, which is informal evidence of
  non-uniqueness. Phase 1 excludes bridging (D3); this is a Phase 3 topic.

## 4. Gaps for Phase 0 of this project

1. **Shape family.** The generator only does linear softening. The Phase 1 family
   s(x; α) = (e^{−αx} − e^{−α})/(1 − e^{−α}) is, by algebra, the ABAQUS built-in
   **exponential softening** law (`*Damage Evolution, type=DISPLACEMENT,
   softening=EXPONENTIAL`, input δ_f − δ_0 and α) [H — confirm with the single-element
   test; also check whether ABAQUS accepts α < 0]. The alternative is
   `softening=TABULAR` (D vs δ_m − δ_0), which handles any shape. Using EXPONENTIAL for
   truth and TABULAR (or piecewise-linear) for inversion would also satisfy the
   "different representation" part of the inverse-crime rule.
2. **Numerical error floor.** No mesh / increment / viscosity convergence study exists
   for P–Δ at the level needed (only one h_coh = 0.05 check in Stage F).
3. **Cost.** ~15 min per run → a 1000-run sweep is ~250 CPU-hours; surrogate or
   job-array design matters for Phase 1C/2.
4. **Raw outputs at risk.** Stage A–E ODBs are only on Bell scratch.
