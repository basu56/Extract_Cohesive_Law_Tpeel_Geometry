# S1 — bulk element verification: Q4 / Q4IM (tslid.fe) vs ABAQUS CPE4I

Units: mm, N, MPa. Plane strain, Al E = 70 000 MPa, ν = 0.33. Cantilever L = 60, H = 6.324,
W = 15. Left edge fully clamped. Mesh hx = 0.1, with y graded as in the T-peel builder (first
element 0.1 at y = 0; `graded()` from `build_tpeel_inp.py`), so both codes have identical nodes.
ABAQUS: Negishi job 45856435, abaqus/2026, decks from `scripts/fe/build_cantilever_inp.py`, raw data in
`/scratch/negishi/basu56/tslid_runs/solver_S1_cantilever/`. JAX: `scripts/fe/s1_compare.py`
(Negishi jobs 45856728 and 45857247; the files here are from 45857247, commit 87c1ce2).

| File | Content |
|---|---|
| `abaqus_reference.csv` | ABAQUS tip quantities per case/frame (the tests in `tests/fe/test_s1_cantilever.py` read this) |
| `s1_linear.csv` | Tests 2–3: tip deflection / reaction / full-field error, Q4 and Q4IM, ny = 2, 4, 8, 20, plus a distorted mesh |
| `s1_nlgeom.csv` | Test 4: Q4IM with SVK and Hencky energies vs CPE4I NLGEOM, 20 load increments up to P = 1400 N (θ = 0.10 rad) |
| `s1_timing.csv` | Test 5: wall times (Negishi node a001, AMD EPYC 7763, 4 cores) |
| `tpeel_rotation.csv` | UR3 of the T-peel load point RP1 in production run sw_s40 (σ_c = 40, α = 0) |
| `s1_bulk.png/pdf` | (a) linear error vs ny, (b) NLGEOM error vs tip rotation |

## Results [D]

**Linear (NLGEOM = NO), P = 230 N** (T-peel load at δ ≈ 0.95 mm, run sw_s40), and v = 0.67 mm for
displacement control:

| ny | Q4 tip deflection error | Q4IM tip deflection error | Q4IM reaction error (disp. control) | Q4IM vs Timoshenko |
|---|---|---|---|---|
| 2 | −19.3 % | −1.2e-8 | +1.9e-8 | |
| 4 | −11.1 % | −2.1e-8 | +6.4e-9 | |
| 8 | −2.29 % | +7.4e-9 | +6.6e-9 | |
| 20 | −0.195 % | −1.2e-8 | −6.3e-9 | −0.48 % |

- Q4IM reproduces CPE4I to ~1e-8 at every ny. Full-field max |ΔU|/max|U| ≤ 4.5e-8, which is the
  single-precision ODB output limit. **Threshold 0.05 %: passed by ~4 orders of magnitude.**
- Distorted mesh (hx = 1, ny = 4, interior nodes shifted ±0.3 mm on alternate rows): Q4IM vs CPE4I
  4.7e-6, Q4 −5.4 %. So ABAQUS's CPE4I behaves like our formulation (Taylor j₀/j scaling) at
  least for this distortion.
- Q4 locking at ny = 20 is −0.195 %, about 5× the T-peel numerical floor (0.04 %). **Q4 is not
  acceptable for S3.**
- Timoshenko reference: −0.48 % for both codes (end effects of the full clamp; not a pass/fail).

**NLGEOM, Q4IM ny = 20, θ up to 0.101 rad (bending strain ≈ 1 %):**

| quantity at θ = 0.10 | SVK | Hencky |
|---|---|---|
| v_tip | −3.6e-5 | +3.9e-5 |
| clamp reaction moment | +3.0e-5 | −7.2e-6 |
| u_tip, relative to its own value | **−1.25 %** | **+0.30 %** |
| u_tip difference / \|v_tip\| | −4.5e-4 | +1.1e-4 |

- v_tip and the clamp moment pass the 0.1 % threshold at every increment, with either energy.
- **u_tip (axial displacement of the mid-height tip node) does NOT pass 0.1 % relative to its own
  value**, for either energy. It is a small quantity: −0.145 mm vs v = 4.06 mm, and it changes
  sign near θ = 0.01 rad, where its relative error peaks at 7 %. Measured against |v_tip| it is
  ≤ 4.5e-4 (SVK) and ≤ 1.1e-4 (Hencky).
- Changing only the strain measure (SVK → Hencky) cuts the u_tip error 4× and flips its sign,
  and the absolute SVK difference grows ∝ load². Together these show that the discrepancy is
  constitutive (second order in strain), not kinematic [D]. The remainder is attributed to ABAQUS's
  hypoelastic Cauchy-stress update vs our hyperelastic Kirchhoff-stress Hencky model [H].
- **T-peel relevance:** the measured load-point rotation in sw_s40 peaks at **0.0325 rad**
  (`tpeel_rotation.csv`). At θ ≈ 0.03: v_tip ≤ 6e-6, moment ≤ 3e-6, u_tip/|v| ≤ 1.3e-4 (SVK) and
  4e-5 (Hencky). These are far below the 0.04 % P–Δ floor [I].

**Timing** (test 5; Negishi a001, 4 cores, not the laptop):

| case | n_dof | JAX first call (incl. JIT) | JAX per solve after JIT | ABAQUS wall |
|---|---|---|---|---|
| linear ny = 20, Q4 | 25 242 | 1.6 s | 1.05 s | |
| linear ny = 20, Q4IM | 73 242 | 3.1 s | 1.78 s | 6.2 s (incl. pre-processing and licence) |
| NLGEOM ny = 20, SVK, 20 incr., 60 Newton its | 73 242 | 54 s | 2.6 s / increment | 49.7 s |
| NLGEOM ny = 20, Hencky | 73 242 | 202 s | 9.8 s / increment | — |

Timings vary between nodes: the same linear Q4IM solve took 2.0 s on node a032. The Hencky series
costs about 4× SVK per tangent.
