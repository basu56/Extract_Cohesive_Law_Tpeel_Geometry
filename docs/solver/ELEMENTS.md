# Bulk elements in `tslid.fe` (S1)

Units: mm, N, MPa. Code: `src/tslid/fe/elements.py`, `kinematics.py`. Tests: `tests/fe/test_s1_*.py`.
Results: `results/solver/S1_bulk/`.

## What we implement

Both elements use 4-node bilinear geometry and displacement interpolation (tatva `element.Quad4`)
and **2 × 2 Gauss integration**, in plane strain.

**(a) Q4** — plain bilinear quad. Baseline; it shear-locks in bending.

**(b) Q4IM** — Q4 enhanced with **4 incompatible modes per element** (Wilson, Taylor & Doherty 1973;
Taylor, Beresford & Wilson 1976):

- Modes: φ₁ = 1 − ξ², φ₂ = 1 − η², one amplitude per mode and displacement component
  (a ∈ ℝ^{2×2}, 4 unknowns per element).
- **Taylor's modification**: mode gradients are mapped with the element-centre Jacobian J₀ and
  scaled by j₀/j(ξ):

  H_enh(ξ) = (j₀ / j(ξ)) · Σ_k a_k ⊗ (J₀⁻¹ ∂φ_k/∂ξ)        (tatva convention J_ij = ∂X_j/∂ξ_i)

  Then ∫_e H_enh dA = j₀ Σ_q w_q (…) = 0 because Σ_q w_q ξ_q = Σ_q w_q η_q = 0. So the enhancement
  carries no average strain, and the patch test is passed on distorted elements ([D], test 1).
- **Additive enhancement of the displacement gradient**: H = ∇u_c + H_enh. For finite rotation,
  F = I + H = F_c + H_enh. This is the enhanced deformation gradient of Simo & Armero (1992,
  IJNME 33:1413), with the enhancement mapped by J₀.
- **No static condensation.** The amplitudes are extra global unknowns placed after the nodal dofs.
  The total energy is minimised over (u, a) jointly. Each amplitude couples only to its own
  element (12 × 12 element blocks), so the AD tangent stays sparse (`tatva.sparse.jacfwd`,
  34 colours vs 18 for Q4).

**Kinematics** (`kinematics.py`): small-strain ψ = λ/2 (tr ε)² + μ ε:ε. Finite-rotation
St. Venant–Kirchhoff ψ = λ/2 (tr E)² + μ E:E, with E = ½(H + Hᵀ + HᵀH). Plane strain:
F₃₃ = 1, E_i3 = 0, and λ, μ are the 3D Lamé constants.

## Relation to ABAQUS CPE4I

> **Citation status: from memory, NOT verified against the manual text.** The ABAQUS documentation
> is not installed on Bell or Negishi (searched 2026-11-14), and help.3ds.com needs a login.
> **User: please confirm the sections below.** Until then, the numerical comparison (S1 tests 3–4)
> is the evidence of equivalence on our meshes, not this text.

- *Abaqus Theory Guide*, Elements → Continuum elements → "Continuum elements with incompatible
  modes" (numbered §3.2.4 in 6.x-era manuals). [unverified recollection] It describes first-order
  elements whose **deformation gradient** (not the displacement) is enhanced by incompatible modes,
  in the spirit of Simo & Armero (1992), with the enhancement built from the Jacobian at the
  element centre so that the patch test is passed. The internal dofs are condensed at element level.
- *Abaqus Analysis User's Guide*, "Solid (continuum) elements" / "Choosing a solid element".
  [unverified recollection] CPE4I has 4 incompatible-mode dofs in 2D. Its accuracy degrades for
  parallelogram/trapezoidal distortion (ABAQUS lists trapezoidal shapes as the problem case).

**What is the same [I]:** the element family (bilinear Q4 + incompatible modes, enhancement of the
deformation gradient, centre-Jacobian mapping, 2 × 2 integration, 4 extra dofs in 2D). Condensed
or joint minimisation gives the same discrete solution, because both minimise the same energy over
the same space.

**What may differ [H]:**
1. Details of the mode basis and scaling on **distorted** elements (e.g. whether ABAQUS uses the
   j₀/j scaling or an orthogonalised variant). On rectangles and parallelograms j = j₀ everywhere and
   J = J₀, so all common variants coincide. Our cantilever and the T-peel arms are rectangular,
   so this cannot matter there. The distorted mesh `skew_ny4_F` is compared for information only.
2. **Constitutive update under NLGEOM.** ABAQUS integrates a hypoelastic (Jaumann-rate type) update
   for `*Elastic` at finite strain. We use SVK. They agree to first order in strain; test 4 measures
   the difference.

## Measured equivalence and element choice for S3 [D]

From S1 (`results/solver/S1_bulk/README.md`):
- Q4IM = CPE4I to ≤ 2e-8 on the production-graded cantilever (ny = 2 … 20), and to 4.7e-6 on a
  skewed mesh.
- Under NLGEOM, the tip deflection and the moment agree to ≤ 4e-5 up to 0.1 rad. The axial tip
  displacement differs by 1.25 % (SVK) or 0.30 % (Hencky) of its own small value. That difference is
  constitutive (hyperelastic vs hypoelastic) and second order in strain.
- Proposed (D21): Q4IM + Hencky for S3. Option (c), quadratic elements, is not needed.
