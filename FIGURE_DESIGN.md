# Figure design notes: IJF manuscript

How the manuscript figures are designed, and why. Written after the redesign
of Figure 3 (November 2026), so later figures can follow the same rules and
skip the dead ends.

The **specification** sections below have the exact values: §3 for Figure 3,
§4 for Figure 1, §5 for Figure 2. All three are implemented in
`dcb_three_laws_TPeel_basu_etal_2.m`, and they are the template for Figures 4–6.

---

## 1. Principles

A fracture-mechanics figure makes one argument. Every element either supports
that argument or comes out.

1. **State the message before plotting.** Figure 3 makes one claim: with
   G∞ = G_c(ℓ), LEFM reproduces the load, the crack length and the crack
   speed. With G∞ = G_c or G∞ = G_c(ℓ̇) it does not. Every choice below
   serves that sentence.
2. **Markers are data, lines are theory.** Measurements are hollow markers on
   a thin line. Models are plain lines with no markers. The reader can tell
   measured from computed before reading a single label.
3. **Colour means a model.** Measurements are black, or a single neutral
   accent (the blue DIC tip). Each fracture law has one colour and one line
   style, the same in every panel and every inset.
4. **Weight follows importance.** The law that works (G_c(ℓ)) is the
   heaviest line. The laws that fail are thinner and broken (dashed, dotted).
   Context such as the elastic branch is thin grey.
5. **Name curves where they are read.** Put labels on the curves where the
   comparison happens. Use a small legend only where symbols must be told
   apart and labels cannot sit next to the data.
6. **Readers do not see the fitting machinery.** Leave out "end of fit",
   notch-tip lines, fitted-window shading, n = 0.50 demo values and the like.
   They belong in the text or the caption, if anywhere.
7. **Show measured values as measured.** Do not smooth beyond the operator
   that defines the quantity. If part of a record is left out, cut it at a
   stated, physical point and say so in the caption. Never reshape it.
8. **Design for the printed size.** Check font sizes at the size the journal
   prints the figure, not on screen (see §3.6).

---

## 2. The Figure 3 argument

| Panel | Shows | What the reader should see |
|---|---|---|
| (a) | P/W against δ | Only G∞ = G_c(ℓ) follows the post-peak record. G∞ = G_c and G∞ = G_c(ℓ̇) drop far below it. |
| (b) | ℓ/L against δ | The two measured crack lengths (compliance, DIC) lie on G∞ = G_c(ℓ). The other two laws over-predict crack growth. |
| (b) inset | ℓ̇ against δ | The measured crack slows down. Only G∞ = G_c(ℓ) slows with it. The rate law predicts the fastest crack. |

Load (a), length (b) and speed (inset) are three independent views of the
same failure. That repetition is the strength of the figure.

Load misfit for the caption (rms over the fitted span, from `_2.m`):

| Law | rms load misfit |
|---|---|
| G∞ = G_c | 27% |
| G∞ = G_c(ℓ), m = +0.63 | 7% |
| G∞ = G_c(ℓ̇), every n in 0.1–1 | 28–30% |

---

## 3. Specification (Figure 3, final)

### 3.1 Canvas
- Figure 29 × 11.8 cm, `tiledlayout(1,2,'TileSpacing','normal','Padding','normal')`.
- Export: PNG at 220 dpi, and PDF with `-painters` so the vector output stays editable.
- Interpreter is LaTeX everywhere. Axes line width 1.0, ticks in, box on, `Layer` top.

### 3.2 Colours and styles (shared by all panels and the inset)

| Series | Colour | Style | Width (main) |
|---|---|---|---|
| G∞ = G_c(ℓ) | `#4C1F7A` purple | solid | 2.8 |
| G∞ = G_c | `#1F7D53` green | dashed `--` | 1.8 |
| G∞ = G_c(ℓ̇) | `#E9762B` orange | dotted `:` | 2.1 |
| Experiment | black | thin line 0.6 + hollow circles | — |
| DIC crack tip | face `#2166AB`, edge `#0D3366` | filled `v`, no line | — |
| Elastic branch (before initiation) | grey 0.55 (`#8C8C8C`) | solid | 1.2 |

Draw order, bottom to top: elastic branch, the experiment line and markers in
(a), then G_c, G_c(ℓ̇), G_c(ℓ). In (b), DIC is drawn before the experiment so
the circles sit on top.

### 3.3 Markers
- Experiment: `'o'`, size 5, white face, black edge 0.6. Positions are spaced
  evenly along the normalised arc length of the curve (`marker_points`), so
  they do not bunch at the steep loading branch. Count: 70 in (a), 26 in (b).
- DIC: `'v'`, size 4.6, edge 0.5, 22 markers.
- Fewer markers in (b) than in (a): two measured series overlap there, and
  dense markers turned the G_c(ℓ) line into clutter.

### 3.4 Panel (a): load against opening
- x: [0, 1.05·max δ of the fitted span]. y: [0, 45] N/mm, ticks 0:10:40.
- Laws are drawn from initiation (δ_init) onward. The shared LEFM loading line
  is drawn once, in grey, from the origin to (δ_init, P_peak/W).
- Curve labels, no legend:
  - G∞ = G_c(ℓ) at (0.60, 33.6), 15 pt, purple;
  - G∞ = G_c at (1.18, 19.8), 14 pt, green;
  - G∞ = G_c(ℓ̇) at (0.50, 10.6), 14 pt, orange;
  - "Experiment" at (0.29, 41.0), 14 pt, black.
- The label text writes the energy balance in full (`G^{\infty} = G_c(\ell)`),
  not just "G_c(ℓ)". It tells the reader what each reconstruction assumes.

### 3.5 Panel (b): crack length against opening
- y: ℓ/L over the whole arm, [0, 1], ticks 0:0.2:1. The full physical range
  reads naturally (0 = load line, 1 = free end), and it leaves the lower
  right empty for the inset.
- The elastic branch is a flat grey line at ℓ_i/L up to δ_init.
- Experiment (compliance crack length): thin black line plus white circles,
  the same symbol as in (a). No uncertainty band. The 1σ band is about
  0.15 mm, invisible at this scale, and a band broke the "thin line and
  circles = experiment" convention.
- Legend: MATLAB `legend`, two entries only ("Experiment", "DIC"),
  northwest, 13 pt, `Box off`, `ItemTokenSize [24 18]`. The laws need no
  entry, because colour and style carry over from (a).

### 3.6 Inset in (b): crack speed
- Position, as fractions of the (b) axes: [x0 0.47, y0 0.165, w 0.49, h 0.36].
- x: [0.2, 1.8] mm, ticks 0.5:0.5:1.5. The width is spent on the range where
  the crack actually runs.
- y in m/s, ticks 0:2e-4:8e-4 with `ax.YAxis.Exponent = -4`. The ticks read
  0–8 and MATLAB places "×10⁻⁴" at the top of the axis. The upper limit is
  1.05 × the fastest law speed, and the axis starts at 0.
- Fonts: ticks 13 pt, axis labels 14 pt. At IJF full width (~17.4 cm, scale
  ≈ 0.6) that prints at about 7.8 pt. At 11 pt ticks it would print at about
  6.6 pt, which is at the legibility floor.
- Lines at 0.85 × the main-panel widths. Experiment markers 3.6 pt, 14 of
  them, on a thin black line.
- **Measured speed** is the derivative of the same 10 s cubic Savitzky–Golay
  fit that defines ℓ. It is shown only from the peak to its minimum:
  0.18 mm/s at δ = 1.40 mm, ℓ/L = 0.67. Values are plotted as measured; only
  the later part is left out.
  - Caption wording: "shown up to the minimum crack speed". Do not call the
    later rise a numerical artefact. The analysis block in `_2.m`
    (v = opening term + load term) attributes it to the measured load
    collapse, through the −Ṗ term.

### 3.7 Panel letters
- "(a)" and "(b)" sit to the **left** of each axes, level with the top of the
  box: `text(-0.20, 1.0, ..., 'Units','normalized', 'VerticalAlignment','top')`,
  16 pt. They do not compete with titles, legends or the "×10⁻⁴" multiplier.

---

## 4. Specification (Figure 1, final)

**Message.** (a) Classical beam theory does not capture the T-peel compliance:
Gross–Srawley is 28–34% above the FE compliance, and Euler–Bernoulli is 29%
below it at the notch. So the FE calibration is used; its fit has an rms
error of 0.38%. (b) In linear elasticity, G∞ from that calibration equals J.
The domain-integral J is 0.984–1.040 × G∞, and the P–θ J is 0.967–1.006 × G∞.

**Colour.** Figure 1 is achromatic. Green, purple and orange belong to the
three fracture laws (§3.2), and Figure 1 contains none of them.

| Series | Style |
|---|---|
| FE calibration (fitG, gauge points) | solid, near-black `0.12·[1 1 1]`, width 2.4 |
| FE runs | white circles, size 5.5, black edge 0.7, no line |
| Euler–Bernoulli | dotted, grey `0.45·[1 1 1]`, width 2.0 |
| Gross and Srawley (1966) | dashed, grey `0.45·[1 1 1]`, width 1.7 |
| G∞, Irwin–Kies (fitR slope) | solid, near-black, width 2.4 |
| J, domain integral | white squares, size 7, edge 0.7 |
| J, P–θ | grey triangles `0.35·[1 1 1]`, size 4.5, edge 0.5 |

**Canvas.** 29 × 11.8 cm, `tiledlayout(1,2)` with normal spacing and padding,
as in Figure 3. Panel letters are placed by `panel_letter`.

**(a) C against ℓ/L.**
- y is logarithmic, [1e-4, 2e-2] mm/N, ticks at the decades. x is [0.2, 0.9].
- Legend, northwest, 13 pt, no box, three entries:
  - **FE calibration**: the heavy line with a thin-edged circle. The caption
    explains that the circles are the compliance at discrete crack lengths.
  - **Euler–Bernoulli**.
  - **Gross and Srawley (1966)**.
- No curve labels.
- Implementation note. MATLAB ties a legend marker's edge width to the line
  width, so the circle is thinned through the legacy `[lg, icons] = legend(...)`
  output. That output raises "Error updating Legend" warnings inside a
  `tiledlayout`, and also with `NaN` proxies. So the legend sits on an
  invisible plain axes laid exactly over (a), with proxies drawn off-axis.
- Inset: C/C_FE (each beam solution divided by the calibration) against ℓ/L.
  - Position [0.50, 0.15, 0.46, 0.36] of (a).
  - y [0.6, 1.5], ticks 0.7, 1, 1.3; x ticks 0.3, 0.6, 0.9.
  - Fonts: ticks 13 pt, labels 14 pt.
  - The calibration is the line at 1.

**(b) G∞ and J against ℓ/L.**
- y is logarithmic, [0.08, 8] kJ/m², ticks 0.1, 0.2, 0.5, 1, 2, 5.
- Legend southwest, 13 pt, no box: G∞, Irwin–Kies; J, domain integral; J, P–θ.
- Inset: J/G∞ against ℓ/L.
  - Position [0.52, 0.58, 0.44, 0.36] of (b), in the empty top right above
    the falling curve.
  - y [0.9, 1.1] with a light ±5% band (grey `0.92`) and the line at 1.
  - Markers as in the main panel, smaller (4.5 and 4).
  - Fonts: ticks 13 pt, labels 14 pt.
- The inset carries the quantitative validation. On the log axis of the main
  panel the three estimates sit exactly on top of each other.

**Rejected for Figure 1.**
- The Figure 3 palette: it would have clashed in meaning with the laws.
- Curve labels for the beam solutions: the user preferred them named in full
  in the legend.
- A J/G∞ ratio as the whole of (b) (option F3): the user kept the G∞
  magnitude in view.
- Slate and umber colours for the beam solutions (option F4).

---

## 5. Specification (Figure 2, final)

**Message.** For this geometry the energy release rate from LEFM and from the
DIC displacement field agree throughout the test. G∞ is the Irwin–Kies value,
which needs the crack length. J comes from the DIC field and needs none.
During propagation J/G∞ has a median of 0.968, with 80% of frames between
0.946 and 1.055. So LEFM describes the energy flux to the crack tip.

**Two framing decisions.**
- **(a) is a driving-force map, not a spatial field.** Its axes are crack
  length and applied opening, (ℓ/L, δ). Each point is one specimen state, and
  its colour is the single number G∞ = P²C′(ℓ)/(2W) for that state. G is the
  total energy flux into the tip, so it has no distribution around the tip.
- **The energy release rate is not G_c.** The figure makes this distinction
  through its design:
  - the map prints its formula, which contains no material property;
  - nothing in the figure is named G_c, toughness or resistance;
  - the measured path shows only which states the specimen passed through.

**Canvas.** 30 × 12.8 cm, `tiledlayout(1,2)` with normal spacing and padding,
and `OuterPosition [0 0 1 0.95]` to leave room for the colour bar above (a).

**Numbered states**, drawn as white numerals on black discs by `state_marks`
(13 pt discs in the main axes, 9 pt in the inset). The same four states
appear in the map, in (b) and in the inset:

1. initiation;
2. plateau onset;
3. peak G∞;
4. end of the invertible span.

**(a) Driving-force map.**
- Filled contours of log₁₀ G∞, 40 levels over [0.05, 8] kJ/m².
- Sequential blue map (`seq_map`): anchors from `[0.97 0.98 1]` through
  `[0.03 0.19 0.42]`, light for low values and dark for high, with no false
  midpoint.
- Colour bar north-outside: ticks 0.1, 0.2, 0.5, 1, 2, 5; label "G∞ (kJ/m²)".
- Iso-G lines at the same values, dark grey 0.6 pt, labelled 11 pt.
- Measured path: black line 1.4 pt with 16 white circles (size 5.5).
- Formula box, bottom right, 15 pt, on white.

**(b) The three estimates along ℓ/L.** Colour shows the method, and line
style shows the crack-length source.

| Series | Style |
|---|---|
| G∞, Irwin–Kies, ℓ from C | black line 0.8 pt + white circles (22, size 5.5): the experiment convention |
| G∞, Irwin–Kies, ℓ from DIC | grey `0.40` dashed 1.3 pt + hollow squares (14, size 5.5) |
| J, P–θ (from the DIC field) | DIC blue `[0.13 0.40 0.67]` line 1.6 pt + filled triangles (16, size 5) |
| pointwise (unsmoothed) G∞, J | light tints, 0.6 pt, underneath |
| loading at ℓ_i | thin black vertical line up to G0 |

- y is [0, 1.4] kJ/m² with ticks every 0.2; x starts at the notch, ℓ0/L.
- Legend northwest, 12 pt, no box.
- Inset: G∞ (black, with the loading branch) and J (blue) against δ.
  - Position [0.55, 0.13, 0.42, 0.285] of (b).
  - x [0, 2.1] mm, y [0, 1.3]; fonts 13/14 pt.
  - The numbered states are marked here too.
- Caption should note two things:
  - The grey dashed estimate runs up to about 10% higher after ③ because the
    DIC crack tip is defined by an opening threshold. It is a different crack
    length, not a different energy.
  - The J/G∞ statistics above.

**Rejected for Figure 2.**
- The diverging red–blue map: it implies a meaningful midpoint.
- Separate panels against ℓ and against δ: they repeat the same curves.
- A J/G∞ ratio inset (option V2): the user preferred G∞ and J against δ.
- Topographic and grey maps (V4, V5).
- The vertical stack sharing ℓ/L (V3).

---

## 6. What was tried and rejected (Figure 3)

| Option | Why it was rejected |
|---|---|
| Every series as a heavy line with markers (original) | All series had equal weight, the large markers crowded the panel, and nothing pointed the reader to the result. |
| A legend box in (a) | It hid the plateau, the key evidence, and made the reader look back and forth. |
| Failing laws in grey, only G_c(ℓ) in colour | Strong pop-out, but it broke the palette shared with the other panels and figures. |
| Rate law moved to an inset of (a) | It hid the plateau. The two-panel crack-length view made the same point better. |
| Residual strip under (a) | Informative, but the final load collapse dominated it. |
| Inset of misfit against n; bar chart of misfits | Correct, but they read as a second plot or as decoration. The numbers belong in the caption. |
| Showing the "end of fit" line, notch-tip line, fitted-span shading | Fitting diagnostics, not physics. |
| Uncertainty band on ℓ(C) | Invisible at this scale, and it broke the experiment convention. |
| Shared legend row above or below both panels | Pulled attention away from the data and added height. |
| One legend for the whole figure inside (a) | Listed DIC in a panel where it is not drawn. |
| Direct labels in (b) | No free space next to the overlapping measured curves, so the labels ended up detached. |
| Crack speed as a third panel (c) | Readable, but the user preferred two panels with the inset. Keep it in mind if the inset gets too dense. |
| Speed axis in 10⁻⁴ m/s, mm/s, or m/s with four-decimal ticks | Replaced by m/s with MATLAB's exponent notation (ticks 0–8, ×10⁻⁴), matching the convention used elsewhere. |

---

## 7. Checklist for the next figure

1. Write the one-sentence message. List which series support it and which
   are context.
2. Data → hollow black markers on a thin black line. Models → plain coloured
   lines. Reuse the palette in §3.2. A law that appears in several figures
   keeps its colour and line style everywhere.
3. Give the result the heaviest line. Make the alternatives thinner, broken
   lines, and the context grey.
4. Remove fitting landmarks, demo parameter values and diagnostic lines.
5. Label curves on the plot. Add a short legend only for symbols that cannot
   be labelled in place.
6. Choose axis limits that show the physical range (ℓ/L from 0 to 1, speeds
   from 0). Use tick labels in plain units, or MATLAB's exponent notation
   rather than long decimals.
7. Panel letters go to the left of the axes, at the top. No titles.
8. Insets: put them in genuinely empty space, give them at least 13 pt
   ticks, scale their lines to about 0.85× the main panel, and keep the
   palette.
9. Check the size the figure will print at. The smallest text should print
   at 7 pt or more.
10. Explore alternatives in a scratch script first, and compare them side by
    side before changing the manuscript script.

---

## 8. Workflow used

- Variants were drawn by a standalone scratch script that loads the Figure 3
  quantities, so the manuscript script was never edited during exploration.
  The quantities came from a copy of `_2.m` that stops after the Figure 3
  identification and saves to a `.mat` file.
- All candidates were compared side by side on one HTML page.
- The scratch scripts lived in a temporary session folder. This file is the
  durable record of the outcome.
