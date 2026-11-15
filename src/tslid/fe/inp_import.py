"""Read the flat ABAQUS decks written by ``src/tslid/build_tpeel_inp.py``.  Step S3a.

Units: mm, N, MPa (as in the deck).  Dependency-free apart from NumPy: no meshio, no ABAQUS.

The parser handles exactly the keyword subset the builder writes (flat model, no *Part):
*Node, *Element (CPE4I / CPE4 / COH2D4), *Nset, *Elset (names of other elsets), *Surface
(type=NODE, one nset), *Coupling + *Kinematic, *Solid Section, *Cohesive Section,
*Section Controls (viscosity), *Material with *Elastic / *Elastic, type=TRACTION /
*Damage Initiation (QUADS) / *Damage Evolution (DISPLACEMENT: LINEAR, EXPONENTIAL, TABULAR),
*Boundary (outside and inside the step), *Step (nlgeom) and *Static.  Anything else is kept
as raw text in ``Deck.unparsed`` so nothing is dropped silently.

Node indices: ABAQUS labels are mapped to 0-based indices in label order (``Deck.index``).
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np


def _kw(line):
    """'*Element, type=CPE4I, elset=ARM_U' -> ('element', {'type': 'CPE4I', 'elset': 'ARM_U'})."""
    parts = [p.strip() for p in line[1:].split(",")]
    opts = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()
        elif p:
            opts[p.lower()] = True
    return parts[0].lower(), opts


def _nums(line):
    return [float(v) for v in line.replace(",", " ").split()]


def _tokens(line):
    return [v.strip() for v in line.split(",") if v.strip()]


@dataclass
class Deck:
    path: str
    labels: np.ndarray                     # (n_nodes,) ABAQUS node labels, sorted
    coords: np.ndarray                     # (n_nodes, 2) mm
    elements: dict                         # elset -> dict(type, labels (n,), conn (n, 4) node *indices*)
    nsets: dict                            # name -> node indices
    elsets: dict                           # name -> list of member elset names (builder style)
    surfaces: dict                         # name -> nset name
    couplings: list                        # dict(name, ref (index), slaves (indices), kind)
    sections: dict                         # 'solid' / 'cohesive' -> list of dicts
    materials: dict                        # name -> dict of parsed cards
    controls: dict                         # section-controls name -> dict (viscosity, ...)
    boundary_initial: list                 # (node indices, first dof, last dof, value) outside the step
    boundary_step: list                    # same, inside the step
    step: dict                             # nlgeom, static (dt0, T, dtmin, dtmax)
    unparsed: list = field(default_factory=list)

    @property
    def index(self):
        return {int(l): i for i, l in enumerate(self.labels)}

    # ---- convenience views for the T-peel ----
    def bulk_elements(self, elsets=None):
        """(conn (n, 4) node indices, elset name per element) for all continuum elements."""
        conns, names = [], []
        for es, d in self.elements.items():
            if d["type"].upper().startswith("CPE4") and (elsets is None or es in elsets):
                conns.append(d["conn"])
                names += [es] * len(d["conn"])
        return np.concatenate(conns), np.array(names)

    def cohesive_chain(self, elset=None):
        """Matched node pairs of a single row of COH2D4 elements.

        COH2D4 connectivity is [b_k, b_k+1, t_k+1, t_k] (face 1 = lower arm, face 2 = upper arm).
        Returns (bot, top) node indices, each (n_seg + 1,), ordered along the row.
        Raises if the row is not a single connected chain.
        """
        cands = [es for es, d in self.elements.items() if d["type"].upper() == "COH2D4"]
        es = elset or (cands[0] if len(cands) == 1 else None)
        if es is None:
            raise ValueError("several cohesive elsets %s: pass elset=" % cands)
        c = self.elements[es]["conn"]
        order = np.argsort(self.coords[c[:, 0], 0])
        c = c[order]
        if not (np.all(c[1:, 0] == c[:-1, 1]) and np.all(c[1:, 3] == c[:-1, 2])):
            raise ValueError("cohesive elements of %s do not form a single chain" % es)
        bot = np.r_[c[:, 0], c[-1, 1]]
        top = np.r_[c[:, 3], c[-1, 2]]
        return bot, top

    def solid_section(self):
        (s,) = self.sections["solid"]
        return s

    def cohesive_section(self, elset=None):
        cs = self.sections["cohesive"]
        return cs[0] if elset is None else next(s for s in cs if s["elset"] == elset)

    def expand_elset(self, name):
        if name in self.elements:
            return [name]
        out = []
        for m in self.elsets.get(name, []):
            out += self.expand_elset(m)
        return out


def read_inp(path):
    with open(path) as fh:
        raw = [l.rstrip("\n") for l in fh]
    lines = [l for l in raw if l.strip() and not l.startswith("**")]

    node_rows = []
    elements, nsets_lab, elsets, surfaces = {}, {}, {}, {}
    couplings, solid, cohesive = [], [], []
    materials, controls = {}, {}
    bnd0, bnd1, step = [], [], {}
    unparsed = []

    i, n = 0, len(lines)
    in_step, cur_mat = False, None

    def block(j):
        out = []
        while j < n and not lines[j].startswith("*"):
            out.append(lines[j])
            j += 1
        return out, j

    while i < n:
        line = lines[i]
        if not line.startswith("*"):
            unparsed.append(line)
            i += 1
            continue
        key, opts = _kw(line)
        data, j = block(i + 1)
        if key == "node":
            node_rows += [_nums(l) for l in data]
        elif key == "element":
            es = opts["elset"]
            rows = np.array([[int(v) for v in _tokens(l)] for l in data], dtype=np.int64)
            elements[es] = dict(type=opts["type"].upper(), labels=rows[:, 0], conn_labels=rows[:, 1:])
        elif key == "nset":
            nsets_lab[opts["nset"]] = [int(v) for l in data for v in _tokens(l)]
        elif key == "elset":
            elsets[opts["elset"]] = [v for l in data for v in _tokens(l)]
        elif key == "surface":
            tok = _tokens(data[0])
            surfaces[opts["name"]] = tok[0]
        elif key == "coupling":
            nxt = lines[j] if j < n else ""
            kind = _kw(nxt)[0] if nxt.startswith("*") else None
            couplings.append(dict(name=opts.get("constraint name"), ref=opts["ref node"],
                                  surface=opts["surface"], kind=kind))
            if kind == "kinematic":
                _, j = block(j + 1)
        elif key == "solid section":
            v = _nums(data[0]) if data else [1.0]
            solid.append(dict(elset=opts["elset"], material=opts["material"], thickness=v[0]))
        elif key == "cohesive section":
            v = _nums(data[0])
            cohesive.append(dict(elset=opts["elset"], material=opts["material"],
                                 response=opts.get("response"), thickness_mode=opts.get("thickness"),
                                 controls=opts.get("controls"), constitutive_thickness=v[0],
                                 width=v[1] if len(v) > 1 else 1.0))
        elif key == "section controls":
            controls[opts["name"]] = {k: (float(v) if k == "viscosity" else v) for k, v in opts.items() if k != "name"}
        elif key == "material":
            cur_mat = opts["name"]
            materials[cur_mat] = {}
        elif key == "elastic":
            materials[cur_mat]["elastic"] = dict(type=str(opts.get("type", "ISOTROPIC")).upper(),
                                                 values=_nums(data[0]))
        elif key == "plastic":
            materials[cur_mat]["plastic"] = [_nums(l) for l in data]
        elif key == "damage initiation":
            materials[cur_mat]["damage_initiation"] = dict(criterion=opts["criterion"].upper(), values=_nums(data[0]))
        elif key == "damage evolution":
            materials[cur_mat]["damage_evolution"] = dict(
                type=opts["type"].upper(), softening=str(opts.get("softening", "LINEAR")).upper(),
                rows=[_nums(l) for l in data])
        elif key == "boundary":
            target = bnd1 if in_step else bnd0
            for l in data:
                tok = _tokens(l)
                if tok[1].upper() == "ENCASTRE":
                    target.append((tok[0], 1, 6, 0.0))
                else:
                    a = int(tok[1])
                    b = int(tok[2]) if len(tok) > 2 else a
                    v = float(tok[3]) if len(tok) > 3 else 0.0
                    target.append((tok[0], a, b, v))
        elif key == "step":
            in_step = True
            step["nlgeom"] = str(opts.get("nlgeom", "NO")).upper() == "YES"
            step["name"] = opts.get("name")
        elif key == "static":
            v = _nums(data[0])
            step["static"] = dict(dt0=v[0], T=v[1], dtmin=v[2] if len(v) > 2 else None,
                                  dtmax=v[3] if len(v) > 3 else None)
        elif key == "end step":
            in_step = False
        elif key in ("heading", "preprint", "kinematic", "output", "node output", "element output",
                     "energy output", "node print"):
            pass
        else:
            unparsed.append(line)
        i = j

    nr = np.array(node_rows)
    order = np.argsort(nr[:, 0])
    labels = nr[order, 0].astype(np.int64)
    coords = nr[order, 1:3]
    if len(np.unique(labels)) != len(labels):
        raise ValueError("duplicate node labels")
    idx = {int(l): k for k, l in enumerate(labels)}

    def to_idx(labs):
        return np.array([idx[int(l)] for l in labs], dtype=np.int64)

    for d in elements.values():
        d["conn"] = np.vectorize(lambda l: idx[int(l)])(d["conn_labels"]).astype(np.int64)
    nsets = {k: to_idx(v) for k, v in nsets_lab.items()}

    def node_target(name):
        return nsets[name] if name in nsets else to_idx([int(name)])

    for c in couplings:
        c["ref"] = int(node_target(c["ref"])[0])
        c["slaves"] = nsets[surfaces[c["surface"]]]
    bnd0 = [(node_target(nm), a, b, v) for nm, a, b, v in bnd0]
    bnd1 = [(node_target(nm), a, b, v) for nm, a, b, v in bnd1]

    deck = Deck(path=path, labels=labels, coords=coords, elements=elements, nsets=nsets, elsets=elsets,
                surfaces=surfaces, couplings=couplings, sections=dict(solid=solid, cohesive=cohesive),
                materials=materials, controls=controls, boundary_initial=bnd0, boundary_step=bnd1,
                step=step, unparsed=unparsed)
    check_orientation(deck)
    return deck


def signed_areas(coords, conn):
    """Shoelace area of each quadrilateral (positive = counter-clockwise)."""
    X = coords[conn]
    x, y = X[..., 0], X[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def check_orientation(deck):
    """Continuum elements must be counter-clockwise (ABAQUS rejects others)."""
    conn, _ = deck.bulk_elements()
    a = signed_areas(deck.coords, conn)
    if (a <= 0).any():
        raise ValueError("%d continuum elements are not counter-clockwise" % int((a <= 0).sum()))
    return a


def cohesive_law(deck, material):
    """Cohesive law of a material in the deck, as tslid.fe interface parameters.

    Returns ("elastic", dict(K0)) for *Elastic, type=TRACTION only, or ("alpha", dict(K0, sigc,
    Gc, alpha, df)) for QUADS + DISPLACEMENT LINEAR / EXPONENTIAL with equal stiffnesses and
    strengths (what build_tpeel_inp.py writes for --law alpha/elastic).  TABULAR is reported as
    ("tabular", dict(K0, sigc, table)) for S3+ use.
    """
    m = deck.materials[material]
    K = m["elastic"]["values"]
    if m["elastic"]["type"] != "TRACTION" or max(K) - min(K) > 1e-9 * max(K):
        raise ValueError("expected *Elastic, type=TRACTION with equal stiffnesses")
    K0 = K[0]
    if "damage_initiation" not in m:
        return "elastic", dict(K0=K0)
    di, de = m["damage_initiation"], m["damage_evolution"]
    s = di["values"]
    if di["criterion"] != "QUADS" or max(s) - min(s) > 1e-9 * max(s):
        raise ValueError("expected QUADS with equal strengths")
    sigc = s[0]
    d0 = sigc / K0
    if de["type"] != "DISPLACEMENT":
        raise ValueError("expected *Damage Evolution, type=DISPLACEMENT")
    if de["softening"] == "LINEAR":
        u = de["rows"][0][0]
        df = d0 + u
        alpha = 0.0
        Gc = 0.5 * sigc * df
    elif de["softening"] == "EXPONENTIAL":
        u, alpha = de["rows"][0][:2]
        df = d0 + u
        g = 1.0 / alpha - 1.0 / math.expm1(alpha)
        Gc = 0.5 * sigc * d0 + sigc * u * g
    elif de["softening"] == "TABULAR":
        return "tabular", dict(K0=K0, sigc=sigc, table=np.array(de["rows"]))
    else:
        raise ValueError("unsupported softening %s" % de["softening"])
    return "alpha", dict(K0=K0, sigc=sigc, Gc=Gc, alpha=alpha, df=df)


def summary(deck):
    """Counts and geometry used by the S3a unit test."""
    conn, names = deck.bulk_elements()
    bot, top = deck.cohesive_chain()
    used = np.unique(np.r_[conn.ravel(), bot, top])
    return dict(n_nodes=len(deck.labels), n_bulk=len(conn),
                n_coh=len(bot) - 1, n_elems=len(conn) + len(bot) - 1,
                bulk_area=float(signed_areas(deck.coords, conn).sum()),
                bond_x0=float(deck.coords[bot[0], 0]), bond_x1=float(deck.coords[bot[-1], 0]),
                n_unused_nodes=int(len(deck.labels) - len(used)))
