"""ABAQUS-compatible outputs of a tslid.fe T-peel run.  Step S3c (energy gate), used again in S3d.

Units: mm, N, MPa, N*mm, s.

Writes, as the run goes, the files that src/tslid/extract_tpeel_odb.py writes for an ABAQUS run, with
the same column names, so that src/tslid/analysis/energy_checks.py runs on them unchanged:
    <name>_response.csv   time, delta, P, delta_mouth, W_ext, ALLSE, ALLPD, ALLDMD, ALLVD, ALLCD, ALLWK
                          (one row per converged increment; ABAQUS writes one per field frame)
    <name>_profiles.npz   x_face, opening (frames x pairs), time, t_bond, x_tip, traction_tip, sdeg_tip
    <name>.json           the deck's metadata (<job>.json of build_tpeel_inp.py) + "solver": "tslid.fe"
    <name>.sta            "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" only when the step completed
                          (energy_checks.completed() reads it)

Energies per converged increment n -> n+1 (W = width; per quadrature point, weights w L/2):
    W_ext   trapezoid of P d(delta), as extract_tpeel_odb.py
    ALLSE   bulk strain energy + W sum [(1 - Dv) K0/2 dm^2 + K0/2 <-dn>^2]   (stored, current Dv)
    ALLDMD  W sum G_d(dmax), the energy the inviscid law dissipates up to dmax, in closed form:
            damage grows only while dm = dmax, where Y = K0/2 dm^2, so
            G_d(d) = int_0^d Y dD = K0 d0^2/2 + F(d) - (1 - D(d)) K0 d^2/2  (d > d0; 0 below),
            F the envelope integral of the law (exact, independent of the increment size)
    ALLCD   sum of W sum 1/2 (Y_n + Y_n+1) ((Dv - D)_n+1 - (Dv - D)_n),   Y = K0/2 dm^2
            (viscous regularisation: the lag Dv - D; trapezoid in time)
    ALLPD = ALLVD = 0; ALLWK = W_ext.
[I] This split of the dissipation is ours (the inviscid law's dissipation at the current dmax, and
the rest of the energy released through Dv); ABAQUS's internal split is not documented to us.
The sum ALLDMD + ALLCD is the total cohesive dissipation either way.
Opening: U2(IFACE_U) - U2(IFACE_L), node pairs matched by x (as the extractor).  traction_tip: normal
cohesive traction per element (mean of its two points); sdeg_tip: max D per element.
"""
import csv
import json
import os

import numpy as np

from .interface import traction


class TPeelWriter:
    def __init__(self, system, deck_json, outdir, name):
        self.S, self.m = system, system.model
        m = self.m
        self.outdir, self.name = outdir, name
        self.meta = json.load(open(deck_json))
        self.meta["solver"] = "tslid.fe"
        json.dump(self.meta, open(os.path.join(outdir, name + ".json"), "w"), indent=1)
        sta = os.path.join(outdir, name + ".sta")
        if os.path.exists(sta):
            os.remove(sta)
        u, l = np.asarray(m.deck.nsets["IFACE_U"]), np.asarray(m.deck.nsets["IFACE_L"])
        u, l = u[np.argsort(m.coords[u, 0], kind="stable")], l[np.argsort(m.coords[l, 0], kind="stable")]
        if np.abs(m.coords[u, 0] - m.coords[l, 0]).max() > 1e-9:
            raise ValueError("IFACE_U / IFACE_L not paired by x")
        self.iu, self.il = 2 * u + 1, 2 * l + 1
        self.x_face = m.coords[u, 0]
        coh = m.coh
        self.wq = np.asarray(coh.w)[None, :] * (0.5 * coh.L)[:, None] * coh.width        # (n_seg, n_q)
        self.x_tip = 0.5 * (m.coords[coh.bot[:-1], 0] + m.coords[coh.bot[1:], 0])
        self.rows, self.open, self.trac, self.sdeg, self.times = [], [], [], [], []
        self.acc = dict(W_ext=0.0, ALLCD=0.0)
        self.prev = None
        self.fh = None

    def _allmd(self, dmax):
        law = self.m.law
        if law is None:
            return 0.0
        K0, d0 = self.m.K0, law.d0
        d = np.maximum(dmax, d0)
        Gd = 0.5 * K0 * d0 ** 2 + np.asarray(law.envelope_integral(d)) \
            - (1.0 - np.asarray(law.damage(d))) * 0.5 * K0 * d ** 2
        return float(np.sum(self.wq * np.where(dmax > d0, Gd, 0.0)))

    def _Y(self, q):
        dn, ds = self.S.separations(q)
        dn, ds = np.asarray(dn), np.asarray(ds)
        return dn, ds, 0.5 * self.m.K0 * (np.maximum(dn, 0.0) ** 2 + ds ** 2)

    def add(self, t, q, hist, P, delta):
        """Record the converged state (q, hist) at time t (call for t = 0 too)."""
        dn, ds, Y = self._Y(q)
        D, Dv = np.asarray(hist["D"]), np.asarray(hist["Dv"])
        K0 = self.m.K0
        se_coh = float(np.sum(self.wq * ((1.0 - Dv) * Y + 0.5 * K0 * np.maximum(-dn, 0.0) ** 2)))
        if self.prev is not None:
            Pp, dp, Yp, Dp, Dvp = self.prev
            self.acc["W_ext"] += 0.5 * (P + Pp) * (delta - dp)
            self.acc["ALLCD"] += float(np.sum(self.wq * 0.5 * (Y + Yp) * ((Dv - D) - (Dvp - Dp))))
        self.prev = (P, delta, Y, D, Dv)
        U = self.m.lift(q)
        op = U[self.iu] - U[self.il]
        row = dict(time=t, delta=delta, P=P, delta_mouth=float(op[0]) if self.x_face[0] < 1e-6 else np.nan,
                   W_ext=self.acc["W_ext"], ALLSE=self.S.bulk_energy(q) + se_coh, ALLPD=0.0,
                   ALLDMD=self._allmd(np.asarray(hist["dmax"])), ALLVD=0.0, ALLCD=self.acc["ALLCD"],
                   ALLWK=self.acc["W_ext"])
        tn, _ = traction(dn, ds, Dv, K0)
        self.open.append(op)
        self.trac.append(np.asarray(tn).mean(axis=1))
        self.sdeg.append(D.max(axis=1))
        self.times.append(t)
        if self.fh is None:
            self.fh = open(os.path.join(self.outdir, self.name + "_response.csv"), "w", newline="")
            self.w = csv.DictWriter(self.fh, fieldnames=list(row))
            self.w.writeheader()
        self.w.writerow({k: "%.10g" % v for k, v in row.items()})
        self.fh.flush()

    def write_profiles(self):
        p = os.path.join(self.outdir, self.name + "_profiles.npz")
        np.savez_compressed(p + ".tmp.npz", x_face=self.x_face, opening=np.array(self.open), time=np.array(self.times),
                            t_bond=self.meta["params"].get("t", 0.4), x_tip=self.x_tip,
                            traction_tip=np.array(self.trac), sdeg_tip=np.array(self.sdeg))
        os.replace(p + ".tmp.npz", p)

    def close(self, completed):
        self.write_profiles()
        if self.fh:
            self.fh.close()
        if completed:
            open(os.path.join(self.outdir, self.name + ".sta"), "w").write(
                "tslid.fe run (not ABAQUS)\n THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n")
