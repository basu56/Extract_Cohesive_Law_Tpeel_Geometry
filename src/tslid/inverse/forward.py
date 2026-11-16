"""Forward model for the inversions: tslid.fe production solve + S5a direct sensitivities, observed with
the D29 bin operator.  I1.

Units: mm, N, MPa.  theta = (ln sigc, ln Gc, alpha) by default (K0 fixed at the deck value, D31); any
subset of ALPHA_LOG can be chosen.  The model (TPeelModel + ElementSystem + DirectSensitivity) is built
once; the law is a jit argument (S5.0), so evaluations at new theta reuse the compiled kernels.  Each
evaluation runs the deck's own adaptive schedule (nothing is replayed, D31) with the D24 solver settings.
"""
import time

import numpy as np

from tslid.fe.assembly import ElementSystem
from tslid.fe.incremental import SolverOptions, newton_energy, run
from tslid.fe.interface import AlphaLawParams
from tslid.fe.linsolve import make_solver
from tslid.fe.observation import observation_matrix
from tslid.fe.sensitivity import ALPHA_LOG, DirectSensitivity, ElementProblem, alpha_log_basis, subset_basis
from tslid.fe.tpeel import TPeelModel


class Forward:
    """evaluate(theta, edges_by_window) -> dict(ok, delta, P, S, obs={window: (y, Sbar)}, wall, ...).

    Args:
        deck: production deck (.inp) of the case (geometry, mesh, K0, viscosity, schedule).
        params: names in ALPHA_LOG that make up theta (default ln_sigc, ln_Gc, alpha).
        linsolve, rtol: D24 defaults.
    """

    def __init__(self, deck, params=("ln_sigc", "ln_Gc", "alpha"), linsolve="pardiso_sym", rtol=1e-6,
                 max_increments=10 ** 6):
        t0 = time.perf_counter()
        self.m = m = TPeelModel(deck)
        self.S = ElementSystem(m, condense=True)
        self.solver = make_solver(linsolve)
        self.params = tuple(params)
        self.cols = [ALPHA_LOG.index(p) for p in self.params]
        self.K0 = float(m.law.K0)
        self.law_deck = m.law
        self.prob = ElementProblem(self.S, self.solver)
        self.ds = DirectSensitivity(self.prob, m.law, basis=subset_basis(alpha_log_basis(m.law), self.cols),
                                    names=self.params)
        self.opt = SolverOptions(rtol=rtol, linsolve=self.solver, diverge_after=4)
        self.st = m.deck.step["static"]
        self.max_increments = max_increments
        # compile once (Newton direction + sensitivity kernels) outside the timed evaluations
        hz = m.zero_history()
        h0 = m.frozen_args(hz, self.st["dt0"])
        q0 = m.apply_bc(m.zero_q(), 0.0)
        E0, r0 = self.S.value_and_grad(q0, h0)
        self.S.direction(q0, h0, r0, self.solver)
        self.ds.step(q0, hz, h0, hz, self.st["dt0"])
        self.ds.reset()
        self.t_setup = time.perf_counter() - t0
        self.n_eval = 0

    def theta_true(self):
        """theta of the deck's law (the truth when the deck is the truth case)."""
        lw = self.law_deck
        full = dict(ln_K0=np.log(lw.K0), ln_sigc=np.log(lw.sigc), ln_Gc=np.log(lw.Gc), alpha=lw.alpha)
        return np.array([full[p] for p in self.params])

    def law(self, theta):
        v = dict(ln_K0=np.log(self.K0), ln_sigc=np.log(self.law_deck.sigc), ln_Gc=np.log(self.law_deck.Gc),
                 alpha=self.law_deck.alpha)
        v.update({p: float(x) for p, x in zip(self.params, theta)})
        return AlphaLawParams(float(np.exp(v["ln_K0"])), float(np.exp(v["ln_sigc"])), float(np.exp(v["ln_Gc"])),
                              float(v["alpha"]))

    def evaluate(self, theta, edges_by_window, sens=True):
        t0 = time.perf_counter()
        lw = self.law(theta)
        self.S.set_theta(lw)
        m = self.m
        if sens:
            self.ds.reset(lw, subset_basis(alpha_log_basis(lw), self.cols))
        res = run(self.S, self.st["T"], self.st["dt0"], self.st["dtmin"], self.st["dtmax"], self.opt,
                  record=lambda mm, t, q, hist, h, r: (float(q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]]),
                                                       float(r[m.out["rp1_u2"]])),
                  newton=newton_energy, on_converged=self.ds.step if sens else None,
                  max_increments=self.max_increments)
        self.n_eval += 1
        out = dict(theta=np.asarray(theta, float).tolist(), law=dict(K0=lw.K0, sigc=lw.sigc, Gc=lw.Gc, alpha=lw.alpha),
                   ok=not res.failed, message=res.message, increments=len(res.records) - 1, cutbacks=res.cutbacks,
                   wall=time.perf_counter() - t0, wall_sens=self.ds.wall if sens else 0.0)
        d = np.array([x[0] for x in res.records])
        if res.failed and not ("max_increments" in res.message and
                               all(d[-1] >= e[-1] for e in edges_by_window.values())):
            return out                                       # failed (cutback limit): no observations
        P = np.array([x[1] for x in res.records])
        S = np.vstack([np.zeros((1, len(self.cols))), self.ds.as_arrays()["P"]]) if sens else None
        out.update(delta=d, P=P, S=S, obs={})
        for w, e in edges_by_window.items():
            O = observation_matrix(d, e)
            out["obs"][w] = (O @ P, O @ S if sens else None)
        return out
