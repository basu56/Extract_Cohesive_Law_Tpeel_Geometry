# UNVERIFIED PROTOTYPE (Claude Web, 2026-11-14). API patterns only; see docs/solver/S0_S2_INSTRUCTIONS.md.
# Dense tangent, plain Quad4, frozen history per step; not validated against ABAQUS or theory.
# Result when written: implicit single-step dP/dsigc = -0.017154 vs FD -0.017140 (0.08 %).
# Smoke test: 2D plane-strain DCB on tatva (Quad4 bulk, Line2 cohesive), bilinear law with
# frozen history per step, Newton solve, and an implicit-function gradient dP/dsigc vs FD.
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from tatva import Mesh, Operator, element
L, H, a0, nx, ny = 30.0, 1.0, 10.0, 60, 2
E, nu = 100e3, 0.3                      # MPa
mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
def arm(y0):
    xs, ys = np.linspace(0, L, nx+1), np.linspace(y0, y0+H, ny+1)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); c = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i*(ny+1)+j
    el = [[idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1)] for i in range(nx) for j in range(ny)]
    return c, np.array(el)
cb, eb = arm(-H); ct, et = arm(0.0); nb = len(cb)
coords = np.vstack([cb, ct]); elems = np.vstack([eb, et+nb])
bot_if = np.array([i*(ny+1)+ny for i in range(nx+1)]); top_if = np.array([i*(ny+1) for i in range(nx+1)]) + nb
bonded = np.where(coords[bot_if,0] >= a0)[0]
lines = np.array([[bonded[k], bonded[k+1]] for k in range(len(bonded)-1)])
op = Operator(Mesh(jnp.array(coords), jnp.array(elems)), element.Quad4())
if_op = Operator(Mesh(jnp.array(coords[bot_if]), jnp.array(lines)), element.Line2())
ndof = 2*len(coords)
def bulk(u):
    g = op.grad(u.reshape(-1,2)); eps = 0.5*(g+jnp.swapaxes(g,-1,-2))
    tr = jnp.trace(eps, axis1=-2, axis2=-1)
    return op.integrate(0.5*lam*tr**2 + mu*jnp.sum(eps*eps, axis=(-2,-1)))
K0, Gc = 1e4, 0.015                      # MPa/mm, N/mm
def coh(u, dmax, sigc):
    U = u.reshape(-1,2); dn = if_op.eval((U[top_if]-U[bot_if])[:,1])
    d0 = sigc/K0; df = 2*Gc/sigc
    # secant energy with damage frozen at dmax: psi = 0.5*k_s*dn^2, k_s from bilinear law at dmax
    dm = jnp.maximum(dmax, d0)
    ts = jnp.where(dm < df, sigc*(df-dm)/(df-d0), 0.0)
    ks = ts/dm
    pos = jnp.maximum(dn, 0.0)
    return if_op.integrate(0.5*ks*pos**2 + 0.5*K0*jnp.minimum(dn,0.0)**2)
def energy(u, dmax, sigc): return bulk(u) + coh(u, dmax, sigc)
xs = coords[:,0]; right = np.where(np.isclose(xs, L))[0]
ltop = np.where(np.isclose(xs,0)&np.isclose(coords[:,1],H))[0]; lbot = np.where(np.isclose(xs,0)&np.isclose(coords[:,1],-H))[0]
fixed = np.r_[2*right, 2*right+1, 2*ltop+1, 2*lbot+1]; free = np.setdiff1d(np.arange(ndof), fixed)
def ufull(uf, Uapp):
    u = jnp.zeros(ndof).at[free].set(uf); u = u.at[2*ltop+1].set(Uapp).at[2*lbot+1].set(-Uapp); return u
res = jax.jit(lambda uf,U,dm,s: jax.grad(energy)(ufull(uf,U),dm,s)[free])
Kff = jax.jit(jax.jacfwd(res, argnums=0))
def solve(uf, U, dm, s):
    for _ in range(30):
        r = res(uf,U,dm,s)
        if jnp.linalg.norm(r) < 1e-9: break
        uf = uf - jnp.linalg.solve(Kff(uf,U,dm,s), r)
    return uf
force = jax.jit(lambda uf,U,dm,s: jnp.sum(jax.grad(energy)(ufull(uf,U),dm,s)[2*ltop+1]))
def opening(uf,U):
    Ufl = ufull(uf,U).reshape(-1,2); return if_op.eval((Ufl[top_if]-Ufl[bot_if])[:,1])
def run(sigc, steps=np.linspace(0,0.25,26)[1:]):
    uf = jnp.zeros(len(free)); dm = jnp.zeros((len(lines),1)); P=[]
    for U in steps:
        for _ in range(20):                      # staggered: re-solve until history stable
            uf = solve(uf,U,dm,sigc); dnew = jnp.maximum(dm, opening(uf,U))
            if jnp.max(jnp.abs(dnew-dm)) < 1e-10: break
            dm = dnew
        P.append(float(force(uf,U,dm,sigc)))
    return np.array(P), uf, dm, steps[-1]
P1, uf, dm, U = run(1.0)
print("P (N/mm) every 10 steps:", np.round(P1[::5],4), " peak", P1.max().round(4))
# implicit gradient of final-step force wrt sigc, holding history fixed (single-step sensitivity)
s = 1.0
drds = jax.jacfwd(res, argnums=3)(uf,U,dm,s)
dudS = -jnp.linalg.solve(Kff(uf,U,dm,s), drds)
dPdu = jax.grad(force, argnums=0)(uf,U,dm,s); dPds_exp = jax.grad(force, argnums=3)(uf,U,dm,s)
print("implicit dP/dsigc (frozen history):", float(dPdu@dudS + dPds_exp))
h=1e-6; up = solve(uf,U,dm,s+h); dn_ = solve(uf,U,dm,s-h)
print("FD       dP/dsigc (frozen history):", float((force(up,U,dm,s+h)-force(dn_,U,dm,s-h))/(2*h)))
