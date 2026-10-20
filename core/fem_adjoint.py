"""Exact shape adjoint for the KS-pooled von Mises stress: every per-tile
sensitivity from ONE matrix factorization, at any tile count.

WHY THIS EXISTS. ccx's *SENSITIVITY is already O(1) in tile count (3 solves),
but it builds d(K_e)/d(x) by FINITE-DIFFERENCING the element stiffness with a
step of one element edge length -- the classic "semi-analytical" method. That
differencing carries an error proportional to the element's rigid-body motion
divided by its size, and on hollow_cube's 840-tile grid it dominates: the
contraction sums to -2.79x the directly measured d(KS)/d(uniform thickening),
i.e. the wrong sign, while the remesh FD sums to 1.0026x (see
evaluators/gradient_sum_rule.py, and the per-tile map in
evaluators/tile_gradient_map.py). On the 48-tile grid the same code gets
0.877x, which is why every earlier validation passed.

Here d(K_e)/d(x) is not differenced at all. K_e(x) is written as a torch
expression over the element's own node coordinates and reverse-mode
autodiff supplies the derivative to machine precision, so the error term that
breaks the ccx route cannot arise.

THE ADJOINT. For KS = KS(u(X), X) with K(X) u = f:

    dKS/dX = @KS/@X|_u  -  lambda^T (@K/@X) u ,     K lambda = (@KS/@u)^T

f does not depend on X here (a fixed resultant on a rigid-coupled face), and
K is symmetric, so lambda reuses the SAME factorization as u: the whole
gradient costs one extra back-substitution, not one extra solve, and nothing
in it scales with the number of design variables.

Neither term is assembled as a matrix. lambda^T (@K/@X) u is the gradient of
the SCALAR sum_e lambda_e^T K_e(X_e) u_e with u and lambda held fixed, which
is one batched backward pass over the elements; @KS/@X|_u and (@KS/@u)^T come
from the same expression graph that computes KS.

Per-tile values follow by the chain rule with the geometric Jacobian
dX/dt from core.geometric_jacobian, exactly as the ccx route does -- that
half was audited and is exact (of hollow_cube's 40320 moving nodes, none is
shared between tiles, and a single-tile bump reproduces dX/dt to 1.00x).

ELEMENT. C3D8I -- ccx's own, to 1.7e-5% -- with 2x2x2 Gauss. Wilson's three
incompatible modes are added and statically condensed out, and autodiff
differentiates straight through the 9x9 condensation, which is exactly what
ccx's semi-analytical route cannot do and why *SENSITIVITY has to fall back to
plain C3D8 for its gradient.

Getting C3D8I to actually BE ccx's C3D8I turns on one detail with no
principled answer, only an empirical one: how the incompatible modes are
mapped to physical coordinates. ccx uses the Jacobian at the ELEMENT CENTRE
and applies NO detJ0/detJ scaling. That was pinned by running ccx itself on a
tapered, sheared, single-element-thick cantilever -- a shape on which the
candidates separate hugely (C3D8 is 79% soft) where a regular parallelepiped
would have shown nothing, J being constant there so every variant collapses
onto the same matrix. Over nine distortions the centre-Jacobian-unscaled form
tracks ccx to <2e-5%, while adding the detJ0/detJ factor (the Taylor form this
file used to carry) costs up to 2.3% and mapping at the Gauss point instead
(classic Wilson) up to 75%.

This matters far more than a percent or two of element accuracy, because the
error it removes is DESIGN-DEPENDENT. Plain C3D8 agreed with ccx to +0.3% at
hollow_cube's uniform t=0.30 start -- which is why every earlier validation of
this file passed -- and drifted to reading 32% LOW by the time topopt job
53474 had thinned the walls (adjoint 76 MPa against ccx's 100.4). A constant
offset can be calibrated out; a drifting one cannot, and in 53474 it moved the
constraint under the optimizer for 30 generations and then left it parked ~12%
heavier than the true feasible frontier.

STRESS. Gauss-point stresses extrapolated to the element's nodes with the
standard 2x2x2 inverse map (the +-sqrt(3) evaluation), then averaged over the
elements meeting at each node -- CalculiX's own recovery -- and pooled with
the same KS as Mesh.get_aggregated_stress.
"""
import os
import threading
import weakref
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

torch.set_default_dtype(torch.float64)

_G = 1.0 / np.sqrt(3.0)
# C3D8 corner order (Abaqus/CalculiX): bottom face 1-4, top face 5-8.
_CORNER = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
_GP = np.array([[a, b, c] for c in (-_G, _G) for b in (-_G, _G) for a in (-_G, _G)])


def _dN(xi):
    """dN/d(xi,eta,zeta) for the 8 corner shape functions at natural points
    xi of shape (P,3). Returns (P,8,3)."""
    s = torch.as_tensor(_CORNER)
    x = torch.as_tensor(xi)
    a = 1.0 + x[:, None, 0] * s[None, :, 0]
    b = 1.0 + x[:, None, 1] * s[None, :, 1]
    c = 1.0 + x[:, None, 2] * s[None, :, 2]
    return 0.125 * torch.stack([s[None, :, 0] * b * c,
                                s[None, :, 1] * a * c,
                                s[None, :, 2] * a * b], dim=-1)


def _elastic(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = torch.zeros(6, 6)
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


# (Voigt row, local dof, derivative axis) -> 1 for the strain-displacement
# pattern, Voigt order (xx, yy, zz, xy, xz, yz) -- CalculiX's own.
_VOIGT = ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 0, 1), (3, 1, 0),
          (4, 0, 2), (4, 2, 0), (5, 1, 2), (5, 2, 1))
_SEL = {}


def _selector(n):
    """Constant (6, 3n, n, 3) 0/1 tensor with B[k, 3i+c] = dNdx[i, a] where
    (k, c, a) is in _VOIGT. Building B as ONE contraction with it, instead of
    81 in-place slice writes, keeps autograd out of CopySlices: the slice-write
    version copied the whole B on every write in the backward pass, which was
    72% of ShapeAdjoint.gradient's time (cross4 at 6,6,4: 311 s single-thread)."""
    if n not in _SEL:
        S = torch.zeros(6, 3 * n, n, 3, dtype=torch.float64)
        for i in range(n):
            for k, c, a in _VOIGT:
                S[k, 3 * i + c, i, a] = 1.0
        _SEL[n] = S
    return _SEL[n]


def _B(dNdx):
    """Strain-displacement matrix from dN/dx (…,n,3) -> (…,6,3n), Voigt order
    (xx, yy, zz, xy, xz, yz) -- CalculiX's own."""
    return torch.einsum("...ia,kcia->...kc", dNdx, _selector(dNdx.shape[-2]).to(dNdx))


def _btdb(B1, D, B2, w):
    """sum_p w_mp B1_mp^T D B2_mp, (M,P,6,i) x (6,6) x (M,P,6,j) -> (M,i,j), as
    one batched matmul. torch.einsum("mpki,mpkl,mplj,mp->mij") with D
    expanded per element contracted this as many small bmm calls (84% of
    element_stiffness's time); this is the same sum in two dense products."""
    M, P = B1.shape[:2]
    DB2 = torch.matmul(D, B2) * w[:, :, None, None]                 # (M,P,6,j)
    return torch.bmm(B1.reshape(M, P * 6, -1).transpose(1, 2), DB2.reshape(M, P * 6, -1))


def _geom(coords):
    """Per-element, per-Gauss-point B, detJ. coords (M,8,3) -> (M,P,6,24), (M,P)."""
    dN = _dN(_GP)                                  # (P,8,3)
    # J[a][b] = dx_b/dxi_a, so the map from natural to physical gradients is
    # its inverse TRANSPOSED: dxi_a/dx_b = (J^-1)[b][a]. Getting this wrong is
    # invisible on a parallelepiped (J is then symmetric enough that the
    # single-element tension test still passes exactly) and wrong on every
    # distorted element, which is all of them on a real mesh.
    J = torch.einsum("pia,mib->mpab", dN, coords)  # (M,P,3,3)
    detJ = torch.linalg.det(J)
    dNdx = torch.einsum("pia,mpba->mpib", dN, torch.linalg.inv(J))
    return _B(dNdx), detJ


def element_stiffness(coords, E, nu, incompatible=True):
    """K_e for a batch of hexes. coords (M,8,3) -> (M,24,24).

    incompatible=True (the default) adds Wilson's three incompatible modes
    (9 internal DOF) and condenses them out, which is what makes a single
    element through a wall able to bend -- the C3D8I this project's forward
    solves use, and the reason plain C3D8 under-predicts peak stress here.
    Pass False only to reproduce a plain-C3D8 reference; it is not the
    element the rest of the pipeline runs on. See _bubble_B for the mapping
    that makes this ccx's C3D8I rather than a lookalike.
    """
    D = _elastic(E, nu).to(coords)
    B, detJ = _geom(coords)
    w = detJ                                        # unit Gauss weights
    Kuu = _btdb(B, D, B, w)
    if not incompatible:
        return Kuu

    Ba = _bubble_B(coords)
    Kua = _btdb(B, D, Ba, w)
    Kaa = _btdb(Ba, D, Ba, w)
    return Kuu - Kua @ torch.linalg.solve(Kaa, Kua.transpose(1, 2))


def gauss_to_node():
    """2x2x2 Gauss -> corner extrapolation matrix (8,8): the trilinear shape
    functions evaluated at +-sqrt(3) in the Gauss points' own natural frame,
    which is CalculiX's stress recovery."""
    s = _CORNER * np.sqrt(3.0)
    g = _GP / _G                                  # the +-1 lattice of the gauss pts
    E = np.array([[np.prod(0.5 * (1 + s[n] * g[p])) for p in range(8)]
                  for n in range(8)])
    return torch.as_tensor(E)


def _pairing(coords, ue, le, E, nu, incompatible):
    """sum_e  le_e^T K_e(coords_e) ue_e, as a single scalar.

    This is what the adjoint term lambda^T (@K/@X) u is the X-gradient of, and
    forming it directly avoids ever building the 24x24 K_e (or its derivative)
    for the whole mesh: the quadratic form costs one B-multiply per Gauss
    point. The condensed bubble block carries through the same way.
    """
    D = _elastic(E, nu).to(coords)
    B, detJ = _geom(coords)
    Bu = torch.einsum("mpki,mi->mpk", B, ue)
    Bl = torch.einsum("mpki,mi->mpk", B, le)
    val = torch.einsum("mpk,kl,mpl,mp->", Bl, D, Bu, detJ)
    if not incompatible:
        return val
    Ba = _bubble_B(coords)
    Au = torch.einsum("mpki,mpk->mi", Ba, torch.einsum("kl,mpl,mp->mpk", D, Bu, detJ))
    Al = torch.einsum("mpki,mpk->mi", Ba, torch.einsum("kl,mpl,mp->mpk", D, Bl, detJ))
    Kaa = _btdb(Ba, D, Ba, detJ)
    return val - torch.einsum("mi,mi->", Al, torch.linalg.solve(Kaa, Au[..., None])[..., 0])


def _bubble_B(coords):
    """Strain-displacement block of Wilson's three incompatible modes,
    (M,P,6,9), mapped with the Jacobian at the ELEMENT CENTRE and with NO
    detJ0/detJ scaling. That combination is not a matter of taste: it is the
    one CalculiX implements, pinned by running ccx itself on a tapered and
    sheared single-element-thick cantilever (see the module header). Adding
    the detJ0/detJ factor -- the Taylor form this used to carry -- costs up to
    2.3% of tip deflection, and mapping with the Gauss-point Jacobian instead
    (classic Wilson) costs up to 75%."""
    dN0 = _dN(np.zeros((1, 3)))
    J0 = torch.einsum("pia,mib->mpab", dN0, coords)[:, 0]
    invJ0 = torch.linalg.inv(J0)
    xi = torch.as_tensor(_GP).to(coords)
    dNa = torch.zeros(len(_GP), 3, 3).to(coords)
    for a in range(3):
        dNa[:, a, a] = -2.0 * xi[:, a]
    dNadx = torch.einsum("pab,mcb->mpac", dNa, invJ0)
    return _B(dNadx)                               # same Voigt pattern, 3 modes -> (M,P,6,9)


class ShapeAdjoint:
    """Linear elastic solve + the exact d(ks_stress)/d(node coordinate), from
    one factorization.

    Reproduces the boundary conditions Mesh._ccx_base_deck writes, so the
    forward answer is comparable with the ccx path it is meant to replace:
    anchored nodes fully fixed, and the loaded nodes tied into ONE rigid body
    (translation + rotation) carrying the resultant force, which is
    CalculiX's *RIGID BODY and MAPDL's CERIG pilot.

    The rigid tie is built once from the reference coordinates and held
    constant through the derivative. That is exact here rather than an
    approximation: the tied nodes sit on the outer, tile-shared faces, which
    IRIT's mesher does not move when a tile's shell thickness changes
    (compute_thickness_jacobian's dX/dt is identically zero there), so the
    tie's own coordinate dependence contributes nothing to any per-tile
    sensitivity. assert_tie_is_fixed() checks that against a given dX/dt
    rather than trusting it.
    """

    def __init__(self, nodes, elements, anchored_ids, force_nodes, total_force,
                 young, poisson, incompatible=True, chunk=4096,
                 rigid_tie=True, nodal_forces=None, e_scale=None, stress_scale=None):
        self.ids = sorted(nodes)
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.N = len(self.ids)
        self.X = torch.tensor([list(nodes[n]) for n in self.ids], requires_grad=True)
        self.eids = sorted(elements)
        self.conn = np.array([[self.index[n] for n in elements[e][:8]]
                              for e in self.eids], dtype=np.int64)
        self.e_scale = (np.ones(len(self.eids)) if e_scale is None
                        else np.asarray(e_scale, dtype=float))
        # stress_scale: per-element multiplier of the stress counted in KS,
        # applied to the REAL material's stress (E, not the penalized s E):
        # the separate stress relaxation. None: the counted stress is the
        # penalized one (multiplier = e_scale, relaxation tied to stiffness).
        self.stress_scale = None if stress_scale is None else np.asarray(stress_scale, dtype=float)
        self.gE = None
        self.gW = None
        self.young, self.poisson = float(young), float(poisson)
        # e_scale: per-element Young's modulus multiplier, self.eids order
        # (stiffness penalty, TOPOPT_STIFF_PENALTY). K_e and the element's
        # stress are both linear in it; gradient() also returns d KS / d scale
        # per element in self.gE.
        self.incompatible = incompatible
        self.chunk = chunk

        fixed = {self.index[n] for n in anchored_ids}
        # rigid_tie=False loads each node directly instead of through the
        # pilot -- not the production model, but the configuration in which
        # this solver can be compared against a ccx deck one difference at a
        # time.
        slave = ([self.index[n] for n in force_nodes if self.index[n] not in fixed]
                 if rigid_tie else [])
        free = [i for i in range(self.N) if i not in fixed and i not in set(slave)]
        self.free, self.slave, self.fixed = free, slave, fixed

        cols = {n: 3 * k for k, n in enumerate(free)}
        p = 3 * len(free)
        self.n_red = p + (6 if slave else 0)
        rows, cs, vals = [], [], []
        for n, c in cols.items():
            for k in range(3):
                rows.append(3 * n + k); cs.append(c + k); vals.append(1.0)
        if slave:
            Xn = self.X.detach().numpy()
            centre = Xn[slave].mean(axis=0)
            for n in slave:
                rx, ry, rz = Xn[n] - centre
                for k, (t, a, b) in enumerate(((0, (4, rz), (5, -ry)),
                                                (1, (5, rx), (3, -rz)),
                                                (2, (3, ry), (4, -rx)))):
                    rows += [3 * n + t, 3 * n + t, 3 * n + t]
                    cs += [p + t, p + a[0], p + b[0]]
                    vals += [1.0, a[1], b[1]]
            self.pilot_col = p
        else:
            self.pilot_col = None
        self.T = sp.coo_matrix((vals, (rows, cs)),
                               shape=(3 * self.N, self.n_red)).tocsr()

        self.f_red = np.zeros(self.n_red)
        if slave:
            self.f_red[p:p + 3] = np.asarray(total_force, dtype=float)
        elif nodal_forces:
            f_full = np.zeros(3 * self.N)
            for nid, fv in nodal_forces.items():
                f_full[3 * self.index[nid]:3 * self.index[nid] + 3] = fv
            self.f_red = self.T.T @ f_full

        self._lu = None
        # Optional KS node mask (bool, self.ids order; True = counts). Nodes
        # left out -- e.g. the load-introduction zone -- get zero weight in
        # ks() and gradient() alike, so value and gradient stay one functional.
        self.ks_mask = None
        self.u = None
        with torch.no_grad():
            _, detJ = _geom(self.X[self.conn[:min(len(self.conn), 20000)]])
        self.min_detJ = float(detJ.min())

    # ------------------------------------------------------------------ solve
    def _assemble(self):
        rows, cols, vals = [], [], []
        for a in range(0, len(self.conn), self.chunk):
            blk = self.conn[a:a + self.chunk]
            with torch.no_grad():
                Ke = element_stiffness(self.X[blk], self.young, self.poisson,
                                        self.incompatible).numpy()
            Ke = Ke * self.e_scale[a:a + self.chunk][:, None, None]
            dof = (3 * blk[:, :, None] + np.arange(3)[None, None, :]).reshape(len(blk), 24)
            rows.append(np.repeat(dof, 24, axis=1).ravel())
            cols.append(np.tile(dof, (1, 24)).ravel())
            vals.append(Ke.reshape(-1))
        K = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(3 * self.N, 3 * self.N)).tocsr()
        return self.T.T @ K @ self.T

    def solve(self):
        """Factorize once and back-substitute for u. The factorization is kept:
        the adjoint reuses it, so the gradient costs no additional solve."""
        self._lu = _SPDFactor(self._assemble())
        q = self._lu.solve(self.f_red)
        self.u = torch.tensor(self.T @ q, requires_grad=True)
        return self.u

    # ------------------------------------------------------- stress and KS
    def _nodal_stress_terms(self, blk, coords, ue, scale=None):
        """Extrapolated stress tensors at each element's own 8 corners,
        (m,8,6), for one chunk.

        Under incompatible modes the element's strain is NOT B ue. The three
        Wilson modes carry their own strain Ba ae, and ae was condensed out of
        the stiffness, so it has to be recovered here:

            ae = -Kaa^-1 Kua^T ue ,     eps = B ue + Ba ae

        Skipping that term is a silent, design-dependent error rather than a
        missing constant: it leaves C3D8I stiffness (so C3D8I displacements)
        feeding a C3D8-shaped stress recovery, which is neither element. On
        hollow_cube it read 12% HIGH at the uniform start and 17% LOW on a
        thinned design -- it does not even keep its sign. autodiff runs
        straight through the 9x9 solve, so the adjoint picks this up for free
        and the value and its gradient stay the same functional.
        """
        D = _elastic(self.young, self.poisson).to(coords)
        B, detJ = _geom(coords)
        eps = torch.einsum("mplj,mj->mpl", B, ue)                   # (m,P,6)
        if self.incompatible:
            Ba = _bubble_B(coords)
            Kua = _btdb(B, D, Ba, detJ)
            Kaa = _btdb(Ba, D, Ba, detJ)
            rhs = torch.einsum("mij,mi->mj", Kua, ue)               # Kua^T ue
            ae = -torch.linalg.solve(Kaa, rhs[..., None])[..., 0]   # (m,9)
            eps = eps + torch.einsum("mplj,mj->mpl", Ba, ae)
        sig_gp = torch.einsum("kl,mpl->mpk", D, eps)                # (m,P,6)
        out = torch.einsum("np,mpk->mnk", gauss_to_node().to(coords), sig_gp)
        if scale is not None:                       # stiffness penalty: sigma = s E eps
            out = out * scale[:, None, None]
        return out

    def nodal_stress(self):
        """Node id -> von Mises, from the element-averaged stress tensor --
        CalculiX's recovery, so this is comparable with parse_frd's output."""
        acc = torch.zeros(self.N, 6)
        cnt = torch.zeros(self.N)
        with torch.no_grad():
            for a in range(0, len(self.conn), self.chunk):
                blk = self.conn[a:a + self.chunk]
                mult = self.e_scale if self.stress_scale is None else self.stress_scale
                s = self._nodal_stress_terms(blk, self.X[blk], self.u[self._dof(blk)],
                                             torch.as_tensor(mult[a:a + self.chunk]))
                flat = torch.as_tensor(blk.reshape(-1))
                acc.index_add_(0, flat, s.reshape(-1, 6))
                cnt.index_add_(0, flat, torch.ones(len(flat)))
        S = acc / cnt[:, None].clamp(min=1)
        return S, _vm(S)

    def _dof(self, blk):
        return torch.as_tensor(
            (3 * blk[:, :, None] + np.arange(3)[None, None, :]).reshape(len(blk), 24))

    def ks(self, rho_mpa=0.15):
        _, vm = self.nodal_stress()
        if self.ks_mask is not None:
            vm = vm[torch.as_tensor(self.ks_mask, dtype=torch.bool)]
        rho = rho_mpa * 1e-6
        m = vm.max()
        return float(m + torch.log(torch.exp(rho * (vm - m)).sum()) / rho)

    # --------------------------------------------------------------- gradient
    def gradient(self, rho_mpa=0.15):
        """d(ks_stress)/d(node coordinate), (N,3) in the order of self.ids.

        Two chunked backward passes and one back-substitution:
          pass 1  @KS/@X at fixed u, and the adjoint load (@KS/@u)^T
          solve   K lambda = (@KS/@u)^T, reusing the existing factorization
          pass 2  -lambda^T (@K/@X) u, as the X-gradient of a scalar pairing
        """
        S, vm = self.nodal_stress()
        rho = rho_mpa * 1e-6
        if self.ks_mask is not None:                      # excluded nodes: zero weight
            vm = torch.where(torch.as_tensor(self.ks_mask, dtype=torch.bool), vm,
                             torch.full_like(vm, -float("inf")))
        w = torch.softmax(rho * vm, dim=0)               # @KS/@sigma_i
        Sg = S.clone().requires_grad_(True)
        _vm(Sg).mul(w).sum().backward()
        d = Sg.grad                                       # (N,6) = w_i @vm/@S_i

        cnt = torch.zeros(self.N)
        for a in range(0, len(self.conn), self.chunk):
            flat = torch.as_tensor(self.conn[a:a + self.chunk].reshape(-1))
            cnt.index_add_(0, flat, torch.ones(len(flat)))
        dn = d / cnt[:, None].clamp(min=1)

        if self.X.grad is not None:
            self.X.grad = None
        if self.u.grad is not None:
            self.u.grad = None
        Es = torch.tensor(self.e_scale, requires_grad=True)
        Ws = Es if self.stress_scale is None else torch.tensor(self.stress_scale, requires_grad=True)
        for a in range(0, len(self.conn), self.chunk):
            blk = self.conn[a:a + self.chunk]
            s = self._nodal_stress_terms(blk, self.X[blk], self.u[self._dof(blk)],
                                         Ws[a:a + self.chunk])
            (s * dn[torch.as_tensor(blk)]).sum().backward()
        gX_explicit = self.X.grad.clone()
        g_u = self.u.grad.detach().numpy()

        lam = torch.tensor(self.T @ self._lu.solve(self.T.T @ g_u))

        self.X.grad = None
        for a in range(0, len(self.conn), self.chunk):
            blk = self.conn[a:a + self.chunk]
            dof = self._dof(blk)
            # lambda_e scaled by the element's stiffness scale: the pairing is
            # sum_e s_e lambda_e^T K_e(E) u_e, so this is -lambda^T (dK/dX) u
            # for the penalized K, and its Es-gradient the implicit d KS / d s_e
            (-_pairing(self.X[blk], self.u[dof].detach(),
                       lam[dof] * Es[a:a + self.chunk][:, None],
                       self.young, self.poisson, self.incompatible)).backward()
        self.gE = Es.grad.detach().numpy().copy()     # d KS / d e_scale, per element
        self.gW = (None if self.stress_scale is None    # d KS / d stress_scale
                   else Ws.grad.detach().numpy().copy())
        return gX_explicit + self.X.grad

    def tile_gradient(self, dX_dt, elem_to_tile, tile_keys, NX, NY, NZ,
                      rho_mpa=0.15):
        """dims key -> d(ks_stress)/d(thickness), chaining the nodal gradient
        with the geometric Jacobian exactly as the ccx route does.

        rho_mpa MUST be the same sharpness the caller reported ks() at.
        This defaulted to gradient()'s own 0.15 while the evaluator passed
        its --stress_ks_rho_mpa to ks() only, so any run that changed rho
        silently optimized one functional's value against a different
        functional's gradient."""
        from .tile_sensitivity import node_to_tile_map, tile_index_to_dims_key
        g = self.gradient(rho_mpa).numpy()
        elements = {e: [self.ids[i] for i in self.conn[k]]
                    for k, e in enumerate(self.eids)}
        n2t = node_to_tile_map(elements, elem_to_tile)
        out = {k: 0.0 for k in tile_keys}
        moved = {k: 0 for k in tile_keys}
        for nid, dx in dX_dt.items():
            t = n2t.get(nid)
            if t is None:
                continue
            gi = g[self.index[nid]]
            key = tile_index_to_dims_key(t, NX, NY, NZ, tile_keys)
            out[key] += float(gi[0] * dx[0] + gi[1] * dx[1] + gi[2] * dx[2])
            if max(abs(v) for v in dx) > 1e-9:
                moved[key] += 1

        # A tile none of whose nodes MOVE with its own design variable gets a
        # contraction of exactly 0.0 -- not a small sensitivity, a structurally
        # absent one. That is not a rounding artifact and it must never be
        # returned quietly: OC reads dG=0 as B = -dG/(lambda dV) = 0 and sets
        # x_new = x * B**eta = 0, so the tile is clipped to its FLOOR on the
        # very next generation. hollow_cube shipped exactly this trap --
        # model_lin.c clamps at HOLLOW_CUBE_MAX_THICKNESS 0.48 while dims.json
        # allowed 0.49, so any tile OC pushed to the cap (the support corners
        # first, having the strongest gradient) froze geometrically and then
        # collapsed to 0.01. Fail loudly instead.
        dead = [k for k, c in moved.items() if c == 0]
        if dead:
            raise AssertionError(
                f"{len(dead)} tiles have no node that moves with their own "
                f"thickness (e.g. {dead[:6]}) -- their sensitivity is "
                "structurally zero, not small. Check the design variable "
                "bounds against the CAD model's own clamps.")
        return out

    def assert_tie_is_fixed(self, dX_dt, tol=1e-9):
        """The rigid tie is held constant through the derivative; that is only
        exact if no tied node moves with any design variable."""
        bad = [self.ids[i] for i in self.slave
               if max(abs(v) for v in dX_dt.get(self.ids[i], (0, 0, 0))) > tol]
        if bad:
            raise AssertionError(
                f"{len(bad)} rigid-tied nodes have nonzero dX/dt (e.g. {bad[:5]}); "
                "the tie's own coordinate dependence would have to be differentiated")


class _SPDFactor:
    """Factorization of the reduced stiffness, which is symmetric positive
    definite (supports eliminated, load through the rigid tie). MKL PARDISO's
    Cholesky when pypardiso is installed, SciPy's SuperLU otherwise.

    SuperLU is a general unsymmetric LU and was half of every label: on
    hollow_cube's 145k-DOF K, 26 s of factorization against PARDISO's 1.0 s on
    one thread. The two solutions agree to 1e-8 relative, which is inside
    SuperLU's own residual (5e-9), so this changes the speed and not the
    labels. MKL runs sequential unless MKL_THREADING_LAYER is already set:
    its OpenMP runtime would otherwise share the process with torch's, and the
    labellers are one process per core anyway."""

    _local = threading.local()   # one PARDISO handle per thread, reused across labels

    @classmethod
    def _pardiso(cls):
        """This thread's PARDISO handle, or None for SuperLU
        (FEM_ADJOINT_SOLVER=superlu, or pypardiso not installed)."""
        if os.environ.get("FEM_ADJOINT_SOLVER", "pardiso") == "superlu":
            return None
        if getattr(cls._local, "solver", None) is None:
            os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
            if "PYPARDISO_MKL_RT" not in os.environ:
                # pypardiso's fallback when ctypes can't find mkl_rt is a
                # recursive glob of sys.prefix: 4.8 s per solver in this env
                import glob, sys
                hit = sorted(glob.glob(os.path.join(sys.prefix, "lib", "libmkl_rt.so*")))
                if hit:
                    os.environ["PYPARDISO_MKL_RT"] = hit[0]
            try:
                from pypardiso import PyPardisoSolver
            except ImportError:
                return None
            cls._local.solver = PyPardisoSolver(mtype=2)   # real symmetric positive definite
        return cls._local.solver

    def __init__(self, K):
        self._ps = self._pardiso()
        if self._ps is None:
            self._lu = spla.splu(K.tocsc())
            return
        self._A = sp.triu(K, format="csr")       # PARDISO reads the upper triangle
        self._A.sort_indices()
        self._ps.factorize(self._A)
        self._ps._owner = weakref.ref(self)

    def solve(self, b):
        if self._ps is None:
            return self._lu.solve(b)
        if self._ps._owner() is not self:
            # another ShapeAdjoint factorized on the shared handle since
            # (pypardiso would detect it and refactorize; be explicit)
            self._ps.factorize(self._A)
            self._ps._owner = weakref.ref(self)
        return self._ps.solve(self._A, np.asarray(b, dtype=np.float64))

    def __del__(self):
        # PARDISO's factor lives in MKL's memory, not Python's: release it
        # (phase 0) once this factorization is dropped, unless a newer one has
        # already replaced it on the shared handle
        ps = getattr(self, "_ps", None)
        if ps is not None and getattr(ps, "_owner", lambda: None)() in (self, None):
            try:
                ps.free_memory(everything=False)
            except Exception:
                pass


def _vm(S):
    """von Mises from the Voigt stress tensor (..., 6) in CalculiX's component
    order -- the same expression utils.frd_parser applies to ccx's own output."""
    sxx, syy, szz, sxy, sxz, syz = (S[..., i] for i in range(6))
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                             + 6 * (sxy ** 2 + syz ** 2 + sxz ** 2)) + 1e-30)
