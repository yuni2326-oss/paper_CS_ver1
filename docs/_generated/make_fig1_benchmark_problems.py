"""
Fig. 1 — The four dimensionless benchmark problems (schematic).
Every displayed mode shape is computed from the governing equations:
  P1: analytic clamped-free Euler-Bernoulli mode 3
  P2: Rayleigh-Ritz Kirchhoff annulus (clamped inner, free outer), m=2, n=0
  P3: two-segment beam + rotational spring at x_c=0.2 (transfer determinant), khat=1
  P4: Q1 plane-stress FEM on the L-shape, fully clamped boundary, mode 1
Colors: normalized modal displacement (dimensionless). Style: colored field
blending into plain gray geometry.

**이 그림은 개요도다.** 계산은 하지만 논문의 어떤 수치도 여기서 읽지 않는다 — 표시된
모드형은 본문에 적힌 지배방정식에서 직접 풀며, 진폭은 보이기 위해 과장했다. 렌더러가
아니라 여기(`docs/_generated/`)에 두는 이유는 `render/`가 "계산하지 않는다"는 규약을
지키기 위해서다. 산출물은 `docs/_generated/fig/paper2/fig1_benchmark_problems.png`.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import brentq
from scipy.linalg import eigh
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CMAP = cm.turbo
GRAY = np.array([0.72, 0.73, 0.78, 1.0])
BG = 'white'

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def blendc(field01, mix):
    """mix: 0 -> gray, 1 -> colormap(field01); shapes broadcast."""
    c = CMAP(field01)
    m = mix[..., None]
    return (1 - m) * GRAY + m * c

# ---------------------------------------------------------------- P1 mode 3
def p1_mode(x, beta):
    s = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    return (np.cosh(beta * x) - np.cos(beta * x)
            - s * (np.sinh(beta * x) - np.sin(beta * x)))

BETA3 = brentq(lambda b: np.cos(b) * np.cosh(b) + 1, 7.5, 8.2)  # ~7.8548

# ---------------------------------------------------------------- P3 (spring)
XC, KHAT = 0.2, 1.0

def p3_det(beta, khat=KHAT, xc=XC):
    b = beta
    def seg(x):
        return np.array([np.cos(b*x), np.sin(b*x), np.cosh(b*x), np.sinh(b*x)])
    def d1(x):
        return b*np.array([-np.sin(b*x), np.cos(b*x), np.sinh(b*x), np.cosh(b*x)])
    def d2(x):
        return b*b*np.array([-np.cos(b*x), -np.sin(b*x), np.cosh(b*x), np.sinh(b*x)])
    def d3(x):
        return b**3*np.array([np.sin(b*x), -np.cos(b*x), np.sinh(b*x), np.cosh(b*x)])
    # w1 basis: cosh-cos, sinh-sin  (clamped at 0)
    w1  = lambda x: np.array([np.cosh(b*x)-np.cos(b*x),  np.sinh(b*x)-np.sin(b*x)])
    w1p = lambda x: b*np.array([np.sinh(b*x)+np.sin(b*x), np.cosh(b*x)-np.cos(b*x)])
    w1pp= lambda x: b*b*np.array([np.cosh(b*x)+np.cos(b*x), np.sinh(b*x)+np.sin(b*x)])
    w1ppp=lambda x: b**3*np.array([np.sinh(b*x)-np.sin(b*x), np.cosh(b*x)+np.cos(b*x)])
    M = np.zeros((6, 6))
    M[0, :2], M[0, 2:] = w1(xc), -seg(xc)                       # displacement
    M[1, :2], M[1, 2:] = w1pp(xc), -d2(xc)                      # moment cont.
    M[2, :2], M[2, 2:] = w1ppp(xc), -d3(xc)                     # shear cont.
    M[3, :2] = w1pp(xc) + khat * w1p(xc)                        # M = khat*[u']
    M[3, 2:] = -khat * d1(xc)
    M[4, 2:] = d2(1.0)                                          # free end
    M[5, 2:] = d3(1.0)
    return np.linalg.det(M)

def p3_roots(n=3):
    bs = np.linspace(0.3, 14, 4000)
    vals = np.array([p3_det(b) for b in bs])
    roots = []
    for i in range(len(bs) - 1):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            roots.append(brentq(p3_det, bs[i], bs[i+1]))
            if len(roots) == n:
                break
    return roots

def p3_mode(x, beta, khat=KHAT, xc=XC):
    b = beta
    def rowmat():
        # solve for nullspace of the 6x6
        M = np.zeros((6, 6))
        w1  = lambda t: np.array([np.cosh(b*t)-np.cos(b*t),  np.sinh(b*t)-np.sin(b*t)])
        w1p = lambda t: b*np.array([np.sinh(b*t)+np.sin(b*t), np.cosh(b*t)-np.cos(b*t)])
        w1pp= lambda t: b*b*np.array([np.cosh(b*t)+np.cos(b*t), np.sinh(b*t)+np.sin(b*t)])
        w1ppp=lambda t: b**3*np.array([np.sinh(b*t)-np.sin(b*t), np.cosh(b*t)+np.cos(b*t)])
        seg = lambda t: np.array([np.cos(b*t), np.sin(b*t), np.cosh(b*t), np.sinh(b*t)])
        d1  = lambda t: b*np.array([-np.sin(b*t), np.cos(b*t), np.sinh(b*t), np.cosh(b*t)])
        d2  = lambda t: b*b*np.array([-np.cos(b*t), -np.sin(b*t), np.cosh(b*t), np.sinh(b*t)])
        d3  = lambda t: b**3*np.array([np.sin(b*t), -np.cos(b*t), np.sinh(b*t), np.cosh(b*t)])
        M[0, :2], M[0, 2:] = w1(xc), -seg(xc)
        M[1, :2], M[1, 2:] = w1pp(xc), -d2(xc)
        M[2, :2], M[2, 2:] = w1ppp(xc), -d3(xc)
        M[3, :2] = w1pp(xc) + khat * w1p(xc)
        M[3, 2:] = -khat * d1(xc)
        M[4, 2:] = d2(1.0)
        M[5, 2:] = d3(1.0)
        return M
    _, _, Vt = np.linalg.svd(rowmat())
    c = Vt[-1]
    A, B = c[0], c[1]
    a2, b2, c2, d2c = c[2:]
    w = np.where(
        x <= xc,
        A*(np.cosh(b*x)-np.cos(b*x)) + B*(np.sinh(b*x)-np.sin(b*x)),
        a2*np.cos(b*x) + b2*np.sin(b*x) + c2*np.cosh(b*x) + d2c*np.sinh(b*x))
    return w

# ---------------------------------------------------------------- P2 (annulus)
def p2_mode(m=2, a=0.42, b=1.0, nu=0.29, nbasis=9):
    r = np.linspace(a, b, 3000)
    def basis(j):
        # R = (r-a)^2 * cos(j*pi*xi), xi=(r-a)/(b-a)  (clamped inner edge)
        L = b - a
        f, f1, f2 = (r-a)**2, 2*(r-a), 2.0
        k = j*np.pi/L
        g  = np.cos(k*(r-a))
        g1 = -k*np.sin(k*(r-a))
        g2 = -k*k*np.cos(k*(r-a))
        return f*g, f1*g + f*g1, f2*g + 2*f1*g1 + f*g2
    N = nbasis
    Rs, R1s, R2s = [], [], []
    for j in range(N):
        R, R1, R2 = basis(j)
        Rs.append(R); R1s.append(R1); R2s.append(R2)
    U = np.zeros((N, N)); Mm = np.zeros((N, N))
    for i in range(N):
        for k in range(i, N):
            Li = R2s[i] + R1s[i]/r - m*m*Rs[i]/r**2
            Lk = R2s[k] + R1s[k]/r - m*m*Rs[k]/r**2
            gauss_i_k = (R2s[i]*(R1s[k]/r - m*m*Rs[k]/r**2)
                         + R2s[k]*(R1s[i]/r - m*m*Rs[i]/r**2)) / 2 \
                        - m*m*((Rs[i]/r)*0 + (R1s[i]/r - Rs[i]/r**2) *
                               (R1s[k]/r - Rs[k]/r**2))
            u = np.trapezoid((Li*Lk - 2*(1-nu)*gauss_i_k) * r, r)
            mm = np.trapezoid(Rs[i]*Rs[k]*r, r)
            U[i, k] = U[k, i] = u
            Mm[i, k] = Mm[k, i] = mm
    w, V = eigh(U, Mm)
    c = V[:, 0]                       # lowest radial order for this m
    Rfun = sum(c[j]*Rs[j] for j in range(N))
    Rfun = Rfun / np.max(np.abs(Rfun))
    return r, Rfun

# ---------------------------------------------------------------- P4 (L-shape FEM)
def p4_mode1(h=1/16, nu=0.29):
    # L-shape: [0,2]x[0,1] U [0,1]x[1,2]; fully clamped boundary; plane stress E=rho=1
    nx = int(round(2/h)) + 1
    coords, index = {}, {}
    def inside(i, j):
        x, y = i*h, j*h
        return (x <= 2+1e-9 and y <= 1+1e-9) or (x <= 1+1e-9 and y <= 2+1e-9)
    nid = 0
    for j in range(nx):
        for i in range(nx):
            if inside(i, j):
                index[(i, j)] = nid
                coords[nid] = (i*h, j*h)
                nid += 1
    nn = nid
    # element stiffness/mass (Q1, plane stress) via 2x2 Gauss
    E = 1.0
    C = E/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    gp = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
          (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    def shp(xi, eta):
        N = 0.25*np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                           (1+xi)*(1+eta), (1-xi)*(1+eta)])
        dN = 0.25*np.array([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
                            [(1+eta), (1+xi)], [-(1+eta), (1-xi)]])
        return N, dN
    Ke = np.zeros((8, 8)); Me = np.zeros((8, 8))
    J = h/2
    for xi, eta in gp:
        N, dN = shp(xi, eta)
        dNdx = dN / J
        Bm = np.zeros((3, 8))
        Bm[0, 0::2] = dNdx[:, 0]
        Bm[1, 1::2] = dNdx[:, 1]
        Bm[2, 0::2] = dNdx[:, 1]
        Bm[2, 1::2] = dNdx[:, 0]
        Ke += Bm.T @ C @ Bm * J*J
        Nm = np.zeros((2, 8))
        Nm[0, 0::2] = N; Nm[1, 1::2] = N
        Me += Nm.T @ Nm * J*J
    rows, cols, kv, mv = [], [], [], []
    elems = []
    for j in range(nx-1):
        for i in range(nx-1):
            ns = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
            if all(n in index for n in ns):
                # cell fully inside?
                xc_, yc_ = (i+0.5)*h, (j+0.5)*h
                if (xc_ <= 2 and yc_ <= 1) or (xc_ <= 1 and yc_ <= 2):
                    dofs = []
                    for n in ns:
                        dofs += [2*index[n], 2*index[n]+1]
                    elems.append([index[n] for n in ns])
                    for aI in range(8):
                        for bI in range(8):
                            rows.append(dofs[aI]); cols.append(dofs[bI])
                            kv.append(Ke[aI, bI]); mv.append(Me[aI, bI])
    K = sp.csr_matrix((kv, (rows, cols)), shape=(2*nn, 2*nn))
    M = sp.csr_matrix((mv, (rows, cols)), shape=(2*nn, 2*nn))
    # clamp entire boundary
    free = []
    for (i, j), n in index.items():
        x, y = i*h, j*h
        on_bnd = (
            abs(y) < 1e-9 or abs(x) < 1e-9 or
            (abs(x-2) < 1e-9 and y <= 1+1e-9) or
            (abs(y-2) < 1e-9 and x <= 1+1e-9) or
            (abs(y-1) < 1e-9 and x >= 1-1e-9) or
            (abs(x-1) < 1e-9 and y >= 1-1e-9))
        if not on_bnd:
            free += [2*n, 2*n+1]
    free = np.array(free)
    Kf = K[free][:, free]; Mf = M[free][:, free]
    vals, vecs = spla.eigsh(Kf, k=1, M=Mf, sigma=0, which='LM')
    u = np.zeros(2*nn); u[free] = vecs[:, 0]
    ux, uy = u[0::2], u[1::2]
    mag = np.sqrt(ux**2 + uy**2); mag /= mag.max()
    s = np.sign(ux[np.argmax(mag)]) or 1.0
    return coords, elems, s*ux, s*uy, mag

# ================================================================ figure
fig = plt.figure(figsize=(13.2, 7.6), facecolor=BG)
gs = fig.add_gridspec(2, 2, left=0.095, right=0.99, top=0.94, bottom=0.02,
                      wspace=0.0, hspace=0.02)

def clean(ax, elev, azim, box=None):
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    if box:
        ax.set_box_aspect(box)

AMP, HW = 0.20, 0.095
yw = np.array([-HW, HW])

def beam_panel(ax, w, title, subtitle, spring=None):
    x = np.linspace(0, 1, 400)
    Xb, Yb = np.meshgrid(x, yw, indexing='ij')
    Z = np.repeat((AMP*w)[:, None], 2, axis=1)
    F = np.repeat(np.abs(w)[:, None], 2, axis=1)
    mix = smoothstep((x - 0.05) / 0.11)
    Mx = np.repeat(mix[:, None], 2, axis=1)
    fc = blendc(F[:-1, :-1], Mx[:-1, :-1])
    ax.plot_surface(Xb, Yb, Z, facecolors=fc, rstride=3, cstride=1,
                    edgecolor=(0, 0, 0, 0.10), linewidth=0.15, shade=False)
    ax.plot_surface(Xb, Yb, np.zeros_like(Z), color=(0.62, 0.63, 0.68, 0.30),
                    edgecolor='none', shade=False)
    ax.add_collection3d(Poly3DCollection(
        [[(-0.012, -0.16, -0.16), (-0.012, 0.16, -0.16),
          (-0.012, 0.16, 0.16), (-0.012, -0.16, 0.16)]],
        facecolor=(0.45, 0.46, 0.5, 1), edgecolor='k', linewidth=0.4))
    ax.text2D(0.02, 0.95, title, transform=ax.transAxes, fontsize=11, weight='bold')
    ax.text2D(0.05, 0.875, subtitle, transform=ax.transAxes, fontsize=8.8)
    ax.text(0.0, 0, -0.24, 'clamped', fontsize=8, ha='center')
    if spring:
        xc, wxc = spring
        zc = AMP*wxc
        tt = np.linspace(0, 6*np.pi, 200)
        ax.plot(xc + 0.028*np.sin(tt), np.zeros_like(tt),
                zc + 0.055 + 0.10*tt/(6*np.pi), color='crimson', lw=1.6)
        ax.plot([xc, xc], [0, 0], [zc, zc+0.055], color='crimson', lw=1.2)
        ax.plot([xc, xc], [0, 0], [-0.20, zc], color='k', lw=0.7, ls=(0, (3, 3)))
        ax.text(xc, 0, zc+0.20, r'$\hat{k}$', color='crimson', fontsize=11, ha='center')
        ax.text(xc, 0, -0.27, r"slope jump $[\![u^\prime]\!]$", fontsize=8, ha='center')
    clean(ax, 20, -80, box=(2.7, 0.8, 0.95))
    ax.set_xlim(0, 1); ax.set_ylim(-0.34, 0.34); ax.set_zlim(-0.27, 0.27)

# ---------------- (a) P1
ax = fig.add_subplot(gs[0, 0], projection='3d', facecolor=BG)
x = np.linspace(0, 1, 400)
w1_ = p1_mode(x, BETA3); w1_ /= np.max(np.abs(w1_))
beam_panel(ax, w1_, '(a)  P1 — clamped–free Euler–Bernoulli beam',
           r'analytic reference, modes 1–10   (shown: mode 3, $\Lambda=(\beta L)^4$)')
ax.text(1.02, 0, AMP*w1_[-1], 'free', fontsize=8, ha='left')

# ---------------- (b) P2
ax = fig.add_subplot(gs[0, 1], projection='3d', facecolor=BG)
a_r = 0.42
r, R = p2_mode()
rg = np.linspace(a_r, 1.0, 60)
tg = np.linspace(0, 2*np.pi, 241)
Rg = np.interp(rg, r, R)
RR, TT = np.meshgrid(rg, tg, indexing='ij')
W = Rg[:, None] * np.cos(2*TT)
Xa = RR*np.cos(TT); Ya = RR*np.sin(TT)
mix = smoothstep((np.pi*1.05 - TT) / 0.55)
Za = 0.22 * W * mix
F = np.abs(W)
fc = blendc(F[:-1, :-1], mix[:-1, :-1])
srf = ax.plot_surface(Xa, Ya, Za, facecolors=fc, rstride=2, cstride=3,
                      linewidth=0.15, shade=False)
srf.set_edgecolor((0, 0, 0, 0.09))
ax.plot(a_r*np.cos(tg), a_r*np.sin(tg), 0*tg, color='k', lw=1.4)
ax.text2D(0.02, 0.95, '(b)  P2 — annular Kirchhoff plate (reference only)',
          transform=ax.transAxes, fontsize=11, weight='bold')
ax.text2D(0.05, 0.875, r'clamped inner / free outer, $a/b=0.42$, $\nu=0.29$;'
          r'  degenerate $\cos m\theta/\sin m\theta$ pairs   (shown: $m=2$, $n=0$)',
          transform=ax.transAxes, fontsize=8.8)
ax.text(0.06, -0.08, 0.30, 'clamped\ninner edge', fontsize=8, ha='center')
ax.text(-1.0, -0.55, -0.28, 'free outer edge', fontsize=8)
clean(ax, 30, -63, box=(1.5, 1.5, 0.55))

# ---------------- (c) P3
ax = fig.add_subplot(gs[1, 0], projection='3d', facecolor=BG)
roots = p3_roots(3)
b2 = roots[1]
w2_ = p3_mode(x, b2); w2_ /= np.max(np.abs(w2_))
beam_panel(ax, w2_,
           '(c)  P3 — beam with zero-width rotational spring',
           r'$x_c/L=0.2$, $\hat{k}=k_\theta L/EI\in\{1,10,100,1000\}$;'
           r'  $[\![u]\!]=0$, $[\![u^\prime]\!]$ free   (shown: $\hat{k}=1$, mode 2)',
           spring=(XC, float(np.interp(XC, x, w2_))))

# ---------------- (d) P4
ax = fig.add_subplot(gs[1, 1], projection='3d', facecolor=BG)
coords, elems, ux, uy, mag = p4_mode1()
nid = np.array(sorted(coords))
XY = np.array([coords[n] for n in nid])
thk = 0.14
scale = 0.14
# node-wise blend and deformation (watertight top face)
mixn = smoothstep((1.55 - (XY[:, 0] + XY[:, 1])) / 0.5)
Xd = XY[:, 0] + scale * mixn * ux
Yd = XY[:, 1] + scale * mixn * uy
polys, cols_, edges_ = [], [], []
for e in elems:
    f = mag[e].mean()
    mv = mixn[e].mean()
    polys.append([(Xd[n], Yd[n], thk) for n in e])
    cols_.append(blendc(np.array(f), np.array(mv)))
    edges_.append((0, 0, 0, 0.13 * mv))
top = Poly3DCollection(polys, facecolors=cols_, edgecolors=edges_, linewidths=0.15)
ax.add_collection3d(top)
outline = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]
walls = []
for p1_, p2_ in zip(outline[:-1], outline[1:]):
    walls.append([(p1_[0], p1_[1], 0), (p2_[0], p2_[1], 0),
                  (p2_[0], p2_[1], thk), (p1_[0], p1_[1], thk)])
ax.add_collection3d(Poly3DCollection(walls, facecolor=(0.60, 0.61, 0.66, 1),
                                     edgecolor=(0.25, 0.25, 0.28, 0.7), linewidth=0.4))
ax.text2D(0.02, 0.95, '(d)  P4 — L-shaped plane-stress eigenproblem',
          transform=ax.transAxes, fontsize=11, weight='bold')
ax.text2D(0.05, 0.875, r'unit arms, $\nu=0.29$, clamped boundary, '
          r'$\Omega^2=\omega^2\rho L^2/E$   (shown: mode 1)',
          transform=ax.transAxes, fontsize=8.8)
ax.text(1.30, 1.42, thk+0.30, 're-entrant\ncorner', fontsize=8, ha='center')
ax.plot([1.28, 1.03], [1.38, 1.03], [thk+0.24, thk+0.01], color='k', lw=0.8)
clean(ax, 42, -108, box=(1.55, 1.55, 0.40))
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_zlim(0, 0.55)

# ---------------- shared colorbar
sm = cm.ScalarMappable(cmap=CMAP); sm.set_array([0, 1])
cax = fig.add_axes([0.040, 0.30, 0.014, 0.40])
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([0, 0.5, 1])
cax.set_title('NORMALIZED\nMODAL\nDISPLACEMENT\n[–]', fontsize=7.2, loc='left', pad=10)

OUTDIR = os.path.join('docs', '_generated', 'fig', 'paper2')
os.makedirs(OUTDIR, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUTDIR, f'fig1_benchmark_problems.{ext}'),
                dpi=300, facecolor=BG, bbox_inches='tight')
print('P3 roots (khat=1):', [round(r_, 4) for r_ in p3_roots(3)], ' P1 beta3:', round(BETA3, 4))
print('saved')
