"""
Individual Architecture Diagrams - 6 Publication-Quality Figures (v2 - LaTeX math)
====================================================================================
Each model gets its own large, clean diagram with proper LaTeX rendering showing
mathematical structure, forward-pass equations, loss functions, and key
initialisation / hyperparameter details.

Verified against source code: 2026-04-18.

Outputs (saved to PLOTS/):
  13A_arch_blackbox.png       Model A - Black-Box FNN
  13B_arch_physreg.png        Model B - Physics-Regularized FNN
  13C_arch_residual.png       Model C - Residual Correction FNN
  13D_arch_lagrangian.png     Model D - Lagrangian Structured FNN
  13E1_arch_ecpinn.png        Model E.1 - Equation-Constrained PINN
  13E2_arch_decomposed.png    Model E.2 - Decomposed Structured PINN
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -- LaTeX rendering ------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

PLOTS_DIR = Path(__file__).resolve().parent.parent


# -- Colours ---------------------------------------------------------------
C_IN   = dict(fc="#DBEAFE", ec="#1E40AF", tc="#1E3A8A")
C_NET  = dict(fc="#FED7AA", ec="#C2410C", tc="#7C2D12")
C_PHY  = dict(fc="#D1FAE5", ec="#065F46", tc="#064E3B")
C_CAL  = dict(fc="#FEF9C3", ec="#92400E", tc="#78350F")
C_OUT  = dict(fc="#EDE9FE", ec="#6D28D9", tc="#4C1D95")
C_OP   = dict(fc="#F3F4F6", ec="#374151", tc="#111827")
C_FRI  = dict(fc="#FCE7F3", ec="#9D174D", tc="#831843")
C_SPD  = dict(fc="#E0F2FE", ec="#0369A1", tc="#0C4A6E")
C_LOSS = dict(fc="#FEF3C7", ec="#B45309", tc="#78350F")


# -- Drawing primitives ---------------------------------------------------

def bx(ax, cx, cy, w, h, lines, cs, fs=9.0, bold_first=True, lh=1.0):
    """Rounded box with multi-line LaTeX text."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04", fc=cs["fc"], ec=cs["ec"], lw=1.4, zorder=3)
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    step = min(h / (n + 0.5) * lh, 0.24)
    y0 = cy + (n - 1) / 2 * step
    for i, ln in enumerate(lines):
        fsi = fs if (i == 0 and bold_first) else fs * 0.88
        fw = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(cx, y0 - i * step, ln,
                ha="center", va="center", fontsize=fsi,
                fontweight=fw, color=cs.get("tc", "#111"), zorder=4)


def circ(ax, cx, cy, r, label, cs=C_OP, fs=14):
    c = plt.Circle((cx, cy), r, fc=cs["fc"], ec=cs["ec"], lw=1.4, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=cs.get("tc", "#111"), zorder=4)


def ar(ax, x0, y0, x1, y1, col="#374151", lw=1.3, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.15",
                                color=col, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def note(ax, x, y, text, fs=7.5, c="#6B7280", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=c,
            fontstyle="italic", zorder=5)


def divider(ax, x0, x1, y, c="#D1D5DB", lw=0.9):
    ax.plot([x0, x1], [y, y], color=c, lw=lw, zorder=1, ls="--")


def header(ax, W, H, title, subtitle, col):
    rect = FancyBboxPatch((0.15, H - 0.80), W - 0.30, 0.65,
                           boxstyle="round,pad=0.04",
                           fc=col + "22", ec=col, lw=1.6, zorder=1)
    ax.add_patch(rect)
    ax.text(W/2, H - 0.44, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=col, zorder=4)
    ax.text(W/2, H - 0.68, subtitle, ha="center", va="center",
            fontsize=8.5, color="#374151", zorder=4)


def loss_band(ax, W, cy, h, text_lines, col="#B45309"):
    rect = FancyBboxPatch((0.3, cy - h/2), W - 0.6, h,
                           boxstyle="round,pad=0.04",
                           fc="#FEF3C7", ec=col, lw=1.4, zorder=3)
    ax.add_patch(rect)
    n = len(text_lines)
    step = min(h / (n + 0.4), 0.26)
    y0 = cy + (n - 1) / 2 * step
    for i, ln in enumerate(text_lines):
        ax.text(W/2, y0 - i * step, ln,
                ha="center", va="center",
                fontsize=9.0 if i == 0 else 8.0,
                fontweight="bold" if i == 0 else "normal",
                color="#78350F", zorder=4)


def setup(W, H):
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, name):
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =====================================================================
# Model A - Black-Box FNN
# =====================================================================
def draw_A():
    W, H = 14.5, 7.5
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model A --- Black-Box FNN (Baseline)",
           r"Unconstrained MLP $\mid$ No physics signal $\mid$ Pure data supervision",
           "#1E40AF")

    # -- Forward pass at y=5.3 --
    y = 5.3
    bx(ax, 1.3, y, 1.9, 1.0,
       [r"\textbf{Input}",
        r"$\mathbf{x} = [\tilde{q},\,\dot{\tilde{q}},\,\ddot{\tilde{q}}]$",
        r"$\in \mathbb{R}^{15}$"], C_IN, fs=9)
    ar(ax, 2.25, y, 2.85, y)

    # Three hidden blocks
    xs = [3.8, 5.85, 7.9]
    dims = [256, 512, 256]
    for lx, d in zip(xs, dims):
        bx(ax, lx, y, 1.85, 1.45,
           [r"$\textbf{Linear}(" + str(d) + r")$",
            r"$\mathrm{LayerNorm}$",
            r"$\mathrm{SiLU}:\; x \mapsto x\,\sigma(x)$",
            r"$\mathrm{Dropout}(p\!=\!0.1)$"], C_NET, fs=8.5)
        if lx < 7.9:
            ar(ax, lx + 0.925, y, lx + 1.125, y)
    ar(ax, 7.9 + 0.925, y, 8.85, y)

    # Output projection
    bx(ax, 9.85, y, 1.5, 0.95,
       [r"$\textbf{Linear}(5)$",
        r"no activation"], C_NET, fs=8.5)
    ar(ax, 10.6, y, 11.15, y)

    # Output
    bx(ax, 12.05, y, 1.7, 1.0,
       [r"\textbf{Output}",
        r"$\hat{\bm{\tau}} \in \mathbb{R}^{5}$"], C_OUT)

    # Annotations
    note(ax, 5.85, 6.45,
         r"Each hidden layer: $\;\mathrm{Linear} \;\to\; \mathrm{LayerNorm} \;\to\; \mathrm{SiLU} \;\to\; \mathrm{Dropout}$",
         fs=8.5, c="#374151")
    note(ax, 1.3, 4.35,
         r"Training: $\tilde{\mathbf{x}} \leftarrow \tilde{\mathbf{x}} + \bm{\epsilon}, \;\; \bm{\epsilon} \sim \mathcal{N}(\mathbf{0},\, 0.02^2\mathbf{I})$",
         fs=8)

    # Forward-pass equation
    ax.text(W/2, 3.65,
            r"$\hat{\bm{\tau}} = \mathbf{W}_4 \; \sigma_3 \!\circ\! \mathbf{W}_3 \; \sigma_2 \!\circ\! \mathbf{W}_2 \; \sigma_1 \!\circ\! \mathbf{W}_1 \, \mathbf{x}, \quad \sigma_k = \mathrm{Dropout}\!\circ\!\mathrm{SiLU}\!\circ\!\mathrm{LN}$",
            ha="center", va="center", fontsize=10.5, color="#1E3A8A", zorder=5)

    divider(ax, 0.3, W - 0.3, 3.0)

    # Init + params
    note(ax, W/2, 2.6,
         r"Weights: Xavier normal $\;\mathbf{W} \sim \mathcal{N}\!\big(0,\; 2/(n_{\mathrm{in}}+n_{\mathrm{out}})\big) \quad|\quad$ Biases: $\mathbf{b} = \mathbf{0}$",
         fs=8.5, c="#374151")

    loss_band(ax, W, 1.60, 1.30, [
        r"$\displaystyle\mathcal{L}_A \;=\; \frac{1}{BJ}\sum_{b,j} w_j \bigl(\hat{\tau}_{bj} - \tau_{bj}^{\mathrm{meas}}\bigr)^{\!2}$",
        r"$\mathbf{w} = [1.0,\; 2.5,\; 1.0,\; 1.0,\; 1.0]$ --- J1 (shoulder) upweighted $2.5\times$ $\quad|\quad$ Validation/test: $\mathbf{w} = \mathbf{1}$",
    ])

    save(fig, "13A_arch_blackbox.png")


# =====================================================================
# Model B - Physics-Regularized FNN
# =====================================================================
def draw_B():
    W, H = 15.5, 10.5
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model B --- Physics-Regularized FNN",
           r"Soft physics loss $\mid$ Affine calibration $\varphi$ $\mid$ EMA-normalised mixture",
           "#C2410C")

    # -- DATA PATH (y=8.5) --------------------------------
    y_d = 8.5
    bx(ax, 1.2, y_d, 1.8, 0.85,
       [r"\textbf{Input}", r"$\mathbf{x} \in \mathbb{R}^{15}$"], C_IN)
    ar(ax, 2.1, y_d, 2.65, y_d)
    bx(ax, 4.2, y_d, 2.8, 1.35,
       [r"\textbf{MLP} (identical to A)",
        r"$15 \to [256, 512, 256] \to 5$",
        r"$\mathrm{SiLU} \;\mid\; \mathrm{Dropout}(0.05)$",
        r"$\mathrm{LayerNorm}$ per layer"], C_NET, fs=8.5)
    ar(ax, 5.6, y_d, 6.2, y_d)
    bx(ax, 7.1, y_d, 1.7, 0.85,
       [r"$\hat{\bm{\tau}} \in \mathbb{R}^{5}$"], C_OUT)

    note(ax, 4.2, 9.55,
         r"Forward pass identical to Model A --- physics enters \textit{only} through the training loss",
         fs=8.5, c="#374151")

    # -- PHYSICS PATH (y=5.8) -----------------------------
    y_p = 5.8
    bx(ax, 1.2, y_p, 2.0, 1.0,
       [r"\textbf{RNEA precomputed}",
        r"$\tau_g,\, \tau_M,\, \tau_C,\, \tau_f$",
        r"$\in \mathbb{R}^{20}$"], C_PHY)
    ar(ax, 2.2, y_p, 2.85, y_p)

    bx(ax, 3.9, y_p, 1.85, 0.90,
       [r"$\displaystyle\sum$",
        r"$\tau_{\mathrm{nom}} = \tau_g + \tau_M + \tau_C + \tau_f$"], C_OP, fs=8.5)
    ar(ax, 4.825, y_p, 5.45, y_p)

    bx(ax, 7.1, y_p, 3.0, 1.55,
       [r"\textbf{Affine calibration} $\varphi$",
        r"$s_j = \mathrm{softplus}(z_j) + 10^{-5}$",
        r"$\varphi(\tau) = \mathrm{diag}(\mathbf{s})\,\tau_{\mathrm{nom}} + \mathbf{b}$",
        r"Init: $\mathbf{s} = \mathbf{1},\; \mathbf{b} = \mathbf{0}$ (identity)",
        r"$\eta_\varphi = \eta$ (same as main net)"], C_CAL, fs=8.5)
    ar(ax, 8.6, y_p, 9.25, y_p)
    bx(ax, 10.15, y_p, 1.7, 0.85,
       [r"$\tau_{\mathrm{phys}}^{\mathrm{cal}}$",
        r"$\in \mathbb{R}^{5}$"], C_PHY)

    # Equations in centre
    ax.text(W/2, 4.6,
            r"$\mathcal{L}_{\mathrm{phys}}^B = \mathrm{MSE}\!\left(\hat{\bm{\tau}},\; \varphi(\tau_{\mathrm{nom}})\right) \;+\; 0.01\,\mathcal{L}_{\mathrm{calib}}$",
            ha="center", va="center", fontsize=10.5, color="#7C2D12", zorder=5)
    ax.text(W/2, 4.1,
            r"$\mathcal{L}_{\mathrm{calib}} = \frac{1}{J}\sum_{j=1}^{J}\!\left[(s_j - 1)^2 + b_j^2\right]$",
            ha="center", va="center", fontsize=10, color="#374151", zorder=5)

    # Loss arrows
    ar(ax, 7.1, y_d - 0.425, 7.1, 5.12, col="#6D28D9", lw=1.2)
    ar(ax, 10.15, y_p - 0.425, 10.15, 5.12, col="#065F46", lw=1.2)

    divider(ax, 0.3, W - 0.3, 3.5)

    # Schedule equations
    ax.text(W/2, 2.95,
            r"$w_p(e) = \alpha \cdot \min\!\left(1,\;\frac{e}{e_w}\right), \quad e_w = \lfloor 0.03\,E \rfloor, \quad \alpha = 0.10, \quad w_d = 1 - w_p$",
            ha="center", va="center", fontsize=10, color="#374151", zorder=5)
    note(ax, W/2, 2.55,
         r"$w_p$: linear ramp $0 \to \alpha$ over first 3\% of epochs, then \textbf{constant} at $\alpha$ (no decay)",
         fs=8.5, c="#6B7280")

    loss_band(ax, W, 1.55, 1.30, [
        r"$\displaystyle\mathcal{L}_B \;=\; w_d\,\mathcal{L}_{\mathrm{data}} \;+\; w_p \cdot \kappa \cdot \mathcal{L}_{\mathrm{phys}}^B$",
        r"$\kappa = \hat{\mu}_d / \hat{\mu}_p, \quad \hat{\mu} \leftarrow 0.98\,\hat{\mu} + 0.02\,\ell \quad\;$ (EMA $\beta = 0.98$ balances gradient magnitudes)",
    ])

    save(fig, "13B_arch_physreg.png")


# =====================================================================
# Model C - Residual Correction FNN
# =====================================================================
def draw_C():
    W, H = 16.0, 10.0
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model C --- Residual Correction FNN",
           r"Analytical torque as explicit input $\mid$ Learned scale $\alpha$ and additive residual $\delta$",
           "#059669")

    # -- Inputs --------------------------------------------
    bx(ax, 1.1, 7.6, 2.0, 0.95,
       [r"\textbf{Kinematics}",
        r"$\mathbf{x} = [\tilde{q},\,\dot{\tilde{q}},\,\ddot{\tilde{q}}] \in \mathbb{R}^{15}$"], C_IN)
    bx(ax, 1.1, 5.7, 2.0, 0.95,
       [r"\textbf{Physics input}",
        r"$\bm{\tau}_{\mathrm{phys}} = \sum_k \tau_k \in \mathbb{R}^{5}$"], C_PHY)

    # Concat
    bx(ax, 3.8, 6.65, 2.0, 1.0,
       [r"\textbf{Concat}",
        r"$\mathbb{R}^{15} \,\|\, \mathbb{R}^{5} \to \mathbb{R}^{20}$"], C_OP)
    ar(ax, 2.1, 7.6, 3.0, 6.85, rad=0.10)
    ar(ax, 2.1, 5.7, 3.0, 6.45, rad=-0.10)
    ar(ax, 4.8, 6.65, 5.45, 6.65)

    # Encoder
    bx(ax, 6.85, 6.65, 2.6, 1.50,
       [r"\textbf{Encoder MLP}",
        r"$20 \to [256, 512, 256]$",
        r"$\tanh \;\mid\; \mathrm{Dropout}(0.05)$",
        r"$\mathrm{LayerNorm}$ per layer",
        r"$\to \mathbf{h} \in \mathbb{R}^{256}$"], C_NET, fs=8.5)
    ar(ax, 8.15, 6.65, 8.85, 6.65)

    # Two heads
    bx(ax, 10.3, 7.9, 2.6, 1.35,
       [r"\textbf{Scale head} $\bm{\alpha}$",
        r"$\bm{\alpha} = \mathrm{softplus}(\mathbf{W}_\alpha\mathbf{h} + \mathbf{b}_\alpha) + 10^{-3}$",
        r"Init: $\mathbf{b}_\alpha = \ln(e^{0.5}\!-\!1) \approx -0.48$",
        r"$\Rightarrow\; \alpha \approx 0.5$ at $t\!=\!0$"], C_CAL, fs=8.2)
    bx(ax, 10.3, 5.4, 2.6, 1.35,
       [r"\textbf{Residual head} $\bm{\delta}$",
        r"$\bm{\delta} = \mathbf{W}_\delta\mathbf{h} + \mathbf{b}_\delta$",
        r"(linear, no activation)",
        r"Init: $\mathbf{W}_\delta = \mathbf{0},\; \mathbf{b}_\delta = \mathbf{0}$"], C_NET, fs=8.2)
    ar(ax, 8.85, 6.85, 9.2, 7.75, rad=0.18)
    ar(ax, 8.85, 6.45, 9.2, 5.55, rad=-0.18)

    # Output assembly
    bx(ax, 13.5, 6.65, 2.3, 1.10,
       [r"\textbf{Output}",
        r"$\hat{\bm{\tau}} = \bm{\alpha} \odot \bm{\tau}_{\mathrm{phys}} + \bm{\delta}$"], C_OUT, fs=9.5)
    ar(ax, 11.6, 7.9, 12.55, 6.95, rad=0.08)
    ar(ax, 11.6, 5.4, 12.55, 6.35, rad=-0.08)
    # tau_phys bypass
    ar(ax, 2.1, 5.7, 12.55, 6.65, col="#065F46", lw=1.0, rad=-0.06)

    # At-init note
    note(ax, 10.3, 8.95,
         r"At $t\!=\!0$: $\hat{\bm{\tau}} = 0.5 \cdot \bm{\tau}_{\mathrm{phys}} + \mathbf{0}$ (warm-start from analytical model, halved)",
         fs=8.5, c="#374151")

    # Forward-pass equation
    ax.text(W/2, 4.25,
            r"$\hat{\bm{\tau}} = \bigl[\mathrm{softplus}(\mathbf{W}_\alpha\,\mathrm{Enc}(\mathbf{x} \,\|\, \bm{\tau}_{\mathrm{phys}})) + 10^{-3}\bigr] \odot \bm{\tau}_{\mathrm{phys}} \;+\; \mathbf{W}_\delta\,\mathrm{Enc}(\mathbf{x} \,\|\, \bm{\tau}_{\mathrm{phys}})$",
            ha="center", va="center", fontsize=10, color="#064E3B", zorder=5)
    note(ax, W/2, 3.6,
         r"Model C is \textbf{not} in \texttt{PHYSICS\_WEIGHT\_MODELS} --- physics is structural (input), not a dynamically weighted loss",
         fs=8.5, c="#9D174D")

    divider(ax, 0.3, W - 0.3, 3.0)

    loss_band(ax, W, 1.88, 1.40, [
        r"$\displaystyle\mathcal{L}_C \;=\; \mathcal{L}_{\mathrm{data}} \;+\; 0.05 \cdot \frac{1}{J}\sum_{j=1}^{J}(\alpha_j - 1)^2$",
        r"$\alpha$-regulariser anchors scale near unity --- prevents $\alpha$ from absorbing all error",
    ])

    save(fig, "13C_arch_residual.png")


# =====================================================================
# Model D - Lagrangian Structured FNN
# =====================================================================
def draw_D():
    W, H = 16.5, 14.5
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model D --- Lagrangian Structured FNN",
           r"Four sub-networks from scratch $\mid$ SPD inertia (Cholesky) $\mid$ Dissipative friction by construction",
           "#7C3AED")

    y_rows = [12.2, 9.65, 7.3, 4.65]

    # - Row 1: Inertia -------------------------------------
    y = y_rows[0]
    bx(ax, 1.1, y, 1.6, 0.80,
       [r"\textbf{Input}", r"$\tilde{q} \in \mathbb{R}^{5}$"], C_IN)
    ar(ax, 1.9, y, 2.5, y)
    bx(ax, 3.55, y, 2.0, 1.35,
       [r"\textbf{M-net}",
        r"$5 \to [256, 512, 256] \to 15$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$",
        r"Diag.\ bias $= -2.0$"], C_NET, fs=8.5)
    ar(ax, 4.55, y, 5.3, y)
    bx(ax, 6.9, y, 3.0, 1.80,
       [r"\textbf{Cholesky} $\to$ \textbf{M}$(q)$ \textbf{SPD}",
        r"$L_{ii} = \mathrm{softplus}(\tilde{L}_{ii}) + \varepsilon_{\mathrm{SPD}}$",
        r"$L_{ij} = \tilde{L}_{ij} \quad (i > j)$",
        r"$\mathbf{M}(q) = \mathbf{L}\mathbf{L}^\top + \varepsilon_{\mathrm{SPD}}\mathbf{I}$",
        r"$\varepsilon_{\mathrm{SPD}} = 10^{-4}$"], C_SPD, fs=8.5)
    ar(ax, 8.4, y, 9.15, y)
    bx(ax, 10.15, y, 2.0, 0.95,
       [r"$\hat{\bm{\tau}}_M = \mathbf{M}(\tilde{q})\,\ddot{\tilde{q}}$",
        r"Inertia torque"], C_SPD, fs=9.0)

    note(ax, 6.9, y + 1.25,
         r"$\mathrm{softplus}(-2.0) \approx 0.126 \;\Rightarrow\; M_{ii} \approx 0.016$ at init (non-degenerate, small)",
         fs=8, c="#0369A1")

    # - Row 2: Coriolis ------------------------------------
    y = y_rows[1]
    bx(ax, 1.1, y, 1.6, 0.80,
       [r"\textbf{Input}", r"$[\tilde{q}, \dot{\tilde{q}}] \in \mathbb{R}^{10}$"], C_IN)
    ar(ax, 1.9, y, 2.5, y)
    bx(ax, 3.55, y, 2.0, 1.25,
       [r"\textbf{C-net} (from scratch)",
        r"$10 \to [256, 512, 256] \to 5$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.5)
    ar(ax, 4.55, y, 5.3, y)
    bx(ax, 6.3, y, 1.85, 0.80,
       [r"$\hat{\bm{\tau}}_C \in \mathbb{R}^{5}$",
        r"Coriolis + centrifugal"], C_PHY)
    note(ax, 3.55, y - 1.0,
         r"No RNEA prior --- $\hat{\tau}_C$ learned entirely from data",
         fs=8, c="#6B7280")

    # - Row 3: Gravity -------------------------------------
    y = y_rows[2]
    bx(ax, 1.1, y, 1.6, 0.80,
       [r"\textbf{Input}", r"$\tilde{q} \in \mathbb{R}^{5}$"], C_IN)
    ar(ax, 1.9, y, 2.5, y)
    bx(ax, 3.55, y, 2.0, 1.25,
       [r"\textbf{g-net} (from scratch)",
        r"$5 \to [256, 512, 256] \to 5$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.5)
    ar(ax, 4.55, y, 5.3, y)
    bx(ax, 6.3, y, 1.85, 0.80,
       [r"$\hat{\bm{\tau}}_g \in \mathbb{R}^{5}$",
        r"Gravity torque"], C_PHY)
    note(ax, 3.55, y - 1.0,
         r"No RNEA prior --- $\hat{\tau}_g$ learned entirely from data",
         fs=8, c="#6B7280")

    # - Row 4: Friction ------------------------------------
    y = y_rows[3]
    bx(ax, 1.1, y, 1.6, 0.80,
       [r"\textbf{Input}", r"$\dot{\tilde{q}} \in \mathbb{R}^{5}$"], C_IN)
    ar(ax, 1.9, y, 2.5, y)
    bx(ax, 3.55, y, 2.0, 1.15,
       [r"\textbf{f-net}",
        r"$5 \to [128, 128] \to 10$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.5)
    ar(ax, 4.55, y, 5.3, y)
    bx(ax, 7.2, y, 3.6, 1.55,
       [r"\textbf{Dissipative friction} (by construction)",
        r"$\hat{\bm{\tau}}_f = -\bigl[\mathrm{sp}(\mathbf{v})\odot\dot{\tilde{q}} + \mathrm{sp}(\mathbf{c})\odot\tanh(\dot{\tilde{q}}/0.04)\bigr]$",
        r"$\mathrm{sp}(\cdot) = \mathrm{softplus}(\cdot) \geq 0$",
        r"$\Rightarrow\; \hat{\tau}_{f,j}\,\dot{\tilde{q}}_j \leq 0 \;\;\forall\, j$ (dissipative)"], C_FRI, fs=8.5)
    note(ax, 3.55, y - 0.95,
         r"Init: $\mathbf{W}_{\mathrm{last}} \sim \mathcal{N}(0, 10^{-3}),\; \mathbf{b}_{\mathrm{last}} = \mathbf{0} \;\Rightarrow\; \hat{\tau}_f \approx \mathbf{0}$",
         fs=8, c="#6B7280")

    # - Summation ------------------------------------------
    x_sum = 12.5; y_sum = 8.5
    circ(ax, x_sum, y_sum, 0.42, r"$\bm{+}$", C_OP, fs=16)

    src_pairs = [
        (10.15 + 1.0, y_rows[0]),
        (6.3 + 0.925, y_rows[1]),
        (6.3 + 0.925, y_rows[2]),
        (7.2 + 1.8, y_rows[3]),
    ]
    for sx, sy in src_pairs:
        ar(ax, sx, sy, x_sum - 0.42, y_sum, col="#374151")

    ar(ax, x_sum + 0.42, y_sum, x_sum + 1.1, y_sum)
    bx(ax, 14.0, y_sum, 1.7, 0.95,
       [r"$\hat{\bm{\tau}} \in \mathbb{R}^{5}$",
        r"$= \hat{\tau}_M + \hat{\tau}_C + \hat{\tau}_g + \hat{\tau}_f$"], C_OUT)

    # Equation of motion
    ax.text(W/2, 3.55,
            r"$\hat{\bm{\tau}} = \mathbf{M}(\tilde{q})\,\ddot{\tilde{q}} \;+\; \hat{\bm{\tau}}_C(\tilde{q}, \dot{\tilde{q}}) \;+\; \hat{\bm{\tau}}_g(\tilde{q}) \;+\; \hat{\bm{\tau}}_f(\dot{\tilde{q}})$",
            ha="center", va="center", fontsize=11, color="#4C1D95", zorder=5)

    # SPD + fric notes
    ax.text(W/2, 2.85,
            r"$\mathcal{L}_{\mathrm{SPD}} = \bigl\langle[\max(0,\, \varepsilon_{\mathrm{SPD}} - \lambda_{\min}(\mathbf{M}))]^2\bigr\rangle, \quad \mathcal{L}_{\mathrm{fric}} = \bigl\langle\max\bigl(0,\, \hat{\bm{\tau}}_f \odot \dot{\tilde{q}}\bigr)\bigr\rangle$",
            ha="center", va="center", fontsize=9.5, color="#374151", zorder=5)
    note(ax, W/2, 2.4,
         r"Both penalties are safety nets --- rarely active due to architectural guarantees",
         fs=8, c="#6B7280")

    divider(ax, 0.3, W - 0.3, 2.0)

    loss_band(ax, W, 1.30, 0.95, [
        r"$\displaystyle\mathcal{L}_D \;=\; \mathcal{L}_{\mathrm{data}} \;+\; 0.01\,\mathcal{L}_{\mathrm{SPD}} \;+\; 0.01\,\mathcal{L}_{\mathrm{fric}}$",
        r"No physics weight schedule $\mid$ No LossNormaliser $\mid$ No nominal anchor --- all sub-networks from scratch",
    ])

    save(fig, "13D_arch_lagrangian.png")


# =====================================================================
# Model E.1 - Equation-Constrained PINN
# =====================================================================
def draw_E1():
    W, H = 16.5, 12.5
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model E.1 --- Equation-Constrained PINN",
           r"Equation residual loss $\mid$ Collocation at synthetic states $\mid$ LossNormaliser $\kappa$",
           "#DC2626")

    # -- DATA PATH (y=10.2) --------------------------------
    y_d = 10.2
    bx(ax, 1.2, y_d, 1.8, 0.85,
       [r"\textbf{Input}", r"$\mathbf{x} \in \mathbb{R}^{15}$"], C_IN)
    ar(ax, 2.1, y_d, 2.7, y_d)
    bx(ax, 4.3, y_d, 2.8, 1.35,
       [r"\textbf{MLP} (identical to A)",
        r"$15 \to [256, 512, 256] \to 5$",
        r"$\mathrm{SiLU} \;\mid\; \mathrm{Dropout}(0.05)$",
        r"$\mathrm{LayerNorm}$ per layer"], C_NET, fs=8.5)
    ar(ax, 5.7, y_d, 6.35, y_d)
    bx(ax, 7.25, y_d, 1.7, 0.85,
       [r"$\hat{\bm{\tau}} \in \mathbb{R}^{5}$"], C_OUT)

    # -- PHYSICS PATH (y=7.6) -----------------------------
    y_p = 7.6
    bx(ax, 1.2, y_p, 2.0, 1.0,
       [r"\textbf{RNEA precomputed}",
        r"$\tau_g, \tau_M, \tau_C, \tau_f$",
        r"$\in \mathbb{R}^{20}$"], C_PHY)
    ar(ax, 2.2, y_p, 2.85, y_p)
    bx(ax, 3.9, y_p, 1.85, 0.90,
       [r"$\displaystyle\sum$",
        r"$\tau_{\mathrm{nom}} \in \mathbb{R}^{5}$"], C_OP, fs=8.5)
    ar(ax, 4.825, y_p, 5.5, y_p)
    bx(ax, 7.25, y_p, 3.2, 1.55,
       [r"\textbf{Calibration} $\varphi$ (separate param group)",
        r"$\varphi(\tau) = \mathrm{diag}(\mathrm{sp}(\mathbf{z})\!+\!10^{-5})\,\tau + \mathbf{b}$",
        r"$\eta_\varphi = 0.1 \times \eta_{\mathrm{main}}$",
        r"$\lambda_{\mathrm{wd}}^\varphi = 0$ (no weight decay)",
        r"Init: $\mathbf{s} = \mathbf{1},\; \mathbf{b} = \mathbf{0}$"], C_CAL, fs=8.3)
    ar(ax, 8.85, y_p, 9.5, y_p)
    bx(ax, 10.3, y_p, 1.7, 0.85,
       [r"$\tau_{\mathrm{phys}}^{\mathrm{cal}}$",
        r"$\in \mathbb{R}^{5}$"], C_PHY)

    # Residual node
    bx(ax, 13.0, 8.9, 3.0, 1.35,
       [r"\textbf{Equation residual}",
        r"$\mathbf{r} = \hat{\bm{\tau}} - \varphi(\tau_{\mathrm{nom}})$",
        r"$\mathcal{L}_{\mathrm{phys}} = \|\mathbf{r}\|^2 + 0.01\,\mathcal{L}_{\mathrm{calib}}$"], C_LOSS, fs=9)
    ar(ax, 7.25 + 0.85, y_d, 12.35, 9.2, col="#6D28D9", lw=1.2)
    ar(ax, 10.3 + 0.85, y_p, 12.35, 8.6, col="#065F46", lw=1.2)

    # Calib reg note
    note(ax, 7.25, 6.45,
         r"$\mathcal{L}_{\mathrm{calib}} = \frac{1}{J}\sum_j[(s_j - 1)^2 + b_j^2]$ --- keeps $\varphi$ near identity",
         fs=8.5, c="#78350F")

    # -- COLLOCATION PATH (y=4.7) -------------------------
    y_c = 4.7
    bx(ax, 1.3, y_c, 2.5, 1.65,
       [r"\textbf{Collocation sampling}",
        r"$n_{\mathrm{col}} = 32$ per epoch",
        r"$q_j \sim \mathcal{U}[\mu_j \!-\! 3\sigma_j,\; \mu_j \!+\! 3\sigma_j]$",
        r"$\dot{q}_j \sim \mathcal{N}(\mu_{\dot{q}_j},\, \sigma_{\dot{q}_j}^2)$",
        r"$\ddot{q}_j \sim \mathcal{N}(\mu_{\ddot{q}_j},\, \sigma_{\ddot{q}_j}^2)$"], C_IN, fs=8.5)
    ar(ax, 2.55, y_c, 3.25, y_c)
    bx(ax, 4.4, y_c, 2.0, 1.05,
       [r"\textbf{RNEA + friction}",
        r"(Pinocchio, physical units)",
        r"$\to$ z-score normalise"], C_PHY, fs=8.5)
    ar(ax, 5.4, y_c, 6.1, y_c)
    bx(ax, 7.4, y_c, 2.3, 1.05,
       [r"\textbf{Calibration} $\varphi$ (shared)",
        r"same params as above"], C_CAL, fs=8.5)
    ar(ax, 8.55, y_c, 9.25, y_c)
    bx(ax, 10.5, y_c, 2.3, 1.05,
       [r"$\mathcal{L}_{\mathrm{col}} = \|\mathbf{r}_{\mathrm{col}}\|^2$",
        r"$\lambda_{\mathrm{col}} = 0.05$"], C_LOSS, fs=9)

    note(ax, 5.5, 3.45,
         r"Collocation extends physics supervision into $\mathcal{U}[\mu \pm 3\sigma]$ --- beyond training distribution",
         fs=8.5, c="#374151")

    divider(ax, 0.3, W - 0.3, 2.9)

    # Schedule equation
    ax.text(W/2, 2.45,
            r"$w_p(e) = 0.10 \cdot \min\!\left(1,\;\frac{e}{\lfloor 0.03\,E \rfloor}\right),\quad w_d = 1 - w_p,\quad \kappa = \hat{\mu}_d / \hat{\mu}_p \;\;(\beta = 0.98)$",
            ha="center", va="center", fontsize=9.5, color="#374151", zorder=5)

    loss_band(ax, W, 1.50, 1.15, [
        r"$\displaystyle\mathcal{L}_{E.1} \;=\; w_d\,\mathcal{L}_{\mathrm{data}} \;+\; w_p \cdot \kappa \cdot \mathcal{L}_{\mathrm{phys}} \;+\; \lambda_{\mathrm{col}}\,\mathcal{L}_{\mathrm{col}}$",
        r"Physics weight: ramp then CONSTANT $\mid$ LossNormaliser $\kappa$ active $\mid$ $\lambda_{\mathrm{col}} = 0.05$",
    ])

    save(fig, "13E1_arch_ecpinn.png")


# =====================================================================
# Model E.2 - Decomposed Structured PINN
# =====================================================================
def draw_E2():
    W, H = 17.5, 16.0
    fig, ax = setup(W, H)
    header(ax, W, H,
           r"Model E.2 --- Decomposed Structured PINN",
           r"Corrections on RNEA nominal $\mid$ SPD inertia $\mid$ Dissipative friction $\mid$ Occam regulariser",
           "#0D9488")

    y_M  = 13.8
    y_C  = 11.0
    y_g  = 8.2
    y_f  = 5.4

    # - Row 1: Inertia (from scratch) ---------------------
    y = y_M
    bx(ax, 1.1, y, 1.6, 0.80,
       [r"$\tilde{q} \in \mathbb{R}^{5}$"], C_IN)
    ar(ax, 1.9, y, 2.5, y)
    bx(ax, 3.55, y, 2.0, 1.25,
       [r"\textbf{M-net (SPD)}",
        r"$5 \to [256,512,256] \to 15$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.5)
    ar(ax, 4.55, y, 5.3, y)
    bx(ax, 6.95, y, 3.0, 1.50,
       [r"\textbf{Cholesky} $\to$ $\mathbf{M}(q)$ SPD",
        r"$L_{ii} = \mathrm{softplus}(\tilde{L}_{ii}) + 10^{-4}$",
        r"$\mathbf{M} = \mathbf{L}\mathbf{L}^\top + 10^{-4}\mathbf{I}$",
        r"diag bias $= -2.0$"], C_SPD, fs=8.5)
    ar(ax, 8.45, y, 9.2, y)
    bx(ax, 10.2, y, 2.0, 0.85,
       [r"$\hat{\bm{\tau}}_M = \mathbf{M}(\tilde{q})\,\ddot{\tilde{q}}$"], C_SPD, fs=9)

    note(ax, 6.95, y + 1.1,
         r"Inertia from scratch (same as Model D) --- no nominal RNEA prior for $\mathbf{M}$",
         fs=8, c="#6B7280")

    # - Row 2: Coriolis correction -------------------------
    y = y_C
    y_up = y + 0.55; y_dn = y - 0.55
    bx(ax, 1.1, y_up, 1.6, 0.75,
       [r"$[\tilde{q}, \dot{\tilde{q}}] \in \mathbb{R}^{10}$"], C_IN)
    bx(ax, 1.1, y_dn, 1.6, 0.75,
       [r"$\tau_C^{\mathrm{nom}}$ (RNEA)"], C_PHY)
    ar(ax, 1.9, y_up, 2.5, y_up)
    ar(ax, 1.9, y_dn, 2.5, y_dn)
    bx(ax, 3.55, y_up, 2.0, 1.05,
       [r"\textbf{dC-net} (correction)",
        r"$10 \to [256,512,256] \to 5$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.2)
    bx(ax, 3.55, y_dn, 2.0, 0.75,
       [r"$\tau_C^{\mathrm{nom}}$ from RNEA"], C_PHY, fs=8.2)
    ar(ax, 4.55, y_up, 5.85, y_up)
    ar(ax, 4.55, y_dn, 5.85, y_dn)
    circ(ax, 6.25, y, 0.30, r"$+$", C_OP, fs=13)
    ar(ax, 5.85, y_up, 5.95, y + 0.25)
    ar(ax, 5.85, y_dn, 5.95, y - 0.25)
    ar(ax, 6.55, y, 7.2, y)
    bx(ax, 8.3, y, 2.3, 0.85,
       [r"$\hat{\bm{\tau}}_C = \tau_C^{\mathrm{nom}} + \delta\mathbf{c}$",
        r"Corrected Coriolis"], C_PHY, fs=8.5)
    note(ax, 3.55, y_dn - 0.65,
         r"Init: $\mathbf{W}_{\mathrm{last}} \sim \mathcal{N}(0,\,10^{-3}),\; \mathbf{b} = \mathbf{0} \;\Rightarrow\; \delta c \approx \mathbf{0}$ (warm-start from RNEA)",
         fs=7.8, c="#6B7280")

    # - Row 3: Gravity correction --------------------------
    y = y_g
    y_up = y + 0.55; y_dn = y - 0.55
    bx(ax, 1.1, y_up, 1.6, 0.75,
       [r"$\tilde{q} \in \mathbb{R}^{5}$"], C_IN)
    bx(ax, 1.1, y_dn, 1.6, 0.75,
       [r"$\tau_g^{\mathrm{nom}}$ (RNEA)"], C_PHY)
    ar(ax, 1.9, y_up, 2.5, y_up)
    ar(ax, 1.9, y_dn, 2.5, y_dn)
    bx(ax, 3.55, y_up, 2.0, 1.05,
       [r"\textbf{dg-net} (correction)",
        r"$5 \to [256,512,256] \to 5$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$"], C_NET, fs=8.2)
    bx(ax, 3.55, y_dn, 2.0, 0.75,
       [r"$\tau_g^{\mathrm{nom}}$ from RNEA"], C_PHY, fs=8.2)
    ar(ax, 4.55, y_up, 5.85, y_up)
    ar(ax, 4.55, y_dn, 5.85, y_dn)
    circ(ax, 6.25, y, 0.30, r"$+$", C_OP, fs=13)
    ar(ax, 5.85, y_up, 5.95, y + 0.25)
    ar(ax, 5.85, y_dn, 5.95, y - 0.25)
    ar(ax, 6.55, y, 7.2, y)
    bx(ax, 8.3, y, 2.3, 0.85,
       [r"$\hat{\bm{\tau}}_g = \tau_g^{\mathrm{nom}} + \delta\mathbf{g}$",
        r"Corrected Gravity"], C_PHY, fs=8.5)
    note(ax, 3.55, y_dn - 0.65,
         r"Init: $\mathbf{W}_{\mathrm{last}} \sim \mathcal{N}(0,\,10^{-3}),\; \mathbf{b} = \mathbf{0} \;\Rightarrow\; \delta g \approx \mathbf{0}$",
         fs=7.8, c="#6B7280")

    # - Row 4: Friction correction (dissipative) ----------
    y = y_f
    y_up = y + 0.60; y_dn = y - 0.60
    bx(ax, 1.1, y_up, 1.6, 0.75,
       [r"$\dot{\tilde{q}} \in \mathbb{R}^{5}$"], C_IN)
    bx(ax, 1.1, y_dn, 1.6, 0.75,
       [r"$\tau_f^{\mathrm{nom}}$ (friction)"], C_PHY)
    ar(ax, 1.9, y_up, 2.5, y_up)
    ar(ax, 1.9, y_dn, 2.5, y_dn)
    bx(ax, 3.8, y_up, 2.5, 1.25,
       [r"\textbf{df-net} (dissipative correction)",
        r"$5 \to [128, 128] \to 10$",
        r"$\tanh \;\mid\; \mathrm{Drop}(0.05)$",
        r"$\to [\mathbf{v},\, \mathbf{c}] \in \mathbb{R}^{10}$"], C_FRI, fs=8.2)
    bx(ax, 3.55, y_dn, 2.0, 0.75,
       [r"$\tau_f^{\mathrm{nom}}$ from friction model"], C_PHY, fs=8.2)
    ar(ax, 5.05, y_up, 5.85, y_up)
    ar(ax, 4.55, y_dn, 5.85, y_dn)
    circ(ax, 6.25, y, 0.30, r"$+$", C_OP, fs=13)
    ar(ax, 5.85, y_up, 5.95, y + 0.25)
    ar(ax, 5.85, y_dn, 5.95, y - 0.25)
    ar(ax, 6.55, y, 7.2, y)
    bx(ax, 8.5, y, 2.7, 1.05,
       [r"$\hat{\bm{\tau}}_f = \tau_f^{\mathrm{nom}} + \delta\mathbf{f}$",
        r"$\delta\mathbf{f} = -[\mathrm{sp}(\mathbf{v})\!\odot\!\dot{\tilde{q}} + \mathrm{sp}(\mathbf{c})\!\odot\!\tanh(\dot{\tilde{q}}/0.04)]$"], C_FRI, fs=8.2)

    # - Final summation ------------------------------------
    x_sum = 13.2; y_sum = 9.5
    circ(ax, x_sum, y_sum, 0.44, r"$\bm{+}$", C_OP, fs=16)

    src_pairs = [
        (10.2 + 1.0, y_M),
        (8.3 + 1.15, 11.0),
        (8.3 + 1.15, 8.2),
        (8.5 + 1.35, 5.4),
    ]
    for sx, sy in src_pairs:
        ar(ax, sx, sy, x_sum - 0.44, y_sum, col="#374151")

    ar(ax, x_sum + 0.44, y_sum, x_sum + 1.2, y_sum)
    bx(ax, 14.85, y_sum, 1.7, 0.95,
       [r"$\hat{\bm{\tau}} \in \mathbb{R}^{5}$"], C_OUT)

    # Full equation
    ax.text(W/2, 3.8,
            r"$\hat{\bm{\tau}} = \mathbf{M}(\tilde{q})\,\ddot{\tilde{q}} \;+\; (\tau_C^{\mathrm{nom}} + \delta\mathbf{c}) \;+\; (\tau_g^{\mathrm{nom}} + \delta\mathbf{g}) \;+\; (\tau_f^{\mathrm{nom}} + \delta\mathbf{f})$",
            ha="center", va="center", fontsize=11, color="#0D9488", zorder=5)

    # Correction regulariser
    ax.text(W/2, 3.15,
            r"$\mathcal{L}_{\mathrm{corr}} = \frac{1}{3}\bigl[\langle\|\delta\mathbf{c}\|^2\rangle + \langle\|\delta\mathbf{g}\|^2\rangle + \langle\|\delta\mathbf{f}\|^2\rangle\bigr]$ --- Occam regulariser: penalises deviation from nominal",
            ha="center", va="center", fontsize=9, color="#374151", zorder=5)

    note(ax, W/2, 2.65,
         r"Nominal consistency loss: \textbf{DISABLED} at runtime (\texttt{tau\_physics\_nom=None} in trainer)",
         fs=8.5, c="#DC2626")
    note(ax, W/2, 2.25,
         r"At $t\!=\!0$: $\hat{\bm{\tau}} \approx \mathbf{M}\ddot{q} + \tau_C^{\mathrm{nom}} + \tau_g^{\mathrm{nom}} + \tau_f^{\mathrm{nom}}$ (warm-start from RNEA)",
         fs=8.5, c="#374151")

    divider(ax, 0.3, W - 0.3, 1.85)

    loss_band(ax, W, 1.20, 0.90, [
        r"$\displaystyle\mathcal{L}_{E.2} \;=\; \mathcal{L}_{\mathrm{data}} \;+\; 0.01\,\mathcal{L}_{\mathrm{SPD}} \;+\; 0.01\,\mathcal{L}_{\mathrm{fric}} \;+\; 0.001\,\mathcal{L}_{\mathrm{corr}}$",
        r"No LossNormaliser $\mid$ No physics weight schedule $\mid$ Nominal consistency anchor $= 0$ (disabled)",
    ])

    save(fig, "13E2_arch_decomposed.png")


# =====================================================================
if __name__ == "__main__":
    draw_A()
    draw_B()
    draw_C()
    draw_D()
    draw_E1()
    draw_E2()
    print("\nAll 6 LaTeX-rendered architecture diagrams saved.")
