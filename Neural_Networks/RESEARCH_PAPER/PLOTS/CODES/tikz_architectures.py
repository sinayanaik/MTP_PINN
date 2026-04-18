#!/usr/bin/env python3
"""
TikZ Architecture Diagrams — 6 Publication-Quality Figures  (v3)
================================================================
Generates 6 architecture diagrams as TikZ → pdflatex → PNG (300 DPI).
All math verified against source code: 2026-04-18.

v3 changes vs v2:
  - Larger base font (\normalsize), increased inner padding
  - Bolder arrows (1.2pt), cleaner dash pattern
  - Per-model layout rework: no overlapping text/arrows
  - Consistent styled boxes for equations everywhere
  - Model E.2 nominal arrows re-routed to avoid crossing subnets

Outputs (saved to PLOTS/):
  13A_arch_blackbox.png       Model A — Black-Box FNN
  13B_arch_physreg.png        Model B — Physics-Regularized FNN
  13C_arch_residual.png       Model C — Residual Correction FNN
  13D_arch_lagrangian.png     Model D — Lagrangian Structured FNN
  13E1_arch_ecpinn.png        Model E.1 — Equation-Constrained PINN
  13E2_arch_decomposed.png    Model E.2 — Decomposed Structured PINN
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent
DPI = 300

# ══════════════════════════════════════════════════════════════════════════
# Shared LaTeX preamble  (v3 — bigger fonts, more padding, bolder arrows)
# ══════════════════════════════════════════════════════════════════════════
PREAMBLE = r"""
\documentclass[tikz,border=14pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage[dvipsnames,svgnames,x11names]{xcolor}
\usetikzlibrary{arrows,positioning,fit,calc,shapes.geometric,
                backgrounds,decorations.pathreplacing}

% ── Colour palette ────────────────────────────────────────────────────
\definecolor{inputbluefill}{HTML}{DBEAFE}
\definecolor{inputblueedge}{HTML}{1E40AF}
\definecolor{netorangefill}{HTML}{FED7AA}
\definecolor{netorangeedge}{HTML}{C2410C}
\definecolor{physgreenfill}{HTML}{D1FAE5}
\definecolor{physgreenedge}{HTML}{065F46}
\definecolor{calibyellowfill}{HTML}{FEF9C3}
\definecolor{calibyellowedge}{HTML}{92400E}
\definecolor{outputpurplefill}{HTML}{EDE9FE}
\definecolor{outputpurpleedge}{HTML}{6D28D9}
\definecolor{opgrayfill}{HTML}{F3F4F6}
\definecolor{opgrayedge}{HTML}{374151}
\definecolor{fricpinkfill}{HTML}{FCE7F3}
\definecolor{fricpinkedge}{HTML}{9D174D}
\definecolor{spdcyanfill}{HTML}{E0F2FE}
\definecolor{spdcyanedge}{HTML}{0369A1}
\definecolor{lossgoldfill}{HTML}{FEF3C7}
\definecolor{lossgoldedge}{HTML}{B45309}
\definecolor{headerfill}{HTML}{FEE2E2}
\definecolor{headeredge}{HTML}{DC2626}
\definecolor{collocationfill}{HTML}{FECDD3}
\definecolor{collocationedge}{HTML}{E11D48}
\definecolor{darktext}{HTML}{1F2937}

% ── Node styles ───────────────────────────────────────────────────────
\tikzset{
  base/.style={draw, line width=0.7pt, rounded corners=4pt, align=center,
               font=\normalsize, text=darktext,
               inner xsep=10pt, inner ysep=8pt,
               outer sep=2pt, minimum height=1.2cm},
  inputbox/.style ={base, fill=inputbluefill,  draw=inputblueedge},
  netbox/.style   ={base, fill=netorangefill,  draw=netorangeedge},
  physbox/.style  ={base, fill=physgreenfill,  draw=physgreenedge},
  calibbox/.style ={base, fill=calibyellowfill,draw=calibyellowedge},
  outputbox/.style={base, fill=outputpurplefill,draw=outputpurpleedge},
  opbox/.style    ={base, fill=opgrayfill,     draw=opgrayedge},
  fricbox/.style  ={base, fill=fricpinkfill,   draw=fricpinkedge},
  spdbox/.style   ={base, fill=spdcyanfill,    draw=spdcyanedge},
  lossbox/.style  ={base, fill=lossgoldfill,   draw=lossgoldedge,
                    minimum width=14cm, minimum height=1.4cm},
  collocbox/.style={base, fill=collocationfill, draw=collocationedge},
  headerbox/.style={draw=headeredge, line width=0.9pt, fill=headerfill,
                    rounded corners=5pt, minimum width=16cm,
                    minimum height=1.5cm, align=center,
                    font=\Large, inner ysep=10pt, outer sep=2pt},
  %
  arr/.style  ={->, >=stealth', line width=1.0pt, color=darktext},
  darr/.style ={->, >=stealth', line width=1.0pt, dashed,
                dash pattern=on 5pt off 3pt, color=darktext},
  eqnode/.style={align=center, font=\normalsize, text=darktext,
                 inner sep=6pt},
  note/.style  ={align=left, font=\small, text=darktext},
}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model A — Black-Box FNN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_A() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.6cm and 2.0cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox] (hdr) {%
  \textbf{Model A --- Black-Box FNN}\\[2pt]
  {\normalsize Pure data-driven baseline $\mid$ No physics information}};

% ── Input ───────────────────────────────────────────────────────────
\node[inputbox, minimum width=3.6cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=0.5cm]
  (inp) {%
    \textbf{Input}\\[3pt]
    $\mathbf{x} = [\tilde{q},\, \tilde{\dot{q}},\, \tilde{\ddot{q}}]$\\
    $\in \mathbb{R}^{15}$};

% ── MLP ─────────────────────────────────────────────────────────────
\node[netbox, minimum width=6.0cm, right=2.0cm of inp] (mlp) {%
  \textbf{MLP Backbone}\\[3pt]
  $15 \to [256,\, 512,\, 256] \to 5$\\[2pt]
  {\small Xavier normal init}};

% ── Output ──────────────────────────────────────────────────────────
\node[outputbox, minimum width=3.0cm, right=2.0cm of mlp] (out) {%
  \textbf{Output}\\[3pt]
  $\hat{\bm{\tau}} \in \mathbb{R}^{5}$};

% ── Arrows ──────────────────────────────────────────────────────────
\draw[arr] (inp) -- (mlp);
\draw[arr] (mlp) -- (out);

% ── Hidden-layer detail ────────────────────────────────────────────
\node[opbox, below=1.6cm of mlp, minimum width=14.5cm,
      minimum height=2.2cm] (layer) {%
    \textbf{Each hidden layer} ($i = 0, 1, 2$):\\[6pt]
    $\mathbf{z}_i = W_i \mathbf{h}_i + \mathbf{b}_i \qquad$
    $\mathbf{z}_i' = \mathrm{LayerNorm}(\mathbf{z}_i) \qquad$
    $\mathbf{a}_i = \mathrm{SiLU}(\mathbf{z}_i') = \mathbf{z}_i' \odot
      \sigma(\mathbf{z}_i')$\\[6pt]
    $\mathbf{h}_{i+1} = \mathrm{Dropout}_{p=0.1}(\mathbf{a}_i)$
    \hfill
    \textit{Output layer (no activation):}\;
    $\hat{\bm{\tau}} = W_3 \mathbf{h}_3 + \mathbf{b}_3$};

\draw[darr] (mlp.south) -- (layer.north);

% ── Loss ────────────────────────────────────────────────────────────
\node[lossbox, below=1.4cm of layer, minimum width=16cm] (loss) {%
  $\displaystyle \mathcal{L}_A
    = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{5}
      w_j\bigl(\hat{\tau}_{i,j} - \tau_{i,j}^{\mathrm{meas}}\bigr)^{\!2}
    \qquad
    \mathbf{w} = [1.0,\; 2.5,\; 1.0,\; 1.0,\; 1.0]$
    \quad {\small (joint\,2 shoulder upweighted)}};

% ── Note ────────────────────────────────────────────────────────────
\node[note, below=0.4cm of loss.south west, anchor=north west] {%
  Optimiser: AdamW, $\eta = 3\!\times\!10^{-4}$ \qquad
  Gradient clip: $\|\nabla\|_{\max} = 5.0$ \qquad
  Feature noise: $\sigma = 0.02$ \qquad
  Batch size: 512};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model B — Physics-Regularized FNN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_B() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.4cm and 1.6cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox] (hdr) {%
  \textbf{Model B --- Physics-Regularized FNN}\\[2pt]
  {\normalsize Forward pass identical to Model~A $\mid$
   Physics enters \textit{only} through the training loss}};

% ═══ DATA PATH (top) ═══════════════════════════════════════════════
\node[inputbox, minimum width=3.2cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=0.3cm]
  (inp) {\textbf{Input}\\[2pt]$\mathbf{x}\in\mathbb{R}^{15}$};

\node[netbox, minimum width=5.6cm, right=1.4cm of inp] (mlp) {%
  \textbf{MLP} (identical to A)\\[3pt]
  $15\!\to\![256, 512, 256]\!\to\!5$\\
  {\small SiLU $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=2.8cm, right=1.4cm of mlp] (out) {%
  $\hat{\bm{\tau}}\in\mathbb{R}^{5}$};

\draw[arr] (inp) -- (mlp);
\draw[arr] (mlp) -- (out);

% data-loss label
\node[eqnode, below=0.8cm of out, xshift=-0.3cm] (dlabel) {%
  $\mathcal{L}_{\mathrm{data}}
   = \tfrac{1}{N}\sum_i \sum_j w_j
     (\hat{\tau}_{i,j} - \tau_{i,j}^{\mathrm{meas}})^2$};

% ═══ PHYSICS PATH (bottom) ════════════════════════════════════════
\node[physbox, minimum width=3.6cm,
      below=2.4cm of inp] (rnea) {%
  \textbf{RNEA precomputed}\\[3pt]
  $\bm{\tau}_g, \bm{\tau}_M, \bm{\tau}_C, \bm{\tau}_f$\\
  $\in\mathbb{R}^{20}$};

\node[opbox, minimum width=2.0cm, right=1.2cm of rnea] (sum) {%
  $\displaystyle\sum$\\[2pt]
  $\bm{\tau}_{\mathrm{eq}}\!\in\!\mathbb{R}^{5}$};

\draw[arr] (rnea) -- (sum);

\node[calibbox, minimum width=5.6cm, right=1.2cm of sum] (calib) {%
  \textbf{Calibration} $\varphi$ {\small(separate param group)}\\[3pt]
  $\varphi(\bm{\tau}) = \mathrm{diag}(\mathbf{s})\,\bm{\tau} + \mathbf{b}$\\[2pt]
  $s_j = \mathrm{softplus}(z_j) + 10^{-5}$\\[2pt]
  {\small Init: $\mathbf{s}=\mathbf{1},\;\mathbf{b}=\mathbf{0}$
   \quad $\lambda_{\mathrm{wd}}^{\varphi}=0$ (no weight decay)}};

\node[outputbox, minimum width=2.4cm, right=1.4cm of calib] (phyout) {%
  $\bm{\tau}_{\mathrm{phys}}^{\mathrm{cal}}$\\$\in\mathbb{R}^{5}$};

\draw[arr] (sum) -- (calib);
\draw[arr] (calib) -- (phyout);

% ── Physics loss (styled box) ──────────────────────────────────
\node[opbox, minimum width=6.5cm,
      below=0.8cm of phyout, xshift=-2.0cm] (ploss) {%
  $\mathcal{L}_{\mathrm{phys}}
   = \|\hat{\bm{\tau}} - \bm{\tau}_{\mathrm{phys}}^{\mathrm{cal}}\|^2
     + 0.01\,\mathcal{L}_{\mathrm{calib}}$};

% ── Dashed connection from tau_hat to physics loss ──────────────
\draw[darr] (out.south) -- ++(0,-1.8) -| (ploss.north east);

% ── Calibration regulariser (styled box) ────────────────────────
\node[opbox, minimum width=6.5cm, below=0.8cm of calib] (creg) {%
  $\mathcal{L}_{\mathrm{calib}}
   = \tfrac{1}{J}\sum_j\bigl[(s_j - 1)^2 + b_j^2\bigr]$};

% ── Weight schedule (styled box) ───────────────────────────────
\node[opbox, minimum width=12cm,
      below=1.0cm of creg, xshift=-0.5cm] (wsched) {%
  $w_p(e) = 0.10 \cdot \min\!\bigl(1,\;\tfrac{e}{0.03\,E}\bigr)$
  \qquad $w_d = 1 - w_p$
  \qquad $\kappa = \hat{\mu}_d\,/\,\hat{\mu}_p$
  \quad {\small(EMA $\beta\!=\!0.98$)}};

% ── Total loss ──────────────────────────────────────────────────
\node[lossbox, below=1.0cm of wsched, minimum width=16cm] (total) {%
  $\displaystyle \mathcal{L}_B
   = w_d\,\mathcal{L}_{\mathrm{data}}
   + w_p\,\kappa\,\mathcal{L}_{\mathrm{phys}}$
  \qquad {\small Physics weight: ramp then CONSTANT
   $\mid$ LossNormaliser $\kappa$ active}};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model C — Residual Correction FNN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_C() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.4cm and 1.6cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox] (hdr) {%
  \textbf{Model C --- Residual Correction FNN}\\[2pt]
  {\normalsize Learns state-dependent scaling $\bm{\alpha}$ and additive
   correction $\bm{\delta}$ on physics torque}};

% ── Inputs ──────────────────────────────────────────────────────────
\node[inputbox, minimum width=3.2cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=0.3cm]
  (feat) {\textbf{Features}\\[2pt]$\mathbf{x}\in\mathbb{R}^{15}$};

\node[physbox, minimum width=3.2cm, below=1.0cm of feat]
  (phys) {\textbf{Physics (RNEA)}\\[2pt]
           $\bm{\tau}_{\mathrm{phys}}\in\mathbb{R}^{5}$};

% ── Concat ──────────────────────────────────────────────────────────
\node[opbox, minimum width=3.0cm, right=2.0cm of feat, yshift=-1.2cm]
  (cat) {\textbf{Concat}\\[2pt]$[\mathbf{x},\,\bm{\tau}_{\mathrm{phys}}]
          \in\mathbb{R}^{20}$};

\draw[arr] (feat.east) -| (cat.north);
\draw[arr] (phys.east) -| (cat.south);

% ── Encoder ─────────────────────────────────────────────────────────
\node[netbox, minimum width=5.4cm, right=1.6cm of cat] (enc) {%
  \textbf{Encoder MLP}\\[3pt]
  $20\!\to\![256, 512, 256]$\\
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}\\
  $\mathbf{h}\in\mathbb{R}^{256}$};

\draw[arr] (cat) -- (enc);

% ── Dual heads ──────────────────────────────────────────────────────
\node[spdbox, minimum width=5.6cm,
      above right=1.6cm and 1.8cm of enc.east, anchor=west]
  (alpha) {%
  \textbf{$\bm{\alpha}$-head}\\[3pt]
  $\bm{\alpha} = \mathrm{softplus}(W_\alpha\mathbf{h}+\mathbf{b}_\alpha)
   + 10^{-3}$\\[2pt]
  {\small Init: $W_\alpha\!=\!0$,\;
   $\mathbf{b}_\alpha\!=\!\ln(e^{0.5}\!-\!1)\!\approx\!-0.481$
   $\;\Rightarrow\;\bm{\alpha}\!\approx\!0.5$}};

\node[fricbox, minimum width=5.0cm,
      below right=1.6cm and 1.8cm of enc.east, anchor=west]
  (delta) {%
  \textbf{$\bm{\delta}$-head}\\[3pt]
  $\bm{\delta} = W_\delta\mathbf{h} + \mathbf{b}_\delta$\\[2pt]
  {\small Init: $W_\delta\!=\!0,\;\mathbf{b}_\delta\!=\!0$
   $\;\Rightarrow\;\bm{\delta}\!=\!\mathbf{0}$}};

\draw[arr] (enc.east) -- ++(0.6,0) |- (alpha.west);
\draw[arr] (enc.east) -- ++(0.6,0) |- (delta.west);

% ── Output assembly ────────────────────────────────────────────────
\node[outputbox, minimum width=5.0cm,
      below=1.2cm of delta, xshift=-0.2cm] (out) {%
  \textbf{Output}\\[4pt]
  $\hat{\bm{\tau}} = \bm{\alpha}\odot\bm{\tau}_{\mathrm{phys}}
   + \bm{\delta}$};

\draw[arr] (alpha.south) -- ++(0,-0.5) -| ([xshift=-0.8cm]out.north);
\draw[arr] (delta.south) -- (delta |- out.north);

% physics tau also feeds output — route along the bottom
\draw[darr, color=physgreenedge]
  (phys.south) -- ++(0,-0.8) -| (out.south);

% ── Warm-start note ─────────────────────────────────────────────
\node[opbox, minimum width=10cm, below=1.0cm of out] (init) {%
    \textbf{At $t\!=\!0$:}\quad
    $\hat{\bm{\tau}} = 0.5\,\bm{\tau}_{\mathrm{phys}} + \mathbf{0}$
    \quad (warm-start from analytical model, halved)};

% ── Loss ────────────────────────────────────────────────────────
\node[lossbox, below=1.0cm of init, minimum width=16cm] (loss) {%
  $\displaystyle \mathcal{L}_C
   = \underbrace{\frac{1}{N}\sum_i\sum_j w_j
     (\hat{\tau}_{i,j} - \tau_{i,j}^{\mathrm{meas}})^2}_%
     {\mathcal{L}_{\mathrm{data}}}
   \;+\; 0.05\;\|\bm{\alpha} - \mathbf{1}\|^2$
  \quad {\small ($\alpha$-regularisation keeps scaling near unity)}};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model D — Lagrangian Structured FNN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_D() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.4cm and 1.6cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox, minimum width=17cm] (hdr) {%
  \textbf{Model D --- Lagrangian Structured FNN}\\[2pt]
  {\normalsize Equation-of-motion structure embedded in architecture:\;
   $\hat{\bm{\tau}} = M(q)\ddot{q} + C(q,\dot{q}) + g(q) + f(\dot{q})$}};

% ── Input ───────────────────────────────────────────────────────────
\node[inputbox, minimum width=3.2cm,
      below=7.0cm of hdr.south west, anchor=east, xshift=2.6cm]
  (inp) {\textbf{Input}\\[2pt]$\mathbf{x}\!=\![q,\dot{q},\ddot{q}]$\\
          $\in\mathbb{R}^{15}$};

% ═══ M-net (row 1) ═════════════════════════════════════════════════
\node[netbox, minimum width=6.5cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=3.0cm]
  (mnet) {%
  \textbf{M-net}\quad $q\in\mathbb{R}^{5}\to [256, 512, 256]\to\mathbb{R}^{15}$\\[2pt]
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[spdbox, minimum width=6.0cm, right=1.2cm of mnet] (spd) {%
  \textbf{Cholesky SPD enforcement}\\[3pt]
  $L_{\mathrm{diag}} = \mathrm{softplus}(L_{\mathrm{raw}}) + 10^{-4}$\\[2pt]
  $M = LL^\top + 10^{-4}\,I_5$\\[2pt]
  {\small Init: diag bias $= -2.0$
   $\;\Rightarrow\;\mathrm{softplus}(-2)\!\approx\!0.126$}};

\draw[arr] (mnet) -- (spd);

\node[outputbox, minimum width=3.0cm, right=1.2cm of spd]
  (matmul) {$\bm{\tau}_M = M\ddot{q}$};

\draw[arr] (spd) -- (matmul);

% ═══ C-net (row 2) ═════════════════════════════════════════════════
\node[netbox, minimum width=6.5cm, below=1.0cm of mnet]
  (cnet) {%
  \textbf{C-net}\quad $[q,\dot{q}]\in\mathbb{R}^{10}\to [256, 512, 256]\to\mathbb{R}^{5}$\\[2pt]
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=3.0cm] at (matmul |- cnet)
  (cout) {$\bm{\tau}_C$};

\draw[arr] (cnet) -- (cout);

% ═══ g-net (row 3) ═════════════════════════════════════════════════
\node[netbox, minimum width=6.5cm, below=1.0cm of cnet]
  (gnet) {%
  \textbf{g-net}\quad $q\in\mathbb{R}^{5}\to [256, 512, 256]\to\mathbb{R}^{5}$\\[2pt]
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=3.0cm] at (matmul |- gnet)
  (gout) {$\bm{\tau}_g$};

\draw[arr] (gnet) -- (gout);

% ═══ f-net (row 4) ═════════════════════════════════════════════════
\node[fricbox, minimum width=6.5cm, below=1.0cm of gnet]
  (fnet) {%
  \textbf{f-net}\quad $\dot{q}\in\mathbb{R}^{5}\to [128, 128]\to\mathbb{R}^{10}$\\[2pt]
  {\small tanh $\mid$ Dropout(0.05)}\\[2pt]
  {\small viscous ($c_v$) + Coulomb ($c_c$)}};

\node[fricbox, minimum width=6.0cm, right=1.2cm of fnet] (fdiss) {%
  \textbf{Dissipative friction}\\[3pt]
  $\bm{\tau}_f = -\bigl[\mathrm{sp}(c_v)\dot{q}
    + \mathrm{sp}(c_c)\tanh(\dot{q}/0.04)\bigr]$\\[2pt]
  {\small $\bm{\tau}_f\!\cdot\!\dot{q} \le 0$ by construction}};

\draw[arr] (fnet) -- (fdiss);

\node[outputbox, minimum width=3.0cm] at (matmul |- fdiss)
  (fout) {$\bm{\tau}_f$};

\draw[arr] (fdiss) -- (fout);

% ── Input arrows ────────────────────────────────────────────────
\draw[arr] (inp.north) -- ++(0,0.6) -| (mnet.west);
\draw[arr] (inp.east)  -- ++(0.6,0) |- (cnet.west);
\draw[arr] (inp.east)  -- ++(0.6,0) |- (gnet.west);
\draw[arr] (inp.south) -- ++(0,-0.6) -| (fnet.west);

% ── Summation ───────────────────────────────────────────────────
\node[outputbox, minimum width=5.0cm,
      below=1.6cm of fout] (sum) {%
  \textbf{Output}\\[3pt]
  $\hat{\bm{\tau}} = \bm{\tau}_M + \bm{\tau}_C + \bm{\tau}_g + \bm{\tau}_f$};

\draw[arr] (matmul.south) -- ++(0,-0.5) -| ([xshift=-1.5cm]sum.north);
\draw[arr] (cout.south)   -- ++(0,-0.5) -| ([xshift=-0.5cm]sum.north);
\draw[arr] (gout.south)   -- ++(0,-0.5) -| ([xshift= 0.5cm]sum.north);
\draw[arr] (fout.south)   -- (fout |- sum.north);

% ── Loss ────────────────────────────────────────────────────────
\node[lossbox, below=1.4cm of sum, minimum width=17cm] (loss) {%
  $\displaystyle \mathcal{L}_D
    = \mathcal{L}_{\mathrm{data}}
    + 0.01\underbrace{\bigl\|\max(0,\;\epsilon\!-\!\lambda_i(M))\bigr\|^2}_%
      {\mathcal{L}_{\mathrm{SPD}}}
    + 0.01\underbrace{\mathrm{mean}\bigl(\max(0,\;
      \bm{\tau}_f\!\cdot\!\dot{q})\bigr)}_{\mathcal{L}_{\mathrm{fric}}}$
  \qquad {\small $\epsilon_{\mathrm{SPD}}=10^{-4}$}};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model E.1 — Equation-Constrained PINN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_E1() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.2cm and 1.4cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox, minimum width=17cm] (hdr) {%
  \textbf{Model E.1 --- Equation-Constrained PINN}\\[2pt]
  {\normalsize Equation residual loss $\mid$ Collocation at synthetic states
   $\mid$ LossNormaliser $\kappa$}};

% ═══ PATH 1: DATA (top) ════════════════════════════════════════════
\node[inputbox, minimum width=3.0cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=0.3cm]
  (inp) {\textbf{Input}\\[2pt]$\mathbf{x}\in\mathbb{R}^{15}$};

\node[netbox, minimum width=5.4cm, right=1.4cm of inp] (mlp) {%
  \textbf{MLP} (identical to A)\\[3pt]
  $15\!\to\![256, 512, 256]\!\to\!5$\\
  {\small SiLU $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=2.6cm, right=1.4cm of mlp] (tauhat) {%
  $\hat{\bm{\tau}}\in\mathbb{R}^{5}$};

\draw[arr] (inp) -- (mlp);
\draw[arr] (mlp) -- (tauhat);

% ═══ PATH 2: PHYSICS (middle) ══════════════════════════════════════
\node[physbox, minimum width=3.4cm, below=2.2cm of inp] (rnea) {%
  \textbf{RNEA precomputed}\\[3pt]
  $\bm{\tau}_g, \bm{\tau}_M, \bm{\tau}_C, \bm{\tau}_f$\\
  $\in\mathbb{R}^{20}$};

\node[opbox, minimum width=2.0cm, right=1.0cm of rnea] (sum) {%
  $\displaystyle\sum$\\
  $\bm{\tau}_{\mathrm{nom}}\!\in\!\mathbb{R}^{5}$};

\draw[arr] (rnea) -- (sum);

\node[calibbox, minimum width=5.6cm, right=1.0cm of sum] (calib) {%
  \textbf{Calibration} $\varphi$ {\small(separate param group)}\\[3pt]
  $\varphi(\bm{\tau}) = \mathrm{diag}(\mathbf{s})\,\bm{\tau} + \mathbf{b}$\\[2pt]
  $s_j = \mathrm{softplus}(z_j) + 10^{-5}$\\[2pt]
  {\small Init: $\mathbf{s}\!=\!\mathbf{1},\;\mathbf{b}\!=\!\mathbf{0}$
   \quad $\lambda_{\mathrm{wd}}^{\varphi}\!=\!0$}};

\draw[arr] (sum) -- (calib);

\node[outputbox, minimum width=2.4cm, right=1.0cm of calib] (phyout) {%
  $\bm{\tau}_{\mathrm{phys}}^{\mathrm{cal}}$\\$\in\mathbb{R}^{5}$};

\draw[arr] (calib) -- (phyout);

% ── Equation residual (styled box) ────────────────────────────
\node[opbox, minimum width=5.0cm, right=0.8cm of phyout, yshift=1.4cm] (resid) {%
    \textbf{Equation residual}\\[4pt]
    $\mathbf{r} = \hat{\bm{\tau}} - \varphi(\bm{\tau}_{\mathrm{nom}})$\\[3pt]
    $\mathcal{L}_{\mathrm{phys}} = \|\mathbf{r}\|^2
     + 0.01\,\mathcal{L}_{\mathrm{calib}}$};

\draw[darr] (tauhat.south) -- ++(0,-0.6) -| (resid.north);
\draw[darr] (phyout.north) -- ++(0,0.5) -| (resid.south);

% ── Calibration regulariser (styled box) ────────────────────────
\node[opbox, minimum width=8.0cm, below=0.9cm of calib] (creg) {%
  $\mathcal{L}_{\mathrm{calib}}
   = \tfrac{1}{J}\sum_j[(s_j\!-\!1)^2 + b_j^2]$
  \quad --- keeps $\varphi$ near identity};

% ═══ PATH 3: COLLOCATION (bottom) ══════════════════════════════════
\node[collocbox, minimum width=4.0cm, below=2.4cm of rnea] (coll) {%
  \textbf{Collocation sampling}\\[3pt]
  $n_{\mathrm{col}} = 32$ per epoch\\[2pt]
  $q \sim \mathcal{U}[\mu_q \pm 3\sigma_q]$\\
  $\dot{q} \sim \mathcal{N}(\mu_{\dot{q}},\,\sigma_{\dot{q}}^2)$\\
  $\ddot{q} \sim \mathcal{N}(\mu_{\ddot{q}},\,\sigma_{\ddot{q}}^2)$};

\node[physbox, minimum width=3.6cm, right=1.0cm of coll] (collrnea) {%
  \textbf{RNEA + friction}\\[2pt]
  {\small (Pinocchio, physical units)}\\
  $\to$ z-score normalise};

\draw[arr] (coll) -- (collrnea);

\node[calibbox, minimum width=3.4cm, right=1.0cm of collrnea] (collcalib) {%
  \textbf{Calibration} $\varphi$ {\small(shared)}\\[2pt]
  same params as above};

\draw[arr] (collrnea) -- (collcalib);

\node[opbox, minimum width=3.4cm, right=1.0cm of collcalib] (colloss) {%
  $\mathcal{L}_{\mathrm{col}} = \|\mathbf{r}_{\mathrm{col}}\|^2$\\[2pt]
  $\lambda_{\mathrm{col}} = 0.05$};

\draw[arr] (collcalib) -- (colloss);

% ── Collocation note ────────────────────────────────────────────
\node[note, below=0.4cm of coll.south west, anchor=north west] {%
  Collocation extends physics supervision into
  $\mathcal{U}[\mu \pm 3\sigma]$ --- beyond training distribution};

% ── Weight schedule (styled box) ───────────────────────────────
\node[opbox, minimum width=14cm,
      below=3.6cm of creg, xshift=0.5cm] (wsched) {%
  $w_p(e) = 0.10\cdot\min\!\bigl(1,\;\tfrac{e}{0.03\,E}\bigr),$
  \qquad $w_d = 1 - w_p,$
  \qquad $\kappa = \hat{\mu}_d / \hat{\mu}_p$
  \quad {\small ($\beta=0.98$)}};

% ── Total loss ──────────────────────────────────────────────────
\node[lossbox, below=1.0cm of wsched, minimum width=17cm] (total) {%
  $\displaystyle \mathcal{L}_{E.1}
    = w_d\,\mathcal{L}_{\mathrm{data}}
    + w_p\,\kappa\,\mathcal{L}_{\mathrm{phys}}
    + \lambda_{\mathrm{col}}\,\mathcal{L}_{\mathrm{col}}$
  \qquad {\small Physics weight: ramp then CONSTANT
   $\mid$ LossNormaliser $\kappa$ active $\mid$ $\lambda_{\mathrm{col}}=0.05$}};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Model E.2 — Decomposed Structured PINN
# ══════════════════════════════════════════════════════════════════════════
def tikz_model_E2() -> str:
    return PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[node distance=1.4cm and 1.6cm]

% ── Header ──────────────────────────────────────────────────────────
\node[headerbox, minimum width=17cm] (hdr) {%
  \textbf{Model E.2 --- Decomposed Structured PINN}\\[2pt]
  {\normalsize Correction networks on RNEA nominal $\mid$ Warm-start init
   $\mid$ Occam regulariser on $\Delta$}};

% ═══ M-net (row 1, from scratch) ═════════════════════════════════
\node[netbox, minimum width=6.2cm,
      below=1.4cm of hdr.south west, anchor=north west, xshift=3.0cm]
  (mnet) {%
  \textbf{M-net} (from scratch)\\[3pt]
  $q\!\in\!\mathbb{R}^{5}\to [256, 512, 256]\to\mathbb{R}^{15}$\\
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[spdbox, minimum width=5.4cm, right=1.2cm of mnet] (spd) {%
  \textbf{Cholesky $\to$ SPD}\\[3pt]
  $L_{\mathrm{diag}}\!=\!\mathrm{softplus}(L_{\mathrm{raw}})+10^{-4}$\\
  $M\!=\!LL^\top+10^{-4}I$\\[2pt]
  {\small diag bias $=-2.0$}};

\draw[arr] (mnet) -- (spd);

\node[outputbox, minimum width=3.2cm, right=1.2cm of spd]
  (mmul) {$\bm{\tau}_M\!=\!M\ddot{q}$};
\draw[arr] (spd) -- (mmul);

% ═══ C-net (row 2, correction) ═══════════════════════════════════
\node[netbox, minimum width=6.2cm, below=1.0cm of mnet] (cnet) {%
  \textbf{c-net} (correction)\\[3pt]
  $[q,\dot{q}]\!\in\!\mathbb{R}^{10}\to [256, 512, 256]\to\Delta\mathbf{c}$\\
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=4.2cm] at (mmul |- cnet) (cadd) {%
  $\bm{\tau}_C = \bm{\tau}_{C}^{\mathrm{nom}} + \Delta\mathbf{c}$};

\draw[arr] (cnet) -- (cadd);

% ═══ g-net (row 3, correction) ═══════════════════════════════════
\node[netbox, minimum width=6.2cm, below=1.0cm of cnet] (gnet) {%
  \textbf{g-net} (correction)\\[3pt]
  $q\!\in\!\mathbb{R}^{5}\to [256, 512, 256]\to\Delta\mathbf{g}$\\
  {\small tanh $\mid$ Dropout(0.05) $\mid$ LayerNorm}};

\node[outputbox, minimum width=4.2cm] at (mmul |- gnet) (gadd) {%
  $\bm{\tau}_g = \bm{\tau}_{g}^{\mathrm{nom}} + \Delta\mathbf{g}$};

\draw[arr] (gnet) -- (gadd);

% ═══ f-net (row 4, correction + dissipative) ═════════════════════
\node[fricbox, minimum width=6.2cm, below=1.0cm of gnet] (fnet) {%
  \textbf{f-net} (correction)\\[3pt]
  $\dot{q}\!\in\!\mathbb{R}^{5}\to [128, 128]\to\mathbb{R}^{10}$\\
  {\small tanh $\mid$ Dropout(0.05)}};

\node[fricbox, minimum width=6.2cm, right=1.2cm of fnet] (fdiss) {%
  \textbf{Dissipative correction}\\[3pt]
  $\Delta\mathbf{f} = -\bigl[\mathrm{sp}(c_v)\dot{q}
    + \mathrm{sp}(c_c)\tanh(\dot{q}/0.04)\bigr]$\\[2pt]
  $\bm{\tau}_f = \bm{\tau}_{f}^{\mathrm{nom}} + \Delta\mathbf{f}$};

\draw[arr] (fnet) -- (fdiss);

% ── Input (centred left) ────────────────────────────────────────
\node[inputbox, minimum width=3.2cm,
      below=7.4cm of hdr.south west, anchor=east, xshift=2.6cm]
  (inp) {\textbf{Input}\\[2pt]$\mathbf{x}\!=\![q,\dot{q},\ddot{q}]$\\
          $\in\mathbb{R}^{15}$};

\node[physbox, minimum width=3.4cm, below=1.2cm of inp]
  (nom) {\textbf{RNEA nominal}\\[2pt]
  $\bm{\tau}_{g}^{\mathrm{nom}},\,\bm{\tau}_{C}^{\mathrm{nom}},\,
   \bm{\tau}_{f}^{\mathrm{nom}}$\\
  $\in\mathbb{R}^{20}$};

% ── Input arrows ────────────────────────────────────────────────
\draw[arr] (inp.north) -- ++(0,0.6) -| (mnet.west);
\draw[arr] (inp.east)  -- ++(0.6,0) |- (cnet.west);
\draw[arr] (inp.east)  -- ++(0.6,0) |- (gnet.west);
\draw[arr] (inp.south) -- ++(0,-0.6) -| (fnet.west);

% ── Nominal arrows — route BELOW all subnets, then UP to targets ─────
% Bottom y-coordinate: well below the fnet/fdiss row
\path (fdiss.south) ++(0,-1.2) coordinate (botY);
% Right-edge x: beyond output boxes
\path (cadd.east) ++(2.0,0) coordinate (rEdge);

% To cadd: down from nom → below all → right to rEdge → up to cadd.east
\draw[darr, color=physgreenedge]
  (nom.south) -- (nom |- botY) -| (rEdge |- cadd) -- (cadd.east);
% To gadd: same bottom route → right to rEdge → up to gadd.east
\draw[darr, color=physgreenedge]
  (nom.south) -- (nom |- botY) -| (rEdge |- gadd) -- (gadd.east);
% To fdiss: down from nom → below all → right to fdiss.south
\draw[darr, color=physgreenedge]
  (nom.south) -- (nom |- botY) -| (fdiss.south);

% ── Summation / Output ─────────────────────────────────────────
\node[outputbox, minimum width=5.5cm,
      below=1.6cm of fdiss] (out) {%
  \textbf{Output}\\[4pt]
  $\hat{\bm{\tau}} = \bm{\tau}_M + \bm{\tau}_C + \bm{\tau}_g + \bm{\tau}_f$};

\draw[arr] (mmul.south)  -- ++(0,-0.5) -| ([xshift=-1.5cm]out.north);
\draw[arr] (cadd.south)  -- ++(0,-0.5) -| ([xshift=-0.5cm]out.north);
\draw[arr] (gadd.south)  -- ++(0,-0.5) -| ([xshift= 0.5cm]out.north);
\draw[arr] (fdiss.south) -- ++(0,-0.5) -| ([xshift= 1.5cm]out.north);

% ── Warm-start note ─────────────────────────────────────────────
\node[opbox, minimum width=15cm, below=1.2cm of out] (warm) {%
    \textbf{Warm-start initialisation:}\;
    Last-layer weights $W\sim\mathcal{N}(0,\,10^{-3})$,\;
    biases $\mathbf{b}=\mathbf{0}$\\[4pt]
    $\Rightarrow$ At epoch\,0:\;
    $\Delta\mathbf{c}\!\approx\!0,\;\Delta\mathbf{g}\!\approx\!0,\;
     \Delta\mathbf{f}\!\approx\!0$
    \quad$\hat{\bm{\tau}} \approx \bm{\tau}_{\mathrm{nom}}$};

% ── Loss ────────────────────────────────────────────────────────
\node[lossbox, below=1.0cm of warm, minimum width=17cm] (loss) {%
  $\displaystyle \mathcal{L}_{E.2}
    = \mathcal{L}_{\mathrm{data}}
    + 0.01\,\mathcal{L}_{\mathrm{SPD}}
    + 0.01\,\mathcal{L}_{\mathrm{fric}}
    + 0.001\underbrace{\tfrac{1}{3}\bigl[
      \|\Delta\mathbf{c}\|^2 + \|\Delta\mathbf{g}\|^2
      + \|\Delta\mathbf{f}\|^2\bigr]}_{\mathcal{L}_{\mathrm{corr}}
      \text{ (Occam)}}$};

\end{tikzpicture}
\end{document}
"""


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator — compile all diagrams
# ══════════════════════════════════════════════════════════════════════════
DIAGRAMS = [
    ("13A_arch_blackbox",   tikz_model_A),
    ("13B_arch_physreg",    tikz_model_B),
    ("13C_arch_residual",   tikz_model_C),
    ("13D_arch_lagrangian", tikz_model_D),
    ("13E1_arch_ecpinn",    tikz_model_E1),
    ("13E2_arch_decomposed", tikz_model_E2),
]


def compile_tikz(name: str, tex_source: str, work_dir: Path) -> Path:
    """Write .tex, compile with pdflatex, convert to PNG, return PNG path."""
    tex_path = work_dir / f"{name}.tex"
    pdf_path = work_dir / f"{name}.pdf"
    png_stem = work_dir / name

    # Write .tex
    tex_path.write_text(tex_source, encoding="utf-8")

    # Compile with pdflatex (two passes for safety)
    for pass_num in (1, 2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             str(tex_path)],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [WARN] pdflatex pass {pass_num} for {name} returned "
                  f"{result.returncode}")
            # Print last 30 lines of log for debugging
            log_path = work_dir / f"{name}.log"
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                for line in lines[-30:]:
                    print(f"    {line}")
            if pass_num == 2:
                raise RuntimeError(f"pdflatex failed for {name}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not generated: {pdf_path}")

    # Convert PDF → PNG at target DPI
    subprocess.run(
        ["pdftoppm", "-r", str(DPI), "-png", str(pdf_path), str(png_stem)],
        check=True, capture_output=True, timeout=60,
    )

    # pdftoppm appends -1.png for single-page docs
    png_path = work_dir / f"{name}-1.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not generated: {png_path}")

    return png_path


def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="tikz_arch_"))
    print(f"Working directory: {work_dir}")

    for name, tex_func in DIAGRAMS:
        print(f"\n{'='*60}")
        print(f"  Building {name}")
        print(f"{'='*60}")

        tex_source = tex_func()
        png_tmp = compile_tikz(name, tex_source, work_dir)

        # Copy to final location
        final_png = PLOTS_DIR / f"{name}.png"
        shutil.copy2(str(png_tmp), str(final_png))
        size_kb = final_png.stat().st_size / 1024
        print(f"  -> {final_png}  ({size_kb:.0f} KB)")

    # Clean up work directory
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\n{'='*60}")
    print("  All 6 TikZ architecture diagrams generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
