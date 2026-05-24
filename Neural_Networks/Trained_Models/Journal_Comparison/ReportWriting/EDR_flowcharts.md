# EDR Architecture — Mermaid Flowcharts (split in parts)

Detailed, mathematically explicit flowcharts of the Equivariant–Decomposed–Residual
(EDR) network for 5-DOF Kikobot inverse-dynamics. Each part is a self-contained
`mermaid` diagram. Formula nodes use **KaTeX math** (`$$ … $$`); process/IO boxes
are plain text. Colour key:

- **green** = inputs / outputs (I/O)
- **blue**  = analytical / fixed physics (RNEA + friction, *not learned*)
- **red**   = learned correction (δ-networks)
- **white** = deterministic operation (normalise, scatter, product, sum)

Render to readable images (math typeset by KaTeX) with:

```bash
mmdc -i EDR_flowcharts.md -o renders/edr.svg -b white -c mermaid.json -p puppeteer.json
# mermaid.json: {"theme":"base","flowchart":{"htmlLabels":true,"useMaxWidth":false}}
# then rasterise each SVG to PNG with headless Chrome (KaTeX renders correctly).
```

> Note: KaTeX labels must be **single-line** (`\\`, `<br/>` collapse); each node is
> one `$$…$$` expression with `\text{}` for words.

---

## Part 1 — Data pipeline & nominal physics decomposition

```mermaid
flowchart TD
  HW["5-DOF Kikobot<br/>measured trajectories"]:::io
  HW --> STATE["$$\text{Joint state}\;\; q,\dot q,\ddot q\in\mathbb{R}^5$$"]:::io
  HW --> TMEAS["$$\text{Measured torque}\;\; \tau_{\text{meas}}\in\mathbb{R}^5$$"]:::io

  STATE --> NORMK["$$\tilde q=\tfrac{q-\mu_q}{\sigma_q},\;\; \tilde{\dot q}=\tfrac{\dot q-\mu_{\dot q}}{\sigma_{\dot q}},\;\; \tilde{\ddot q}=\tfrac{\ddot q-\mu_{\ddot q}}{\sigma_{\ddot q}}$$"]:::op
  NORMK --> FEAT["$$\text{Kinematic features}\;\; \tilde x\in\mathbb{R}^{15}$$"]:::io

  STATE --> RNEA["Pinocchio RNEA<br/>(URDF · fixed)"]:::phys
  STATE --> FRICM["Identified friction<br/>model (fixed)"]:::phys
  RNEA --> TG["$$\tau_g=g(q)$$"]:::phys
  RNEA --> TM["$$\tau_M=M(q)\,\ddot q$$"]:::phys
  RNEA --> TC["$$\tau_C=C(q,\dot q)\,\dot q$$"]:::phys
  FRICM --> TF["$$\tau_f(\dot q)$$"]:::phys

  TG --> NORMP["$$\tilde\tau_x=\tfrac{\tau_x-\mu_\tau/4}{\sigma_\tau},\;\; x\in\{g,M,C,f\}$$"]:::op
  TM --> NORMP
  TC --> NORMP
  TF --> NORMP
  NORMP --> PVEC["$$p=[\tilde\tau_g,\tilde\tau_M,\tilde\tau_C,\tilde\tau_f]\in\mathbb{R}^{20},\;\; \textstyle\sum_x\tilde\tau_x=\tilde\tau$$"]:::io

  TMEAS --> NORMT["$$\tilde\tau=\tfrac{\tau_{\text{meas}}-\mu_\tau}{\sigma_\tau}$$"]:::op
  NORMT --> TARGET["$$\text{Target}\;\; \tilde\tau\in\mathbb{R}^5$$"]:::io

  FEAT --> EDR["EDR model — Part 2"]:::learn
  PVEC --> EDR
  EDR --> THAT["$$\hat{\tilde\tau}\;\; \text{(normalised)}$$"]:::io
  THAT --> DEN["$$\hat\tau=\hat{\tilde\tau}\odot\sigma_\tau+\mu_\tau$$"]:::op
  DEN --> TPHYS["$$\text{physical torque}\;\; \hat\tau\in\mathbb{R}^5$$"]:::io
  TARGET -. training loss .-> EDR

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef phys fill:#e7edf9,stroke:#3b5b92,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 2 — EDR forward pass (top level)

```mermaid
flowchart TD
  IN1["$$\text{Kinematic features}\;\; \tilde x\in\mathbb{R}^{15}\to q,\dot q,\ddot q$$"]:::io
  IN2["$$\text{Decomposed physics}\;\; p\in\mathbb{R}^{20}$$"]:::io
  IN1 --> BCI["build_correction_inputs<br/>trig · velocity products ·<br/>physics-conditioning slices"]:::op
  IN2 --> BCI
  IN2 --> PS["$$\text{physics slices}\;\; \tilde\tau_g,\tilde\tau_M,\tilde\tau_C,\tilde\tau_f$$"]:::phys

  subgraph BR["Correction branches"]
    BCI --> DG["$$\delta g(q)\in\mathbb{R}^5\;\; \text{(Part 3)}$$"]:::learn
    BCI --> DM["$$\delta M(q)\in\mathbb{R}^{5\times5}\ \text{sym. (Part 4)}$$"]:::learn
    DM --> DMQ["$$\delta M\,\ddot q\in\mathbb{R}^5$$"]:::op
    BCI --> DC["$$\delta C\,\dot q\in\mathbb{R}^5\;\; \text{(Part 5)}$$"]:::learn
    BCI --> DF["$$\delta\tau_f(\dot q)\in\mathbb{R}^5\;\; \text{(Part 6)}$$"]:::learn
  end

  GAMMA["$$\text{capacity gate}\;\; \gamma\in[0,1]\;\; \text{(Part 7)}$$"]:::op
  GAMMA --> GMUL["$$\gamma\,(\delta M\,\ddot q)$$"]:::op
  GAMMA --> GMUL2["$$\gamma\,(\delta C\,\dot q)$$"]:::op
  DMQ --> GMUL
  DC --> GMUL2

  PS --> AG["$$\tilde\tau_g+\delta g$$"]:::op
  DG --> AG
  PS --> AM["$$\tilde\tau_M+\gamma\,\delta M\,\ddot q$$"]:::op
  GMUL --> AM
  PS --> AC["$$\tilde\tau_C+\gamma\,\delta C\,\dot q$$"]:::op
  GMUL2 --> AC
  PS --> AF["$$\tilde\tau_f+\delta\tau_f$$"]:::op
  DF --> AF

  AG --> SUM["$$\hat{\tilde\tau}=(\tilde\tau_g+\delta g)+(\tilde\tau_M+\gamma\delta M\ddot q)+(\tilde\tau_C+\gamma\delta C\dot q)+(\tilde\tau_f+\delta\tau_f)$$"]:::io
  AM --> SUM
  AC --> SUM
  AF --> SUM
  SUM --> OUT["$$\text{at init }\delta\approx 0\Rightarrow\hat{\tilde\tau}=\tilde\tau_{\text{phys}}$$"]:::io

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef phys fill:#e7edf9,stroke:#3b5b92,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 3 — Gravity correction

```mermaid
flowchart LR
  Q["$$\tilde q\;\; \text{(normalised)}$$"]:::io --> RAW["$$q_{\text{raw}}=\tilde q\odot\sigma_q+\mu_q$$"]:::op
  RAW --> TRIG["$$[\,\tilde q,\ \sin q_{\text{raw}},\ \cos q_{\text{raw}}\,]\in\mathbb{R}^{15}$$"]:::op
  TGC["$$\tilde\tau_g\;\; \text{(phys. cond.)}$$"]:::phys --> CAT["$$\text{concat}\to z_g\in\mathbb{R}^{20}$$"]:::op
  TRIG --> CAT
  CAT --> MLP["$$\mathrm{MLP}_g\,[48,48]\ \text{SiLU},\ W^{(L)}\!\sim\!\mathcal{N}(0,10^{-4})$$"]:::learn
  MLP --> OUT["$$\delta g\in\mathbb{R}^5\;\; \text{(function of }q\text{ only)}$$"]:::learn

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef phys fill:#e7edf9,stroke:#3b5b92,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 4 — Inertia correction

```mermaid
flowchart LR
  Z["$$z_M\in\mathbb{R}^{20}:\ [\,\tilde q,\sin q_{\text{raw}},\cos q_{\text{raw}},\tilde\tau_M\,]$$"]:::op --> MLP["$$\mathrm{MLP}_M\,[48,48]\to\mathbb{R}^{m},\ m=\tfrac{n(n+1)}{2}=15$$"]:::learn
  MLP --> ENT["$$\text{triangular entries}\;\; c_k(q)$$"]:::learn
  ENT --> SYM["$$\delta M=\textstyle\sum_k c_k\,E^{\text{sym}}_k\Rightarrow\delta M=\delta M^\top$$"]:::op
  ENT -.-> PSD["$$\text{PSD option:}\;\; \delta M=LL^\top\succeq 0$$"]:::op
  SYM --> DM["$$\delta M(q)\in\mathbb{R}^{5\times5}$$"]:::learn
  PSD -.-> DM
  QDD["$$\ddot q$$"]:::io --> MUL["$$\delta M\cdot\ddot q\;\; \text{(mat--vec)}$$"]:::op
  DM --> MUL
  MUL --> OUT["$$\delta M\,\ddot q\in\mathbb{R}^5$$"]:::learn

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 5 — Coriolis correction (three constructions)

```mermaid
flowchart TD
  Z["$$z_C\in\mathbb{R}^{40}:\ [\,\tilde q,\sin q_{\text{raw}},\cos q_{\text{raw}},\dot q,\{\dot q_j\dot q_k\},\tilde\tau_C\,]$$"]:::op
  QD["$$\dot q$$"]:::io

  subgraph EW["(a) Element-wise (reference)"]
    Z --> MLPa["$$\mathrm{MLP}_C\,[48,48]\to\mathbb{R}^5$$"]:::learn
    MLPa --> HAD["$$\delta C\,\dot q=\dot q\odot\mathrm{MLP}_C(z_C)$$"]:::op
    QD --> HAD
  end

  subgraph MAT["(b) Matrix form"]
    Z --> MLPb["$$\mathrm{MLP}_C\to\mathbb{R}^{25}$$"]:::learn
    MLPb --> RES["$$\text{reshape}\to B(q,\dot q)\in\mathbb{R}^{5\times5}$$"]:::op
    RES --> BMV["$$\delta C\,\dot q=B\,\dot q\quad(B\cdot 0=0)$$"]:::op
    QD --> BMV
  end

  subgraph ST["(c) Christoffel (passive)"]
    DMref["$$\delta M(q)\;\; \text{from inertia net}$$"]:::learn --> JAC["$$J=\partial\delta M/\partial q\;\; (\text{vmap}\circ\text{jacrev})$$"]:::op
    JAC --> CHR["$$\delta c_{ijk}=\tfrac12\big(\partial_k\delta M_{ij}+\partial_j\delta M_{ik}-\partial_i\delta M_{jk}\big)$$"]:::op
    CHR --> QUAD["$$(\delta C\,\dot q)_i=\textstyle\sum_{jk}\delta c_{ijk}\,\dot q_j\dot q_k$$"]:::op
  end

  HAD --> OUT["$$\delta C\,\dot q\in\mathbb{R}^5\quad(\to 0\text{ as }\dot q\to 0)$$"]:::learn
  BMV --> OUT
  QUAD --> OUT

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef phys fill:#e7edf9,stroke:#3b5b92,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 6 — Friction correction (odd by construction)

```mermaid
flowchart TD
  QD["$$\dot q$$"]:::io --> ABS["$$|\dot q|_\varepsilon=\sqrt{\dot q^2+\varepsilon}\;\; \text{(even)}$$"]:::op
  QDD["$$\ddot q\;\; \text{(optional)}$$"]:::io --> ABS2["$$|\ddot q|_\varepsilon$$"]:::op
  TFC["$$\tilde\tau_f\;\; \text{(phys. cond.)}$$"]:::phys --> ABS3["$$|\tilde\tau_f|_\varepsilon$$"]:::op
  ABS --> CAT["$$\text{concat magnitudes}\to\mathbb{R}^{15}$$"]:::op
  ABS2 --> CAT
  ABS3 --> CAT

  subgraph MLPf["(a) MLP form (reference)"]
    CAT --> H["$$h_\phi(\cdot)\,[24,24]\to\mathbb{R}^5\;\; \text{(even)}$$"]:::learn
    H --> PROD["$$\delta\tau_f=\dot q\odot h_\phi\quad(\text{odd}\odot\text{even}=\text{odd})$$"]:::op
    QD --> PROD
  end

  subgraph STR["(b) Stribeck form"]
    CAT --> P4["$$\mathrm{MLP}\to F_c,F_s,v_s,F_v\;\; (4n)$$"]:::learn
    P4 --> SB["$$\delta\tau_f=\mathrm{sgn}_s(\dot q)\odot[\,F_c+F_s\,e^{-(|\dot q|/v_s)^2}\,]+F_v\dot q$$"]:::op
    QD --> SB
  end

  PROD --> OUT["$$\delta\tau_f\in\mathbb{R}^5,\quad \delta\tau_f(-\dot q)=-\delta\tau_f(\dot q)$$"]:::learn
  SB --> OUT

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef phys fill:#e7edf9,stroke:#3b5b92,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 7 — Smooth capacity gate (curriculum)

```mermaid
flowchart LR
  E["$$\text{epoch }e,\ \text{budget }E,\ \rho=0.30$$"]:::io --> COS["$$\gamma(e)=\tfrac12\big(1-\cos\tfrac{\pi(e-1)}{\lceil\rho E\rceil}\big),\ e\le\lceil\rho E\rceil;\ \ \gamma=1\ \text{else}$$"]:::op
  COS --> G["$$\gamma\in[0,1]$$"]:::op
  G --> APPLY["$$\gamma\;\text{scales }\delta M\ddot q,\ \delta C\dot q\;\text{only (gravity, friction full)}$$"]:::op
  APPLY --> INF["$$\text{inference / after ramp:}\;\; \gamma=1$$"]:::io

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```

---

## Part 8 — Training objective & optimisation

```mermaid
flowchart TD
  THAT["$$\hat{\tilde\tau}\;\; \text{(prediction)}$$"]:::io --> LD["$$\mathcal{L}_{\text{data}}=\tfrac{1}{Bn}\textstyle\sum_{b,i} w_i(\hat{\tilde\tau}_i-\tilde\tau_i)^2$$"]:::op
  TGT["$$\tilde\tau\;\; \text{(target)}$$"]:::io --> LD

  DG["$$\delta g$$"]:::learn --> RG["$$\mathcal{L}^{(g)}_{\text{corr}}=\overline{\lVert\delta g\rVert^2}$$"]:::op
  DM["$$\delta M$$"]:::learn --> RM["$$\mathcal{L}^{(M)}_{\text{corr}}=\overline{\lVert\delta M\rVert_F^2}/n$$"]:::op
  DC["$$\delta C\,\dot q$$"]:::learn --> RC["$$\mathcal{L}^{(C)}_{\text{corr}}=\overline{\lVert\delta C\,\dot q\rVert^2}$$"]:::op
  DF["$$\delta\tau_f$$"]:::learn --> RF["$$\mathcal{L}^{(f)}_{\text{corr}}=\overline{\lVert\delta\tau_f\rVert^2}$$"]:::op

  LD --> SUM["$$\mathcal{L}=\mathcal{L}_{\text{data}}+\textstyle\sum_x\lambda_x\,\mathcal{L}^{(x)}_{\text{corr}}+\lambda_{\text{stab}}\mathcal{L}_{\text{stab}}$$"]:::io
  RG --> SUM
  RM --> SUM
  RC --> SUM
  RF --> SUM
  SUM --> OPT["AdamW (single group) · warm-up cosine LR<br/>grad-clip 1.0 · float32 (no AMP) · weight EMA<br/>early stop on val macro-RMSE"]:::op

  classDef io fill:#e8f4ea,stroke:#3a7d44,color:#000;
  classDef learn fill:#fdeaea,stroke:#b03a3a,color:#000;
  classDef op fill:#ffffff,stroke:#888888,color:#000;
```
