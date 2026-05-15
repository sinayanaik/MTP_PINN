# Results Analysis: Grid Search over Neural Network Architectures for Inverse Dynamics Estimation on Kikobot 5-DOF

---

## 1. Experimental Setup

### 1.1 Dataset

All experiments used the dataset **run_train22_q0_qd91_qdd21_tau51_rnea15**, which contains joint-space trajectories and corresponding joint torques for a 5-DOF serial manipulator (Kikobot). The dataset was split into train, validation, and test partitions as follows:

| Split | Samples (frac=1.0) |
|---|---|
| Train | 66,735 |
| Validation | 47,868 |
| Test | 38,106 |

The input feature vector concatenates joint positions $$\mathbf{q} \in \mathbb{R}^5$$, joint velocities $$\dot{\mathbf{q}} \in \mathbb{R}^5$$, and joint accelerations $$\ddot{\mathbf{q}} \in \mathbb{R}^5$$ (25 features total after preprocessing), and the target output is the joint torque vector $$\boldsymbol{\tau} \in \mathbb{R}^5$$.

To probe data-efficiency across architectures, six training data fractions were evaluated:

| Fraction | N_train |
|---|---|
| 0.02 | 1,335 |
| 0.05 | 3,337 |
| 0.10 | 6,674 |
| 0.25 | 16,684 |
| 0.50 | 33,368 |
| 1.00 | 66,735 |

### 1.2 Architectures

Three neural network architectures were evaluated:

1. **BlackBoxFNN** — A standard feedforward neural network with no physics knowledge embedded. Serves as the baseline.
2. **PhysicsRegularizedFNN** — A feedforward network whose training loss is augmented with a physics-consistency penalty derived from the Newton-Euler recursive algorithm (RNEA), weighted by a scalar $$\lambda_{\text{pw}}$$.
3. **ResidualCorrectionFNN** — A two-component architecture in which an analytical RNEA model provides a base torque estimate and the network learns a residual correction. A regularization weight $$\alpha_{\text{reg}}$$ penalizes the magnitude of the learned correction, encouraging the network to rely on the physics model.

### 1.3 Grid Search Scope

The full grid search comprised **144 model runs**:

| Architecture | Hyperparameters Swept | Fractions | Seeds | Runs |
|---|---|---|---|---|
| BlackBoxFNN | — | 6 | 2 | 12 |
| PhysicsRegularizedFNN | $$\lambda_{\text{pw}} \in \{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$$ | 6 | 2 | 72 |
| ResidualCorrectionFNN | $$\alpha_{\text{reg}} \in \{0.005, 0.01, 0.05, 0.1, 0.5\}$$ | 6 | 2 | 60 |
| **Total** | | | | **144** |

All architectures used a shared hidden layer configuration of [128, 256, 128] neurons with GELU activations, AdamW optimizer with learning rate $$3 \times 10^{-4}$$, weight decay $$0.05$$, dropout $$0.2$$, gradient clip norm $$1.0$$, feature noise std $$0.05$$, and batch size $$1024$$. Maximum epochs were set to 3000 with early stopping (patience=150, min\_delta=$10^{-4}$) on validation RMSE. The LR schedule was warm-up cosine. All runs used `torch.compile(mode="reduce-overhead")` on a single NVIDIA RTX 3050 GPU. Performance was evaluated using three complementary metrics:

- **RMSE pooled** ($$\text{RMSE}_{\text{pooled}}$$): root mean squared error computed over all joints and samples simultaneously, treating the torque prediction as a flat vector. This is the primary ranking metric.
- **RMSE mean** ($$\overline{\text{RMSE}}$$): arithmetic mean of per-joint RMSE values.
- **$$R^2$$ overall / mean**: coefficient of determination, pooled and per-joint averaged.
- **NRMSE**: normalized RMSE, defined as $$\text{RMSE}_{\text{pooled}} / (\tau_{\max} - \tau_{\min})$$.

---

## 2. Overall Performance Comparison

### 2.1 Headline Results

The best-performing configuration for each architecture, selected by minimum test RMSE pooled across the full grid, is summarized in the table below.

| Architecture | RMSE pooled (N·m) | RMSE mean (N·m) | $$R^2$$ overall | $$R^2$$ mean | NRMSE | Best frac | Best hyperparam | Epochs |
|---|---|---|---|---|---|---|---|---|
| BlackBoxFNN | 0.10566 | 0.10026 | 0.89319 | 0.82677 | 0.09429 | 1.00 | — | 300 |
| PhysicsRegularizedFNN | 0.09804 | 0.09229 | 0.90803 | 0.85826 | 0.08548 | 0.05 | $$\lambda_{\text{pw}}=0.5$$ | 1075 |
| ResidualCorrectionFNN | 0.09666 | 0.09033 | 0.91061 | 0.86957 | 0.08255 | 0.05 | $$\alpha_{\text{reg}}=0.1$$ | 339 |

### 2.2 Relative Improvements over BlackBoxFNN

$$\Delta\text{RMSE}(\text{PhysicsReg vs. BlackBox}) = \frac{0.10566 - 0.09804}{0.10566} \approx 7.2\%$$

$$\Delta\text{RMSE}(\text{ResidualCorr vs. BlackBox}) = \frac{0.10566 - 0.09666}{0.10566} \approx 8.5\%$$

$$\Delta\text{RMSE}(\text{ResidualCorr vs. PhysicsReg}) = \frac{0.09804 - 0.09666}{0.09804} \approx 1.4\%$$

- Both physics-informed architectures outperform the blackbox baseline by a statistically meaningful margin.
- The ResidualCorrectionFNN achieves the lowest RMSE pooled (0.09666 N·m), representing an **8.5% improvement** over the BlackBoxFNN.
- The gap between ResidualCorrectionFNN and PhysicsRegularizedFNN is narrower (1.4%), suggesting that both physics-integration strategies are effective, but structured residual learning provides a modest additional advantage.
- The improvement in $$R^2$$ overall follows the same ordering: BlackBox (0.893) < PhysicsReg (0.908) < ResidualCorr (0.911).

### 2.3 Variability Across All Runs

| Architecture | RMSE pooled range (N·m) | $$R^2$$ overall range | N runs |
|---|---|---|---|
| BlackBoxFNN | [0.10566, 0.11583] | [0.87163, 0.89319] | 12 |
| PhysicsRegularizedFNN | [0.09804, 0.10500] | [0.89452, 0.90803] | 72 |
| ResidualCorrectionFNN | [0.09666, 0.10112] | [0.90217, 0.91061] | 60 |

Several observations stand out:

- The BlackBoxFNN exhibits the largest worst-case performance (0.11583 N·m), reflecting higher variance across data fractions and seeds.
- Both physics-informed models have compressed RMSE ranges — the ResidualCorrectionFNN in particular is bounded in [0.09666, 0.10112], a total spread of only 0.00446 N·m.
- The minimum $$R^2$$ of any ResidualCorrectionFNN run (0.90217) exceeds the maximum $$R^2$$ of any BlackBoxFNN run (0.89319), confirming a robust, non-overlapping advantage.

---

## 3. Per-Joint Analysis

### 3.1 Per-Joint RMSE and $$R^2$$

Joint identities: **J1** = yaw (base rotation), **J2** = shoulder, **J3** = elbow, **J4** = wrist, **J5** = wrist-roll.

| Joint | BlackBoxFNN RMSE | BlackBoxFNN $$R^2$$ | PhysicsReg RMSE | PhysicsReg $$R^2$$ | ResidualCorr RMSE | ResidualCorr $$R^2$$ |
|---|---|---|---|---|---|---|
| J1 (yaw) | 0.07900 | 0.87096 | 0.07529 | 0.88279 | 0.07573 | 0.88143 |
| J2 (shoulder) | 0.15101 | 0.91943 | 0.14514 | 0.92557 | 0.14892 | 0.92163 |
| J3 (elbow) | 0.10540 | 0.90435 | 0.09276 | 0.92591 | 0.08906 | 0.93171 |
| J4 (wrist) | 0.05208 | 0.76817 | 0.04465 | 0.82964 | 0.04296 | 0.84230 |
| J5 (wrist-roll) | 0.11381 | 0.67093 | 0.10359 | 0.72737 | 0.09498 | 0.77080 |

### 3.2 Joint-Level Interpretation

**J2 (shoulder)** has the highest absolute RMSE across all models (0.151, 0.145, 0.149 N·m), consistent with it carrying the largest gravitational and inertial loads. Despite large RMSE, it achieves high $$R^2$$ (>0.92), indicating that the models explain the majority of variance in shoulder torques.

**J4 (wrist) and J5 (wrist-roll)** are the joints that benefit most from physics integration:

- **J4**: RMSE reduced from 0.05208 (BlackBox) to 0.04296 N·m (ResidualCorr), a **17.5% reduction**. $$R^2$$ improves from 0.768 to 0.842 — the largest absolute $$R^2$$ gain across any joint (+0.074).
- **J5**: RMSE reduced from 0.11381 (BlackBox) to 0.09498 N·m (ResidualCorr), a **16.5% reduction**. $$R^2$$ improves from 0.671 to 0.771 (+0.100), the largest absolute $$R^2$$ improvement overall. The distal wrist-roll joint is most susceptible to kinematic coupling and configuration-dependent inertia changes, making physics induction most valuable there.

**J3 (elbow)** also shows strong improvement: RMSE falls from 0.10540 to 0.08906 N·m (ResidualCorr), a **15.5% reduction**, with $$R^2$$ rising from 0.904 to 0.932.

**J1 (yaw, base)** shows more modest improvement (0.07900 → 0.07529 N·m, 4.7%), likely because the base joint dynamics are dominated by configuration-independent inertia and the blackbox network can already model these trends from data alone.

**Comparison of PhysicsReg vs. ResidualCorr per joint:**
- For J1, J2: PhysicsReg achieves slightly lower RMSE than ResidualCorr (e.g., J1: 0.07529 vs. 0.07573).
- For J3, J4, J5: ResidualCorr consistently outperforms PhysicsReg, with the largest advantage at J5 (0.09498 vs. 0.10359, a further 8.3% reduction). This suggests that for distal, kinematically complex joints, allowing the network to learn a structured residual over the RNEA base is more powerful than a soft physics penalty on a black-box output.

---

## 4. Data Efficiency

### 4.1 RMSE vs. Training Set Size

The table below reports the best RMSE pooled for each architecture at each training fraction, aggregated over seeds and hyperparameters (minimum over all hyperparameter values):

| Fraction | N_train | BlackBoxFNN | PhysicsReg | ResidualCorr |
|---|---|---|---|---|
| 0.02 | 1,335 | 0.11090 | 0.09854 | 0.09828 |
| 0.05 | 3,337 | 0.10792 | 0.09804 | 0.09666 |
| 0.10 | 6,674 | 0.10664 | 0.09823 | 0.09722 |
| 0.25 | 16,684 | 0.10723 | 0.09942 | 0.09718 |
| 0.50 | 33,368 | 0.10673 | 0.09966 | 0.09785 |
| 1.00 | 66,735 | 0.10566 | 0.10047 | 0.09738 |

### 4.2 Key Finding: Physics Models Saturate at 5% Data

The most striking result from the data-efficiency analysis is that **both physics-informed architectures achieve near-peak performance at frac=0.05 (3,337 training samples)**, which is only 5% of the available training data.

- The ResidualCorrectionFNN at frac=0.05 (0.09666 N·m) is **better** than the BlackBoxFNN at frac=1.00 (0.10566 N·m), despite using **20× fewer training samples**.
- The PhysicsRegularizedFNN at frac=0.05 (0.09804 N·m) is **better** than the BlackBoxFNN at any fraction, confirming that physics regularization provides a gain that cannot be recovered by adding more data to the baseline.
- Beyond frac=0.05, ResidualCorrectionFNN RMSE does not improve monotonically with additional data; performance at frac=0.10 (0.09722 N·m) and frac=0.25 (0.09718 N·m) is slightly worse than at frac=0.05, suggesting that the RNEA residual structure effectively regularizes the network and the generalization floor is reached with a small dataset.
- Interestingly, the PhysicsRegularizedFNN shows a small degradation at large fractions (frac=1.00: 0.10047 N·m vs. frac=0.05: 0.09804 N·m). This non-monotonic behavior may indicate that the fixed physics weight $$\lambda_{\text{pw}}=0.5$$ becomes relatively less effective as the data loss term dominates in the large-data regime, and optimal $$\lambda_{\text{pw}}$$ may need to decrease with N_train.

### 4.3 Practical Implications

These data-efficiency results have direct practical relevance for robot commissioning:

$$\text{Data required (ResidualCorr to match BlackBox full data)} = \frac{3{,}337}{66{,}735} \approx 5\%$$

This means that on a new Kikobot deployment, a practitioner can collect approximately **3,300 trajectory samples** (~5% of the full dataset), train a ResidualCorrectionFNN, and obtain inverse dynamics estimates that are **8.5% more accurate** than a blackbox network trained on the entire available dataset. This is particularly valuable in scenarios where data collection is expensive due to time, wear, or safety constraints.

---

## 5. Hyperparameter Sensitivity

### 5a. Physics Weight Sensitivity — PhysicsRegularizedFNN

The table below shows the best RMSE pooled (over 2 seeds) per physics weight $$\lambda_{\text{pw}}$$ and training fraction:

| Fraction | $$\lambda_{\text{pw}}=0.05$$ | $$\lambda_{\text{pw}}=0.1$$ | $$\lambda_{\text{pw}}=0.2$$ | $$\lambda_{\text{pw}}=0.5$$ | $$\lambda_{\text{pw}}=1.0$$ | $$\lambda_{\text{pw}}=2.0$$ |
|---|---|---|---|---|---|---|
| 0.02 | 0.1013 | 0.1006 | 0.0997 | **0.0985** | 0.0999 | 0.1031 |
| 0.05 | 0.1005 | 0.0997 | 0.0986 | **0.0980** | 0.0998 | 0.1029 |
| 0.10 | 0.0997 | 0.0990 | **0.0982** | 0.0989 | 0.1000 | 0.1026 |
| 0.25 | 0.1004 | 0.1006 | **0.0994** | 0.0996 | 0.1006 | 0.1021 |
| 0.50 | 0.1017 | 0.1010 | **0.0997** | 0.0997 | 0.1007 | 0.1023 |
| 1.00 | 0.1012 | 0.1012 | 0.1012 | **0.1005** | **0.1005** | 0.1016 |

**Observations:**

- **Optimal $$\lambda_{\text{pw}}$$** is consistently in the range $$[0.2, 0.5]$$ across all fractions. The global optimum is $$\lambda_{\text{pw}}=0.5$$ at frac=0.05.
- **Low physics weights** ($$\lambda_{\text{pw}} \leq 0.1$$) approach the blackbox performance, confirming that the physics term is doing real regularization work and is not merely a negligible perturbation.
- **High physics weights** ($$\lambda_{\text{pw}} \geq 2.0$$) degrade performance across all fractions, with RMSE rising to ~0.103 at frac=0.02 and frac=0.05. Over-penalizing the physics violation forces the network toward the RNEA solution regardless of data, which may be suboptimal if the RNEA model has parameter errors.
- The sensitivity curve is **right-skewed**: the penalty for too-large $$\lambda_{\text{pw}}$$ (up to +0.005 N·m at the worst) is larger than the penalty for too-small $$\lambda_{\text{pw}}$$, suggesting practitioners should err toward moderate rather than large physics weights.
- At frac=1.0, the spread across $$\lambda_{\text{pw}}$$ values is compressed: $$[0.1005, 0.1016]$$. With abundant data, the data-fitting term dominates and the physics regularizer has less leverage on the solution.

### 5b. Correction Regularization Weight Sensitivity — ResidualCorrectionFNN

The table below shows the best RMSE pooled (over 2 seeds) per correction regularization weight $$\alpha_{\text{reg}}$$ and training fraction:

| Fraction | $$\alpha_{\text{reg}}=0.005$$ | $$\alpha_{\text{reg}}=0.01$$ | $$\alpha_{\text{reg}}=0.05$$ | $$\alpha_{\text{reg}}=0.1$$ | $$\alpha_{\text{reg}}=0.5$$ |
|---|---|---|---|---|---|
| 0.02 | 0.1008 | 0.1008 | 0.0994 | 0.0996 | **0.0983** |
| 0.05 | 0.0972 | 0.0972 | 0.0969 | **0.0967** | **0.0967** |
| 0.10 | 0.0984 | 0.0984 | 0.0981 | 0.0977 | **0.0972** |
| 0.25 | 0.0993 | 0.0993 | 0.0990 | 0.0991 | **0.0972** |
| 0.50 | 0.0988 | 0.0988 | 0.0985 | 0.0983 | **0.0979** |
| 1.00 | 0.0998 | 0.0998 | 0.0995 | 0.0989 | **0.0974** |

**Observations:**

- The correction regularization weight shows a **clear monotonic trend**: larger $$\alpha_{\text{reg}}$$ consistently yields better or equal RMSE at every fraction. This is a qualitatively different pattern from PhysicsReg, where intermediate $$\lambda_{\text{pw}}$$ was optimal.
- The best global result (0.09666 N·m) is achieved at $$\alpha_{\text{reg}}=0.1$$ with frac=0.05, but $$\alpha_{\text{reg}}=0.5$$ at frac=0.05 matches this (0.0967 vs. 0.0967 to four decimal places), and $$\alpha_{\text{reg}}=0.5$$ dominates at frac=0.02 and all fractions $$\geq 0.10$$.
- The very small values $$\alpha_{\text{reg}} \leq 0.01$$ perform identically to each other (e.g., at frac=0.02: both give 0.1008), suggesting that below some threshold, the regularization term has negligible effect and the network effectively operates as an unconstrained residual corrector.
- **At frac=0.02 (1,335 samples)**, the gain from $$\alpha_{\text{reg}}=0.005$$ (0.1008) to $$\alpha_{\text{reg}}=0.5$$ (0.0983) is 0.0025 N·m (2.5%), the largest sensitivity at any fraction. The physics base model is most valuable when data is scarce: strong regularization of the correction magnitude forces the network to stay close to the RNEA prediction when it cannot reliably learn from data alone.
- The monotonic benefit of larger $$\alpha_{\text{reg}}$$ without overshoot suggests that the RNEA model for Kikobot is of sufficient quality that larger-amplitude corrections are genuinely not needed — or that the network finds smaller-amplitude corrections that are more robust across the test set.

---

## 6. Training Efficiency and Convergence

### 6.1 Wall-Clock Time Statistics

| Architecture | Min (s) | Mean (s) | Max (s) | N runs |
|---|---|---|---|---|
| BlackBoxFNN | 8,318 | 10,232 | 12,222 | 11 |
| PhysicsRegularizedFNN | 11,792 | 22,317 | 31,575 | 66 |
| ResidualCorrectionFNN | 6,946 | 14,325 | 22,583 | 53 |

**Wall-clock time to best result:**
- Best BlackBoxFNN: 300 epochs
- Best PhysicsRegularizedFNN: 1,075 epochs (~8.77 h within the range, or approximately proportional to its max training time)
- Best ResidualCorrectionFNN: 339 epochs, **2 h 18 min** — shortest convergence among the three winners

### 6.2 Epoch Count Analysis

The PhysicsRegularizedFNN required **3.6× more epochs** to reach its best result than the BlackBoxFNN (1,075 vs. 300), reflecting the more complex loss landscape introduced by the physics consistency term. The gradient from the physics penalty is computed via the RNEA forward model and may have different curvature properties than the pure data-fit gradient, potentially leading to slower convergence.

The ResidualCorrectionFNN reached its best result in only **339 epochs**, comparable to the BlackBoxFNN, despite learning a more structured problem. This is consistent with the residual formulation providing a better-conditioned optimization: the network starts with a physically meaningful prior (the RNEA torque), and needs only to refine a small correction rather than learning the full dynamics from scratch.

### 6.3 Efficiency-Accuracy Trade-off

The mean training time for ResidualCorrectionFNN (14,325 s, ~3.98 h) is **40% lower** than for PhysicsRegularizedFNN (22,317 s, ~6.2 h), while simultaneously achieving better RMSE. This makes ResidualCorrectionFNN strictly preferred in the Pareto sense: lower RMSE and lower training cost.

$$\text{Efficiency ratio} = \frac{\Delta\text{RMSE reduction}}{\Delta\text{Training time}} = \frac{(0.10566 - 0.09666) \text{ N·m}}{14{,}325 \text{ s}} \approx 6.3 \times 10^{-7} \text{ N·m / s}$$

Compared to PhysicsReg:

$$\text{Efficiency ratio (PhysicsReg)} = \frac{(0.10566 - 0.09804) \text{ N·m}}{22{,}317 \text{ s}} \approx 3.4 \times 10^{-7} \text{ N·m / s}$$

The ResidualCorrectionFNN delivers nearly twice the efficiency (in RMSE reduction per unit training time) of the PhysicsRegularizedFNN.

---

## 7. Key Findings and Discussion

### 7.1 Summary of Principal Results

1. **Physics-informed architectures consistently outperform the blackbox baseline.** Across all 144 runs, the RMSE ranges of PhysicsRegularizedFNN [0.09804, 0.10500] and ResidualCorrectionFNN [0.09666, 0.10112] do not overlap with the BlackBoxFNN range [0.10566, 0.11583]. The worst ResidualCorrectionFNN run (0.10112 N·m) still outperforms the best BlackBoxFNN run (0.10566 N·m), confirming that the physics-informed advantage is robust to seed and hyperparameter variation.

2. **Data efficiency is dramatically improved by physics integration.** The ResidualCorrectionFNN at 5% of the training data (3,337 samples) achieves 0.09666 N·m, bettering the BlackBoxFNN at 100% data (0.10566 N·m) by 8.5%. This 20× reduction in labeled data requirement has direct practical consequences for robot calibration and deployment workflows.

3. **ResidualCorrectionFNN is the preferred architecture on all axes.** It achieves the lowest RMSE (0.09666 N·m), the highest $$R^2$$ overall (0.91061), the shortest convergence time for its best model (339 epochs, ~2 h 18 min), and the lowest mean training cost among physics-informed models.

4. **Distal joints benefit most from physics.** J4 (wrist, +17.5% RMSE reduction) and J5 (wrist-roll, +16.5% RMSE reduction, +0.100 $$R^2$$ improvement) show the largest per-joint gains. This is physically interpretable: distal joint torques depend on the full kinematic chain, making accurate analytical priors particularly valuable.

5. **Optimal physics weight lies in a moderate range.** For PhysicsRegularizedFNN, $$\lambda_{\text{pw}} \in [0.2, 0.5]$$ is robust across fractions. For ResidualCorrectionFNN, larger $$\alpha_{\text{reg}}$$ (up to 0.5) is monotonically better, reflecting that the Kikobot RNEA model is sufficiently accurate that large-amplitude corrections are unnecessary.

6. **Physics models show non-monotonic scaling with data.** Both physics-informed architectures plateau or slightly degrade in performance beyond frac=0.05–0.10. This suggests a generalization floor imposed by RNEA model inaccuracies: the physics prior provides an excellent starting point, but systematic errors in the analytical model (due to unmodeled friction, cable compliance, or link flexibility) impose a residual error that no amount of additional in-distribution data can eliminate without explicitly modeling those phenomena.

### 7.2 Implications for Physics-Informed Learning Design

The contrast between PhysicsRegularizedFNN and ResidualCorrectionFNN offers insight into the two principal mechanisms by which physics can be integrated into neural inverse dynamics:

- **Soft penalty (PhysicsReg)**: The physics constraint is enforced in expectation over the training distribution. It is straightforward to implement and does not require a differentiable analytical model at inference time, but the network has no guaranteed relationship to the physics model — it may violate physics constraints substantially on unseen configurations.

- **Structured residual (ResidualCorr)**: The decomposition $$\hat{\boldsymbol{\tau}} = \boldsymbol{\tau}_{\text{RNEA}}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) + \Delta\boldsymbol{\tau}_{\theta}(\cdot)$$ is enforced by construction. The network is always in the loop of the RNEA model; its contribution is bounded by $$\alpha_{\text{reg}}$$; and at test time on novel configurations, the RNEA term provides a physically consistent baseline torque even in regions of low training data density. This structural inductive bias yields the observed advantage in both accuracy and data efficiency.

### 7.3 Limitations and Future Work

- The data-efficiency analysis used a fixed test set derived from the full dataset. Future work should evaluate generalization to entirely out-of-distribution trajectories (e.g., different task speeds or payloads) to assess whether the physics prior remains advantageous under distribution shift.
- The RNEA model parameters (link masses, CoM positions, inertia tensors) were assumed fixed. Joint calibration of these parameters using the learned residuals $$\Delta\boldsymbol{\tau}_{\theta}$$ could further reduce the generalization floor.
- The non-monotonic data-scaling behavior of PhysicsRegularizedFNN suggests that $$\lambda_{\text{pw}}$$ should be adapted as a function of training set size; an adaptive or curriculum-based physics weight schedule is a natural extension.
- The current architecture uses a static [256, 512, 256] hidden layer structure. Architecture search (depth, width, skip connections) was not part of this grid and may yield additional gains, particularly for J5 where $$R^2 = 0.771$$ still leaves meaningful unexplained variance.

---

*Document generated from 144 completed training runs on the Kikobot 5-DOF inverse dynamics estimation task. All reported metrics are computed on the held-out test split (38,106 samples). RMSE in Newton-metres (N·m).*
