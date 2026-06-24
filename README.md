# PDSVM: Penalized Deep Support Vector Machine

Simultaneous nonlinear classification and variable selection via a penalized deep support vector machine. PDSVM couples a deep network with a convolution-smoothed hinge loss and a group-sparse penalty on the first layer, so that classification and feature selection are performed in a single unified framework.

This repository contains the R implementation and the scripts that reproduce the simulations and real-data analyses in the paper.

---

## Idea in one line

Replace the linear score of a soft-margin SVM with a deep network *f*(**x**; **W**), smooth the hinge loss so the objective is differentiable, and place an (adaptive) group lasso on the input-layer weight columns so that entire input features are driven to exactly zero.

---

## Model

A multilayer perceptron with *L* − 1 hidden layers maps **x** ∈ ℝ<sup>*p*</sup> to a scalar score

```math
\mathbf{h}^{(\ell)}(\mathbf{x}) = \sigma_\ell\left(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}(\mathbf{x}) + \mathbf{b}^{(\ell)}\right),
\qquad
f(\mathbf{x};\mathcal{W}) = \left(\mathbf{w}^{(L)}\right)^{\top}\mathbf{h}^{(L-1)}(\mathbf{x}) + b^{(L)},
```

with **h**<sup>(0)</sup> = **x** and **W** = { **W**<sup>(ℓ)</sup>, **b**<sup>(ℓ)</sup> }. The activation σ<sub>ℓ</sub> is twice continuously differentiable with bounded first and second derivatives (e.g. the hyperbolic tangent). The predicted label is sign *f*(**x**; **W**).

The first-layer weight matrix is grouped by input feature: **w**<sup>(1)</sup><sub>·,j</sub> is the column of **W**<sup>(1)</sup> connecting feature *x*<sub>*j*</sub> to the first hidden layer. Sending **w**<sup>(1)</sup><sub>·,j</sub> = **0** removes *x*<sub>*j*</sub> from the model entirely.

---

## Smoothed hinge loss

The hinge loss φ<sub>H</sub>(*u*) = (1 − *u*)<sub>+</sub> is nondifferentiable at *u* = 1. Convolving it with a uniform kernel *K*<sub>h</sub>(*v*) = (1 / 2*h*) · **1**(|*v*| ≤ *h*) gives a closed-form smoothed surrogate with bandwidth *h* > 0:

```math
\phi_{H,h}(u) =
\begin{cases}
1 - u, & u \le 1-h, \\
\dfrac{(1-u+h)^2}{4h}, & |u-1| < h, \\
0, & u \ge 1+h,
\end{cases}
\qquad
\phi_{H,h}'(u) =
\begin{cases}
-1, & u \le 1-h, \\
-\dfrac{1-u+h}{2h}, & |u-1| < h, \\
0, & u \ge 1+h.
\end{cases}
```

It is 1-Lipschitz, convex, and recovers the hinge as *h* → 0, which makes the whole objective amenable to gradient-based training.

---

## Penalized objective

PDSVM minimizes the smoothed empirical risk plus a first-layer group penalty and an upper-layer ridge penalty:

```math
Q_n(\mathcal{W}) =
\frac{1}{n}\sum_{i=1}^{n}\phi_{H,h}\left(y_i\, f(\mathbf{x}_i;\mathcal{W})\right)
+ \lambda_1 \sum_{j=1}^{p} \hat{w}_j\left\|\mathbf{w}^{(1)}_{\cdot,j}\right\|_2
+ \lambda_2 \sum_{\ell=2}^{L}\left\|\mathbf{W}^{(\ell)}\right\|_F^{2}.
```

The first term is the smooth data-fit *F*(**W**), and the remaining two terms are the penalties.

- λ₁ controls feature sparsity through the group lasso on first-layer columns.
- λ₂ is a ridge on the upper layers for stability.
- The adaptive weights ŵ<sub>*j*</sub> = ( ‖pilot column *j*‖<sub>2</sub> + ε<sub>n</sub> )<sup>−γ</sup> are computed from an unpenalized pilot fit (with γ = 1). Setting ŵ<sub>*j*</sub> ≡ 1 recovers the plain group lasso.

---

## Optimization: proximal gradient descent

The objective splits into a smooth part *F* and a nonsmooth separable penalty. Proximal gradient descent alternates a gradient step on *F* with a closed-form proximal step on the first-layer columns:

```math
\mathcal{W}^{t+1/2} = \mathcal{W}^{t} - \eta\, \nabla F(\mathcal{W}^{t}),
\qquad
\mathbf{w}^{(1),\, t+1}_{\cdot,j} = \mathcal{S}_{\eta \lambda_1 \hat{w}_j}\left(\mathbf{w}^{(1),\, t+1/2}_{\cdot,j}\right),
```

where the group-soft-thresholding (block) operator is

```math
\mathcal{S}_{\tau}(\mathbf{u}) = \left(1 - \frac{\tau}{\|\mathbf{u}\|_2}\right)_{+}\mathbf{u}.
```

The proximal step yields **exact zeros**, so selection is read directly from the fitted weights: ‖**w**<sup>(1)</sup><sub>·,j</sub>‖<sub>2</sub> = 0 means feature *j* is excluded, with no post-hoc thresholding required.

---

## Theory (informal)

Let *R*(*f*) = ℙ( *Y* ≠ sign *f*(**X**) ) be the misclassification risk and *R*\* its infimum. For an approximate minimizer with optimization tolerance ε<sub>opt,n</sub>, with probability at least 1 − δ,

```math
\mathcal{R}(\widehat{f}) - \mathcal{R}^{\ast}
\le
C\left[\mathrm{Rad}_n(\mathcal{F}_{\mathrm{NN}}) + B_{\Omega,h}\sqrt{\frac{\log(1/\delta)}{n}}\right]
+ \frac{h}{2} + \widehat{\mathcal{A}}_{\mathrm{reg}} + \varepsilon_{\mathrm{opt},n}.
```

The bound separates four sources of error: the empirical-process complexity Rad<sub>n</sub>(ℱ<sub>NN</sub>) = O(*n*<sup>−1/2</sup>), the O(*h*) smoothing bias, the regularized approximation error, and the optimization tolerance ε<sub>opt,n</sub>. The smoothed hinge is Fisher consistent, so minimizing the surrogate is aligned with the Bayes rule up to the O(*h*) bias.

---

## Methods compared

| Tag | Loss | Penalty | Role |
|-----|------|---------|------|
| `KSVM` | hinge (kernel) | none | nonlinear SVM baseline |
| `D-CE` | cross-entropy | none | deep classifier baseline |
| `D-Ridge` (DSVM) | smoothed hinge | ridge only | nonsparse deep SVM |
| `SPINN` | cross-entropy | group lasso (layer 1) | sparse deep CE baseline |
| `PDSVM-GL` | smoothed hinge | group lasso (layer 1) | proposed (non-adaptive) |
| `PDSVM-AGL` | smoothed hinge | adaptive group lasso (layer 1) | **proposed** |

---

## Repository layout

```
Deep_SVM_v6.R                 # core: smoothed hinge, proximal operator,
                              #       forward/backward pass, training routines
fn_variable_selection_M1.R    # simulation: high-dimensional interaction signal (p > n)
fn_variable_selection_M4.R    # simulation: additive nonlinear model (appendix)
DSVM_table1_ex1_vf.R          # simulation: nonlinear boundary + label-noise robustness
theorems_visualization.R      # empirical illustration of the risk bound and PGD behavior
02_celeba_local_parallel.R    # real data: CelebA eyeglasses (image features)
parkinsons_pdsvm.R            # real data: Parkinson's speech features
```

---

## Quick start

```r
source("Deep_SVM_v6.R")

# X has a leading column of 1s for the bias; y in {-1, +1} or {0, 1}.
fit <- PDSVM3_val(
  W1, W2, W3, W0,
  X = cbind(1, Xtr), y = ytr,
  X_val = cbind(1, Xval), y_val = yval,
  Epoch = 3000, rate = 0.05,
  lambda1 = 0.05, lambda2 = 1e-4, h = 0.01,
  penalty_type = "prox", activation = "tanh"
)

selected <- pdsvm_selected_features(fit, sel_thresh = 1e-5)$selected
```

Hyperparameters (*h*, learning rate, λ₁, λ₂) are chosen by validation accuracy; see the simulation and real-data scripts for the full tuning pipelines.

---

## Citation

```bibtex
@article{pdsvm,
  title  = {Simultaneous Nonlinear Classification and Variable Selection
            via Penalized Deep Support Vector Machines},
  author = {Shin, Jungmin and Chung, Dongjun and Bang, Sungwan},
  year   = {2026}
}
```
