# Research Plan: Robustness in Information Bottleneck Systems

This document contains both the scientific plan and the implementation
specification. Sections 1--16 describe the motivation and experiments. Section
17 onward fixes the defaults needed to implement the study reproducibly. When
an earlier "suggested" setting is less specific than the implementation
specification, the implementation specification takes precedence. Any change
to a fixed default must be recorded in the run configuration and experiment
log.

## 1. Research Objective

The overarching research question is:

> **How does restricting the information capacity of a learned representation affect its robustness, and can the relationship be explained by changes in representation geometry, invariance, and semantic separation?**

The working conceptual hypothesis is:

\[
\text{Bottleneck Strength}
\rightarrow
\text{Representation Geometry}
\rightarrow
\begin{cases}
\text{Local invariance}\\
\text{Semantic separation}
\end{cases}
\rightarrow
\text{Robustness}.
\]

We do **not** assume that stronger compression necessarily reduces robustness.

A plausible expectation is that moderate compression may remove nuisance variation and therefore improve some forms of robustness, while aggressive compression may begin removing task-relevant information and reduce semantic margins. This could lead to a non-monotonic relationship:

\[
\text{weak bottleneck}
\rightarrow
\text{moderate bottleneck}
\rightarrow
\text{strong bottleneck}
\]

\[
\text{low/normal robustness}
\rightarrow
\text{improved invariance}
\rightarrow
\text{semantic collapse}.
\]

These are **expected outcomes to test**, not desired outcomes. A monotonic relationship, no relationship, or a bottleneck-family-dependent relationship would all be scientifically valid results.

---

# 2. Bottleneck Model Families

The main principle is to keep the task, dataset, and backbone as similar as possible while changing **how information is restricted**.

| # | Bottleneck | Compression control | Main purpose |
|---|---|---|---|
| 1 | Deterministic dimensional | \(d_z\) | Cleanest controlled experiment |
| 2 | Variational Information Bottleneck | \(\beta\) | Restrict information rather than dimension |
| 3 | Vector-quantized / discrete | Codebook \(K\) | Finite/discrete representation |
| 4 | Quantized continuous | Bits per latent element | Precision-based compression |
| 5 | Learned autoencoder | Latent dimension/capacity | Reconstruction-oriented validation |

## 2.1 Model 1 — Deterministic Dimensional Bottleneck

Use an ImageNet-style ResNet-18 backbone trained on Imagenette.

Architecture:

\[
x
\rightarrow
\text{ResNet-18 backbone}
\rightarrow
h\in\mathbb R^{512}
\rightarrow
W_b
\rightarrow
z\in\mathbb R^{d_z}
\rightarrow
\text{classifier}.
\]

Sweep:

\[
d_z\in\{512,256,128,64,32,16\}.
\]

The learned 512-dimensional version acts as the weakest bottleneck in the
primary sweep. A separate identity-projection control, specified in Section
17.4, determines whether it is defensible to describe this as a no-bottleneck
baseline.

A simple PyTorch structure would be:

```python
class DimensionalBottleneck(nn.Module):
    def __init__(self, backbone, dz, num_classes=10):
        super().__init__()
        self.encoder = backbone
        self.bottleneck = nn.Linear(512, dz)
        self.classifier = nn.Linear(dz, num_classes)

    def encode(self, x):
        h = self.encoder(x)
        return self.bottleneck(h)

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        logits = self.classifier(z)

        if return_latent:
            return logits, z

        return logits
```

This is the primary model for establishing causality because almost everything except \(d_z\) remains fixed.

## 2.2 Model 2 — Variational Information Bottleneck

Keep the latent dimensionality fixed, initially at:

\[
d_z=128,
\]

but control information using a KL penalty.

The encoder predicts:

\[
\mu(x), \log\sigma^2(x)
\]

and samples:

\[
z=\mu+\sigma\epsilon,
\qquad
\epsilon\sim \mathcal N(0,I).
\]

Train using:

\[
\mathcal L
=
\mathcal L_{\rm CE}
+
\beta
D_{\rm KL}\left(q(z|x)\Vert\mathcal N(0,I)\right).
\]

Suggested initial sweep:

\[
\beta\in
\{0,\;10^{-4},\;3\times10^{-4},
10^{-3},\;3\times10^{-3},\;10^{-2}\}.
\]

Implementation:

```python
mu = self.fc_mu(h)
logvar = self.fc_logvar(h)

std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)

z = mu + std * eps
logits = self.classifier(z)

kl = -0.5 * (
    1 + logvar - mu.pow(2) - logvar.exp()
).sum(dim=1).mean()

loss = cross_entropy(logits, y) + beta * kl
```

This model is crucial because it separates:

> reducing dimensionality

from

> reducing the amount of information the representation retains.

Because this model is stochastic, adversarial evaluation must use Expectation over Transformation (EoT) rather than treating randomness as genuine robustness.

## 2.3 Model 3 — Vector-Quantized / Discrete Bottleneck

Use the same 512-dimensional ResNet representation and divide/project it into tokens.

For example:

\[
h\in\mathbb R^{512}
\rightarrow
16\times32
\]

giving 16 latent tokens of dimension 32.

Each token is replaced with its nearest codebook vector:

\[
Q(z_i)
=
c_{\arg\min_k \|z_i-c_k\|}.
\]

Keep the number of tokens fixed and vary:

\[
K\in\{512,256,128,64,32,16\}.
\]

The nominal representation capacity becomes approximately:

\[
C=16\log_2K
\]

bits.

Use the standard straight-through estimator:

```python
z_e = self.projection(h).view(B, 16, 32)

# nearest codebook vector
dist = ((z_e.unsqueeze(2) -
         codebook.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)

indices = dist.argmin(dim=-1)
z_q = codebook[indices]

# straight-through estimator
z_st = z_e + (z_q - z_e).detach()
```

Include codebook and commitment losses.

This family will be especially important for the representation-collision experiment because actual discrete equality can be measured:

\[
Q(E(x_i))=Q(E(x_j)).
\]

## 2.4 Model 4 — Quantized Continuous Bottleneck

Take the deterministic model and fix, for example:

\[
d_z=128.
\]

Then quantize each latent element to \(b\) bits.

Suggested settings:

\[
b\in\{\text{FP32},8,6,4,3,2\}.
\]

One implementation is to constrain the representation to \([-1,1]\):

\[
z'=\tanh(z)
\]

and uniformly quantize:

\[
Q_b(z')
=
2
\frac{
\operatorname{round}
[(z'+1)(2^b-1)/2]
}
{2^b-1}
-1.
\]

Use a straight-through gradient during training:

```python
z = torch.tanh(z)

levels = 2 ** bits - 1

z_q = torch.round((z + 1) * levels / 2)
z_q = 2 * z_q / levels - 1

z_st = z + (z_q - z).detach()
```

This directly tests whether **precision reduction**, rather than dimensional reduction, produces similar robustness behavior.

## 2.5 Model 5 — Learned Autoencoder Bottleneck

This provides a model whose purpose is reconstruction rather than classification.

Use:

\[
x
\rightarrow E(x)
\rightarrow z
\rightarrow D(z)
\rightarrow \hat{x}.
\]

Suggested latent dimensions:

\[
d_z\in\{512,256,128,64,32\}.
\]

Train the autoencoder using reconstruction loss, initially:

\[
\mathcal L_{\rm AE}
=
\|x-\hat{x}\|_2^2.
\]

A fixed independently trained ResNet-18 classifier then evaluates:

\[
h(\hat{x}).
\]

This setup is important because it asks whether the phenomenon persists when the bottleneck was **not explicitly optimized for classification**.

It also gives access to both:

- reconstruction quality;
- semantic/task robustness.

---

# 3. Common Dataset and Training Setup

The primary dataset for the initial study will be **Imagenette**.

Imagenette provides natural ImageNet-derived images with substantially greater visual complexity and resolution than CIFAR-10 while retaining only 10 classes, making it suitable for controlled bottleneck experiments without the computational expense of full ImageNet.

## 3.1 Dataset Setup

Start from the standard Imagenette training and validation splits. Section
17.3 defines a fixed tuning subset of the official training data so the
official validation split can be retained for final evaluation.

Images should be processed at approximately:

\[
224\times224
\]

resolution.

Suggested training preprocessing:

- resize the shorter side to 256 pixels;
- random resized crop to \(224\times224\);
- random horizontal flip;
- standard ImageNet normalization.

Suggested validation preprocessing:

- resize the shorter side to 256 pixels;
- center crop to \(224\times224\);
- standard ImageNet normalization.

The same development-train, development-tuning, and final-evaluation manifests
and preprocessing must be used across all bottleneck model families.

## 3.2 Training Protocol

Use **three independent training seeds** for every reported configuration.

The fixed Phase 1 starting setup is:

```text
Optimizer: AdamW
Learning rate: 3e-4
Weight decay: 1e-4
Batch size: 128
Epochs: 200
Scheduler: 5-epoch warmup followed by cosine decay
Seeds: 3
```

These are controlled implementation defaults rather than scientifically
important quantities. The limited pilot and checkpoint-selection rules in
Section 17.7 must be followed before examining final-evaluation results.

Where GPU memory makes a batch size of 128 impractical for \(224\times224\) images, gradient accumulation may be used to preserve the effective batch size.

---

# 4. Common Robustness Evaluation

There should be two attack surfaces:

1. **Input-space attacks**
2. **Latent-space attacks**

## 4.1 Input-Space Attack

Solve:

\[
\max_{\|\delta\|_\infty\leq\epsilon}
\mathcal L(h(E(x+\delta)),y).
\]

Suggested perturbation sweep:

\[
\epsilon
\in
\left\{
0,\frac1{255},\frac2{255},\frac4{255},
\frac8{255},\frac{12}{255},\frac{16}{255}
\right\}.
\]

Use:

- 40 PGD iterations;
- random initialization;
- 5 restarts for final experiments.

For stochastic models use EoT-PGD.

For quantized and VQ models use BPDA/straight-through gradients and include a gradient-free attack such as Square Attack as a check against gradient masking.

This is important because increased apparent robustness from quantization may simply reflect poor gradients.

## 4.2 Latent-Space Attack

For each clean representation:

\[
z=E(x),
\]

optimize:

\[
z_{\rm adv}=z+\eta
\]

to maximize downstream loss.

Because different dimensions and model families have different latent scales, do not use a raw fixed \(\|\eta\|_2\).

Instead define:

\[
\rho=
\frac{\|\eta\|_2}{\|z\|_2}.
\]

Suggested sweep:

\[
\rho\in
\{0.01,0.02,0.05,0.10,0.20,0.30\}.
\]

Use projected gradient ascent:

```python
eta = torch.zeros_like(z).uniform_(-small, small)

for _ in range(40):
    z_adv = z + eta

    logits = model.classify_latent(z_adv)
    loss = F.cross_entropy(logits, y)

    grad = torch.autograd.grad(loss, eta)[0]

    eta = eta + step_size * normalize(grad)

    max_norm = rho * z.norm(dim=1)
    eta = project_l2(eta, max_norm)
```

For Model 5, perturb \(z\), decode it, and classify the resulting reconstruction.

---

# 5. Experiment 1 — Bottleneck Strength vs. Adversarial Robustness

## Research Question

> **Does changing bottleneck capacity systematically change adversarial robustness?**

## Primary Model

Model 1: deterministic dimensional bottleneck.

Train:

\[
d_z=\{512,256,128,64,32,16\}.
\]

## Measurements

For every model record:

- clean accuracy;
- input robust accuracy across \(\epsilon\);
- latent robust accuracy across \(\rho\);
- attack success rate (ASR) calculated only over originally correctly classified samples;
- median minimum successful perturbation;
- robust-accuracy area under the curve (AUC).

Conditional ASR is important because bottleneck strength itself may affect clean accuracy.

## Hypothesis

Bottleneck strength affects adversarial robustness rather than robustness being independent of representation capacity.

## Expected Result

The expected relationship is **not necessarily monotonic**.

A plausible outcome is that:

\[
512\rightarrow128
\]

removes some nuisance variation and improves or maintains input robustness, followed by:

\[
128\rightarrow16
\]

where task-relevant information begins to be lost and robustness deteriorates.

Latent-space robustness may begin deteriorating earlier because representation margins may become smaller.

This is only an expected outcome, not a target result.

## Main Figures

Plot:

\[
d_z \text{ vs. input robust AUC}
\]

and:

\[
d_z \text{ vs. latent robust AUC}.
\]

Also report the full robust-accuracy curves.

---

# 6. Experiment 2 — Generalization Across Bottleneck Mechanisms

## Research Question

> **Is the observed relationship specific to dimensionality reduction, or is it common to different learned information bottlenecks?**

## Models

Use all five model types.

Vary:

\[
\begin{aligned}
\text{Dimensional:}&\quad d_z\\
\text{VIB:}&\quad\beta\\
\text{VQ:}&\quad K\\
\text{Quantized:}&\quad b\\
\text{Autoencoder:}&\quad d_z.
\end{aligned}
\]

Perform the same input- and latent-robustness evaluation.

## Important Design Choice

Do **not** claim that:

\[
d_z=64
\]

is somehow equivalent to:

\[
K=64
\]

or:

\[
b=4.
\]

These bottleneck strengths represent fundamentally different quantities.

Analyze trends **within each family**.

For cross-family visualization, bottleneck strength may be normalized to an ordinal scale:

\[
0=\text{weakest bottleneck},
\qquad
1=\text{strongest bottleneck}.
\]

A stronger cross-family comparison will come from relating robustness to the representation-geometry measurements in Experiment 3.

## Hypothesis

Different ways of restricting information should produce at least some common changes in robustness if the underlying phenomenon genuinely concerns learned information bottlenecks.

## Expected Result

The exact curves are expected to differ across model types, but stronger bottlenecks may generally:

1. reduce representational variability;
2. eventually reduce semantic separation;
3. alter adversarial margins.

Discrete and quantized systems may show sharper transitions than continuous dimensional bottlenecks.

## What Would Falsify the Broad Claim?

If dimensional reduction exhibits a strong relationship but VIB, VQ, quantization, and autoencoding show no comparable behavior, then the paper should **not** claim a universal bottleneck effect.

The result would instead be mechanism-specific.

---

# 7. Experiment 3 — Representation Geometry

This is one of the most important explanatory experiments.

## Research Question

> **What geometric change in \(Z\) explains the relationship between bottleneck strength and robustness?**

For every trained model, extract \(z\) for the complete validation set.

## 7.1 Intra-Class Distance

\[
D_{\rm intra}
=
\mathbb E[
\|\bar z_i-\bar z_j\|_2
\mid y_i=y_j
],
\]

where:

\[
\bar z=\frac{z}{\|z\|_2}.
\]

## 7.2 Inter-Class Distance

\[
D_{\rm inter}
=
\mathbb E[
\|\bar z_i-\bar z_j\|_2
\mid y_i\neq y_j
].
\]

## 7.3 Separation Ratio

\[
S=
\frac{D_{\rm inter}}
{D_{\rm intra}+\epsilon}.
\]

## 7.4 Nearest Opposing-Class Distance

For every example:

\[
M_i
=
\min_{j:y_j\neq y_i}
\|\bar z_i-\bar z_j\|_2.
\]

This is particularly important because average distances can hide local class overlap.

## 7.5 Effective Dimensionality

Compute covariance:

\[
\Sigma_Z
\]

and eigenvalues \(\lambda_i\).

Normalize:

\[
p_i=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

Then compute effective rank:

\[
r_{\rm eff}
=
\exp
\left(
-\sum_i p_i\log p_i
\right).
\]

A nominal 128-dimensional representation may actually use far fewer effective dimensions.

## 7.6 Encoder Sensitivity

Estimate a Jacobian norm:

\[
J_E(x)=
\frac{\partial E(x)}{\partial x}.
\]

The complete Jacobian does not need to be formed explicitly.

Use JVP/VJP power iteration to estimate the largest singular value for perhaps 500–1,000 validation images.

## 7.7 Empirical Contraction

Measure cross-semantic ratios such as:

\[
C_{ij}
=
\frac{
\|x_i-x_j\|/\sqrt{d_x}
}{
\|z_i-z_j\|/\sqrt{d_z}
}.
\]

Report both:

- median;
- 95th or 99th percentile.

Avoid relying solely on the maximum because it may be dominated by unusual pairs.

## Hypothesis

Bottleneck strength affects robustness because it changes representation geometry.

## Expected Result

Expected trends include:

- effective rank decreases;
- intra-class distances initially decrease;
- inter-class separation remains relatively stable under mild compression;
- aggressive compression decreases inter-class and nearest-opposing-class distances;
- adversarial robustness correlates more strongly with local semantic margin than with nominal \(d_z\).

The particularly valuable result would be:

\[
\text{Bottleneck parameter}
\rightarrow
\text{semantic margin}
\rightarrow
\text{robustness}.
\]

## Statistical Analysis

Across configurations, calculate Spearman correlation between:

\[
R_{\rm adv}
\]

and each of:

\[
D_{\rm intra},
D_{\rm inter},
S,
M,
r_{\rm eff},
\|J_E\|,
C.
\]

---

# 8. Experiment 4 — Invariance Versus Semantic Separation

## Research Question

> **Does compression make the representation invariant to nuisance variation while eventually collapsing meaningful semantic variation?**

This experiment directly tests the central conceptual argument.

## 8.1 Construct Semantic-Preserving Pairs

For every validation image \(x\), construct benign variants \(T(x)\):

- small translations;
- small brightness changes;
- small contrast changes;
- mild Gaussian noise;
- horizontal flip where appropriate.

Measure:

\[
D_{\rm nuisance}
=
\mathbb E
\|
\bar E(x)-\bar E(T(x))
\|_2.
\]

Lower values indicate greater invariance.

## 8.2 Construct Semantically Different Pairs

For cross-class examples:

\[
y_i\neq y_j,
\]

measure:

\[
D_{\rm semantic}
=
\mathbb E
\|
\bar E(x_i)-\bar E(x_j)
\|_2.
\]

Also retain nearest cross-class distance from Experiment 3.

Define:

\[
Q=
\frac{
D_{\rm semantic}
}{
D_{\rm nuisance}+\epsilon
}.
\]

A useful representation should ideally have:

\[
D_{\rm nuisance}\downarrow
\]

while maintaining:

\[
D_{\rm semantic}\uparrow.
\]

## Hypothesis

Increasing bottleneck strength initially removes nuisance variation, but sufficiently strong compression begins collapsing task-relevant variation as well.

## Expected Result

A possible expected pattern is:

```text
Increasing compression

D_nuisance:  ↓ ↓ ↓ ↓
D_semantic:  ≈ ≈ ↓ ↓↓↓
```

This would mean mild compression improves invariance, while strong compression eventually damages semantic separation.

Consequently, \(Q\) may peak at an intermediate bottleneck strength.

That intermediate maximum would be scientifically interesting, but it is not something the experiment should be designed to force.

---

# 9. Experiment 5 — Semantic Representation Collisions

This is the broadest conceptual extension of the prior collision result.

## Research Question

> **As bottleneck capacity decreases, does it become easier for semantically different inputs to occupy the same or nearly the same learned representation?**

There should be two versions.

## 9.1 Natural Collision Analysis

For each validation sample, calculate its nearest representation from another class:

\[
M_i=
\min_{j:y_j\neq y_i}
d(z_i,z_j).
\]

For continuous representations define a collision threshold \(\tau\).

One defensible approach is to calibrate \(\tau\) from the empirical within-class distance distribution rather than choosing an arbitrary constant.

Measure:

\[
P[
d(z_i,z_j)<\tau
\mid y_i\neq y_j
].
\]

For VQ systems, use actual code equality:

\[
P[
Q(E(x_i))=Q(E(x_j))
\mid y_i\neq y_j
].
\]

## 9.2 Adversarially Induced Collisions

Take:

- source \(x_s\), class \(y_s\);
- target \(x_t\), class \(y_t\neq y_s\).

Optimize:

\[
\min_{x_{\rm adv}}
\|E(x_{\rm adv})-E(x_t)\|_2
\]

subject to:

\[
\|x_{\rm adv}-x_s\|_\infty
\leq\epsilon.
\]

Use an independent reference classifier \(g\) to verify:

\[
g(x_{\rm adv})=y_s.
\]

Thus, the source image remains semantically source-like while its bottleneck representation approaches the target.

Implementation outline:

```python
x_adv = x_src.clone().detach()

for _ in range(attack_steps):
    x_adv.requires_grad_(True)

    z_adv = model.encode(x_adv)
    z_target = model.encode(x_target).detach()

    latent_loss = distance(z_adv, z_target)

    ref_logits = reference_classifier(x_adv)
    semantic_loss = F.cross_entropy(ref_logits, y_src)

    loss = latent_loss + lambda_sem * semantic_loss

    grad = torch.autograd.grad(loss, x_adv)[0]

    x_adv = optimizer_step(x_adv, grad)
    x_adv = project_linf(x_adv, x_src, epsilon)
```

## 9.3 VQ-Specific Attack

For the target's code sequence:

\[
(k_1^t,\ldots,k_M^t),
\]

optimize the source's **pre-quantization vectors** toward the corresponding target codebook vectors:

\[
\mathcal L
=
\sum_m
\|z_m(x_{\rm adv})-c_{k_m^t}\|_2^2.
\]

Success requires:

\[
Q(E(x_{\rm adv}))
=
Q(E(x_t)).
\]

This gives an exact semantic representation collision.

## 9.4 Quantized-Continuous Model

Similarly optimize the source latent toward the interior of the quantization bins occupied by the target.

## Hypothesis

Lower-capacity bottlenecks decrease cross-semantic representation margins and make natural or adversarially induced collisions easier.

## Expected Result

Expected trends include:

- nearest cross-class distances shrink;
- collision success rates increase under strong bottlenecks;
- required perturbation for target matching decreases;
- discrete and low-bit quantized bottlenecks exhibit the strongest collision behavior.

---

# 10. Experiment 6 — Local Robustness Versus Semantic-Separation Robustness

This is the synthesis experiment and may become the paper's central result.

It does not require training additional models. It combines Experiments 1–5.

## Research Question

> **Are local adversarial robustness and global semantic-separation robustness complementary or competing properties as representation capacity changes?**

Define a local robustness metric such as:

\[
R_{\rm local}
=
\operatorname{AUC}
[
\text{robust accuracy}(\epsilon)
].
\]

Alternatively use:

\[
R_{\rm local}
=
\operatorname{median}\epsilon^*.
\]

Define semantic-separation robustness using:

\[
R_{\rm sep}
=
\operatorname{median}
\left[
\min_{j:y_j\neq y_i}
d(z_i,z_j)
\right].
\]

And independently:

\[
R_{\rm collision}
=
1-\text{collision ASR}.
\]

Plot:

\[
R_{\rm local}
\quad\text{vs}\quad
R_{\rm sep}
\]

for every bottleneck level and model family.

## Hypothesis

Compression may improve local invariance while simultaneously reducing the ability to maintain global semantic separation.

## Expected Result

The expected outcome is more likely to resemble a **frontier** than one metric dominating everywhere.

For example:

```text
Semantic
separation
robustness
    ^
    | *
    |   *
    |      *
    |         *
    |             *
    +--------------------> Local robustness
```

There may also be a curved relationship in which intermediate bottleneck strength provides the best compromise.

The strongest possible interpretation would be:

> Learned bottlenecks do not simply make representations "more robust" or "less robust"; they redistribute robustness between invariance to local perturbations and preservation of global semantic distinctions.

That conclusion should only be made if the data support it.

---

# 11. Experimental Matrix

The six experiments do not all require every model equally.

| Experiment | Dimensional | VIB | VQ | Quantized | Autoencoder |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Capacity → robustness | **Primary** | — | — | — | — |
| 2. Generalization | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3. Geometry | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4. Invariance/separation | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5. Semantic collisions | ✓ | ✓ | **Primary** | **Primary** | ✓ |
| 6. Robustness trade-off | ✓ | ✓ | ✓ | ✓ | ✓ |

This avoids implementing every model before knowing whether Experiment 1 produces an interesting signal.

---

# 12. Critical Experimental Controls

Several factors could produce misleading conclusions if not controlled carefully.

## 12.1 Clean-Accuracy Confounding

Suppose:

\[
d_z=512\rightarrow\text{high accuracy}
\]

and:

\[
d_z=16\rightarrow\text{much lower accuracy}.
\]

Lower adversarial robustness in the second model could simply reflect poorer classification.

Therefore report:

1. clean accuracy;
2. robust accuracy;
3. conditional ASR among clean-correct examples.

Also compare subsets of models with similar clean accuracy.

## 12.2 Latent-Scale Confounding

A latent with norm 5 and another with norm 100 cannot fairly receive the same raw \(L_2\) perturbation.

Hence use:

\[
\rho=
\frac{\|\eta\|_2}{\|z\|_2}.
\]

For representation geometry, report both raw and normalized representations where appropriate.

## 12.3 Gradient Masking

This is especially important for Models 3 and 4.

A quantized model may appear robust because:

\[
\nabla Q(z)=0
\]

almost everywhere.

Therefore use:

- BPDA/STE attacks;
- multiple restarts;
- transfer attacks;
- at least one gradient-free sanity check.

If black-box attacks dramatically outperform gradient-based attacks, do **not** claim robustness.

## 12.4 Stochastic Masking

For VIB, use EoT:

\[
\nabla_x
\mathbb E_{\epsilon}
[
L(f(x,\epsilon),y)
].
\]

Randomness must not be mistaken for genuine adversarial robustness.

## 12.5 Training Randomness

Train at least three seeds and report:

\[
\text{mean}\pm\text{standard deviation}.
\]

For important validation-set metrics, bootstrap confidence intervals may additionally be reported.

---

# 13. Recommended Code Organization

Build every family around one model/output contract so attacks and metric code
do not need family-specific calling conventions. Section 17.5 defines that
contract, including canonical, sampled, and pre-quantization latents. Section
22 defines the package layout, commands, configuration system, and portable
artifact formats.

Then Experiments 3, 4, and much of Experiment 6 can be performed without rerunning the network repeatedly.

---

# 14. Execution Order

## Phase 1 — Feasibility

Train only Model 1:

\[
d_z=\{512,256,128,64,32,16\}.
\]

Run:

- Experiment 1;
- Experiment 3;
- Experiment 4.

At this point, determine whether bottleneck strength produces an interesting and reproducible relationship.

## Phase 2 — Generalization

Implement:

- VIB;
- VQ;
- quantized continuous bottleneck;
- learned autoencoder.

Run Experiment 2 and repeat the key geometry analyses.

## Phase 3 — Collision Study

Run Experiment 5, particularly on:

- VQ;
- quantized continuous;
- deterministic dimensional bottlenecks.

## Phase 4 — Synthesis

Run Experiment 6 and determine which explanation is actually supported by the evidence.

This order avoids spending substantial time implementing all five bottleneck families before confirming that the foundational phenomenon exists.

---

# 15. Experimental Logic of the Paper

If the project behaves as hypothesized, the evidence should build sequentially:

\[
\boxed{
\text{Experiment 1: Bottleneck strength changes robustness}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Experiment 2: The effect is not unique to one bottleneck}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Experiment 3: Compression changes representation geometry}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Experiment 4: Geometry reveals invariance/separation behavior}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Experiment 5: Strong bottlenecks permit semantic collapse}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Experiment 6: Local and semantic robustness can be compared directly}
}
\]

Importantly, **Experiment 1 alone does not determine the conclusion**.

If stronger compression makes adversarial robustness monotonically worse, Experiments 3–5 can explain why.

If stronger compression improves robustness, those same experiments can determine whether improvement comes from nuisance contraction and whether there is a hidden cost in semantic separation.

If robustness does not change meaningfully, that result itself challenges the proposed connection.

The research question therefore remains falsifiable rather than designing the experiments around obtaining a particular compression–robustness curve.

---

# 16. Initial Implementation Priority

The first implementation milestone should be **Experiment 1 using Imagenette**.

The initial code should include:

1. Imagenette data loader and preprocessing;
2. ResNet-18 deterministic dimensional bottleneck model;
3. training loop for:
   \[
   d_z=\{512,256,128,64,32,16\};
   \]
4. clean-accuracy evaluation;
5. input-space PGD evaluation;
6. latent-space PGD evaluation;
7. conditional attack success rate;
8. latent extraction;
9. representation-geometry metrics;
10. automated generation of the primary figures and tables.

This is the least expensive experiment that can determine whether the broader research direction is empirically promising.

---

# 17. Implementation Specification and Fixed Decisions

The scientific plan above leaves several choices open that materially affect
the results. In particular, it does not by itself specify whether the backbone
is pretrained, where normalization occurs during an attack, which latent is
used for a stochastic or discrete model, how checkpoints are selected, or how
metrics are aggregated. This section resolves those ambiguities.

The values below are the initial, preregistered defaults. They may be changed
after pilot runs reveal a correctness or feasibility problem, but the reason
for every change must be documented before running the final evaluation.

## 17.1 Scope of the First Software Milestone

The first milestone implements only the deterministic dimensional family and
Experiments 1, 3, and 4. It is complete when all six dimensions and all three
seeds can be trained and evaluated by configuration, without editing Python
source code.

The primary Phase 1 matrix comprises:

```text
6 dimensions x 3 training seeds = 18 trained models
```

One required seed-0 identity-baseline diagnostic makes 19 training jobs in the
initial milestone. If that diagnostic differs materially from the learned
512-dimensional projection, expanding it to three seeds adds two more jobs.

The VIB, VQ, quantized, and autoencoder families should use the same interfaces
and artifact formats, but their implementation is Phase 2 work. This staged
scope prevents family-specific complexity from delaying validation of the main
claim.

## 17.2 Software Stack

Use Python 3.11 or newer and PyTorch with torchvision. The project should use a
`pyproject.toml` and an exact lock file. At minimum, the environment will need:

- PyTorch and torchvision for models, data, and automatic differentiation;
- NumPy, pandas, and SciPy for numerical analysis;
- scikit-learn for selected statistical utilities;
- Hydra/OmegaConf or an equivalent composition system for immutable run
  configurations;
- safetensors and Parquet support for portable latent and metadata artifacts;
- matplotlib and seaborn for figures;
- pytest for unit and smoke tests;
- a maintained Square Attack or AutoAttack implementation for the independent
  gradient-obfuscation check.

Exact versions are not fixed in this prose because CUDA compatibility differs
across machines. The resolved versions, CUDA version, GPU model, git commit,
and lock-file hash must be saved with each run.

## 17.3 Dataset, Splits, and Sample Identity

Use the 320-pixel Imagenette release as the source data. Load it as an
ImageFolder-style dataset and preserve the official class-directory names.
The data preparation command must create a manifest containing:

```text
sample_id, relative_path, class_name, class_index, original_split, sha256
```

`sample_id` is the normalized relative path and must remain stable across all
runs. The sorted class-to-index mapping and an archive checksum must be stored
in the same manifest.

Do not use the official validation split for hyperparameter selection. Make one
fixed, stratified split of the official training data:

- 90% development-train;
- 10% development-tuning;
- the complete official validation split as the final evaluation set.

Use split seed `2025` once and reuse the resulting manifest for every model
family and training seed. Choose checkpoints and pilot hyperparameters using
only the development-tuning set. Inspect final-evaluation results only after
the corresponding configuration and attack settings have been frozen.

The initial image pipeline operates in raw floating-point RGB space in
`[0, 1]`:

```text
Training:
  RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3), antialias=True)
  RandomHorizontalFlip(p=0.5)

Tuning/final evaluation:
  Resize(shorter_side=256, antialias=True)
  CenterCrop(224)
```

Use bilinear interpolation. Apply ImageNet channel normalization inside a
model wrapper, not in the stored attack variable:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

This convention ensures that an input-space value such as
`epsilon = 4 / 255` always refers to raw pixels. All attacks project and clamp
in `[0, 1]` before normalization.

## 17.4 Backbone and Deterministic Bottleneck

The primary experiment uses torchvision's standard ResNet-18 architecture
with random initialization. Remove its final fully connected layer and retain
global average pooling, so that the backbone returns `h` with shape `[B, 512]`.
Do not use ImageNet-pretrained weights in the primary experiment, because
Imagenette is derived from ImageNet and pretraining would make the provenance
of task information difficult to interpret. A pretrained-backbone experiment
may be added later and must be labeled as a separate ablation.

Train the complete backbone, bottleneck, and classifier end to end. For every
dimension use:

```text
h [B, 512]
  -> Linear(512, dz, bias=True)
  -> z [B, dz]
  -> Linear(dz, 10, bias=True)
```

Do not add an activation, normalization layer, or dropout around `z` in the
primary experiment. Such additions change latent scale and geometry. The
`dz=512` primary baseline retains the learned `Linear(512, 512)` so that the
architecture differs only in width. Also train one seed of an identity
`h -> classifier` control. If the learned 512-dimensional projection changes
the result substantially, expand the identity control to all three seeds and
report it as a baseline rather than calling the projected model a
"no-bottleneck" model.

## 17.5 Common Model Contract

Returning either a tensor or a tuple depending on a flag becomes fragile once
models have auxiliary losses and multiple latent representations. Use an
explicit result type:

```python
@dataclass
class BottleneckOutput:
    logits: torch.Tensor
    # Exact value consumed by the classifier/decoder in this forward pass.
    latent: torch.Tensor
    # Deterministic value used by the primary geometry analysis.
    canonical_latent: torch.Tensor
    # ResNet representation before family-specific projections.
    backbone_features: torch.Tensor
    # Continuous value immediately before sampling/rounding/code assignment;
    # for a dimensional model this is the input to the dimensional projection.
    pre_bottleneck: torch.Tensor
    aux_losses: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]

class BottleneckModel(nn.Module):
    def encode_features(self, x): ...
    def apply_bottleneck(self, features, *, sample: bool): ...
    def encode(self, x, *, sample: bool = False) -> BottleneckOutput: ...
    def classify_latent(self, z): ...
    def forward(self, x, *, sample: bool = False) -> BottleneckOutput: ...
```

`x` is always raw `[0, 1]` RGB after spatial augmentation. The model's input
wrapper performs channel normalization. `latent` is the value actually consumed
by the classifier or decoder in that forward pass, so for a sampled VIB forward
it is a posterior sample. `canonical_latent` is the deterministic representation
used for the primary geometry analysis. `pre_bottleneck` is the continuous
value to which the capacity-restricting operation is applied: `h` before the
dimensional projection, `z_e` before VQ assignment, and the `tanh` output before
scalar rounding. Family-specific values such as VIB `mu` and `logvar` also
belong in `metadata`. Every model must document the shapes and meanings of all
fields.

The canonical latent used for geometry and saved as `latent` is:

| Family | Canonical latent |
|---|---|
| Dimensional | output of the learned linear bottleneck |
| VIB | posterior mean `mu`, unless a stochastic metric is explicitly named |
| VQ | flattened quantized token vectors, using exact forward quantization |
| Quantized continuous | exactly rounded post-quantization vector |
| Autoencoder | encoder output consumed by the decoder |

Save the pre-quantization latent as a separate artifact for VQ and continuous
quantization. Never silently substitute it for the canonical latent. The
extraction command writes `canonical_latent`; ordinary forward passes return
the exact `latent` used for their prediction.

## 17.6 Phase 2 Family-Specific Definitions

These definitions remove the remaining ambiguity before the additional model
families are implemented.

### VIB

Use `dz=128`. Both `mu` and `logvar` are independent linear projections from
the 512-dimensional backbone output. Clamp `logvar` to `[-10, 10]` before
forming the standard deviation. During training, use one reparameterized sample
per input. Define KL as the sum over latent coordinates followed by the mean
over the batch:

\[
L_{\rm KL} =
\frac{1}{B}\sum_n \frac{1}{2}\sum_j
(\mu_{nj}^2 + \exp(\log\sigma^2_{nj}) - 1 - \log\sigma^2_{nj}).
\]

The total loss is `CE + beta * KL`; do not divide KL by `dz`. For clean
stochastic evaluation, average class probabilities over 32 posterior samples
and take the argmax. For geometry use `mu`. Report both deterministic-mean and
32-sample clean accuracy so the chosen convention is visible. Keep `beta`
constant from the first optimizer step in the primary sweep; KL warmup or
cyclical annealing is a separately labeled optimization ablation.

### VQ

Project `[B, 512]` to `[B, 16, 32]`. Use one learned codebook of shape
`[K, 32]`, shared by all 16 token positions, nearest-neighbor Euclidean
assignment, and the straight-through value for classification. Flatten the
quantized tokens to `[B, 512]` before the linear classifier. Use:

\[
L = L_{\rm CE}
+ \|\operatorname{sg}(z_e)-z_q\|_2^2
+ 0.25\|z_e-\operatorname{sg}(z_q)\|_2^2,
\]

where each squared-error term is averaged over batch, token, and embedding
coordinates. Use gradient-updated codebook vectors for the initial study; do
not mix this with an EMA codebook in the same sweep. Record codebook perplexity,
the fraction of active codes, and token counts each epoch. A run with less than
10% active codes for five consecutive epochs is marked as codebook collapse and
must not be interpreted as an ordinary bottleneck-strength result.

Initialize codebook entries independently from
`Uniform(-1/sqrt(32), 1/sqrt(32))`. Use the standard torchvision ResNet
initialization and PyTorch default initialization for newly added linear and
convolutional layers. The exact framework version and seed remain part of the
run record.

The expression `16 * log2(K)` is a nominal upper bound on code capacity, not an
estimate of mutual information, and figures must label it accordingly.

### Quantized Continuous

Use a learned `Linear(512, 128)`, then `tanh`, then the exact uniform quantizer
from Section 2.4. The FP32 control applies `tanh` but no rounding; this keeps it
architecturally matched to the low-bit models. Training uses the straight-
through derivative, while tuning and final evaluation use exact rounding in the
forward pass. Tests must verify that a `b`-bit model emits no more than `2**b`
distinct scalar levels and that outputs remain in `[-1, 1]`.

### Autoencoder

Use the ResNet-18 convolutional encoder through global average pooling,
followed by `Linear(512, dz)`. The initial decoder is fixed as:

```text
Linear(dz, 512*7*7), reshape to [B, 512, 7, 7]
five 2x upsampling blocks with channels 512->256->128->64->32->16
3x3 convolution 16->3
sigmoid to [0, 1]
```

Each upsampling block uses nearest-neighbor upsampling, a 3x3 convolution,
GroupNorm with `min(32, channels)` groups, and SiLU. This produces `224 x 224`
output without the checkerboard
artifacts commonly introduced by transposed convolutions. Train only with
pixel MSE for the primary autoencoder comparison; perceptual losses are a
separate ablation.

Train one independent standard ResNet-18 classifier on raw Imagenette using the
same data split and training protocol, then freeze it. Autoencoder clean and
robust accuracy are measured as `g(decode(encode(x)))`. Also report the frozen
classifier's accuracy on the original images to separate reconstruction damage
from classifier error. Use reference-classifier seed 0 for the primary matrix
so all autoencoders face the same downstream decision function; if conclusions
depend strongly on that classifier, repeat the analysis with reference seeds 1
and 2. Select autoencoder checkpoints by lowest tuning MSE rather than tuning
classification accuracy.

## 17.7 Training and Checkpoint Selection

Use the following fixed Phase 1 defaults:

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
weight_decay: 1.0e-4
epochs: 200
warmup_epochs: 5
scheduler: cosine
minimum_learning_rate: 1.0e-6
effective_batch_size: 128
label_smoothing: 0.0
gradient_clip_norm: 5.0
mixed_precision: bf16_if_supported_else_fp16
seeds: [0, 1, 2]
```

Exclude biases and normalization parameters from weight decay. If memory does
not permit a physical batch of 128, divide the loss by the number of gradient
accumulation steps and update the scheduler once per optimizer step. BatchNorm
statistics are updated on physical microbatches; therefore record the physical
batch size, and keep it constant across the compared Phase 1 runs on a given
hardware setup.

Save `last` and `best_tune_accuracy` checkpoints. The reported checkpoint is
the epoch with highest development-tuning clean accuracy; break ties in favor
of the earlier epoch. Do not choose a checkpoint using tuning robustness. The
checkpoint contains model, optimizer, scheduler, gradient-scaler, epoch, seed,
and resolved configuration so a run can be resumed exactly.

First tune the common optimizer and schedule on only the `dz=512` and `dz=64`
pilot models using seed 0. Freeze the common training configuration before the
18 Phase 1 runs. Do not tune a separate optimizer for each dimension in the
primary controlled comparison. Later, family-specific optimization ablations
may be used to determine whether a negative result was caused by undertraining.

Use deterministic data-sampler generators and record seeds for model
initialization, data order, augmentation, stochastic bottlenecks, and attacks
separately. Enable deterministic operations when feasible and record any CUDA
operation that prevents exact determinism. Reproducibility means statistically
equivalent reruns, not an unsupported promise of bitwise identity across GPU
types.

---

# 18. Exact Robustness Evaluation Protocol

## 18.1 Input-Space PGD

The primary attack is untargeted cross-entropy PGD in raw pixel space. Set the
model to evaluation mode so BatchNorm statistics do not change. For each
nonzero `epsilon`:

```text
constraint: L-infinity
steps: 40
step size: 2 * epsilon / 40
initialization: uniform independently in [-epsilon, epsilon]
restarts: 5
projection: intersect L-infinity ball around clean x with [0, 1]
selection: any misclassified candidate outranks a correctly classified one;
           within the same success state, retain the highest-loss candidate
attack arithmetic: float32, even if training used mixed precision
```

Track the highest-loss point during the trajectory rather than only the final
iterate. Evaluate every sample, not only clean-correct samples; the latter
restriction is used only for conditional ASR. For `epsilon=0`, call the clean
evaluation path rather than PGD.

For VIB, average the attack loss gradient over 8 fresh posterior samples per
step during pilots and 32 per step for final results. Also use 32 samples for
the attacked prediction. Fix the random-number stream per restart so all VIB
strengths receive comparable EoT noise.

For VQ and quantized models, use exact quantization in the forward pass and an
STE/BPDA derivative in the backward pass. Select one model-independent,
stratified set of up to 1,000 final-evaluation sample IDs and also run Square
Attack with a 5,000-query budget on those IDs. Conditional ASR uses the subset
that is clean-correct for the model being evaluated; additionally report the
shared clean-correct intersection for direct model comparisons. Report the
stronger success rate of PGD and Square Attack on that subset, while still
showing both separately. A transfer attack from the nearest-capacity continuous
model is an additional masking diagnostic.

## 18.2 Latent-Space PGD

For continuous dimensional and VIB models, attack the canonical latent consumed
by the classifier. For the autoencoder, attack the encoder output consumed by
the decoder. For each sample define:

\[
r_i = \rho\max(\|z_i\|_2, 10^{-12}).
\]

Initialize uniformly inside the per-sample L2 ball, use 40 normalized-gradient
steps with step length `2*r_i/40`, project after every step, use 5 restarts, and
retain the highest-loss point. The classifier or decoder parameters remain
frozen during the attack.

Discrete models require two explicitly named evaluations:

1. **Pre-quantization attack (primary):** perturb the continuous encoder output,
   reapply exact VQ or scalar quantization on every forward pass, and then
   classify. Normalize the budget using the clean pre-quantization norm.
2. **Post-bottleneck ambient attack (diagnostic):** perturb the quantized vector
   directly before the classifier. This measures downstream decision margin but
   creates off-codebook or off-grid vectors and must not be described as a
   valid discrete-channel perturbation.

Never combine these two latent attack surfaces in the same curve.

## 18.3 Metrics

For `N` final-evaluation samples define:

\[
\text{clean accuracy}=\frac{1}{N}\sum_i
\mathbf{1}[f(x_i)=y_i],
\]

\[
\text{robust accuracy}(r)=\frac{1}{N}\sum_i
\mathbf{1}[f(x_i^{\rm adv}(r))=y_i],
\]

and

\[
\text{conditional ASR}(r)=
\frac{\sum_i \mathbf{1}[f(x_i)=y_i]
\mathbf{1}[f(x_i^{\rm adv}(r))\ne y_i]}
{\sum_i\mathbf{1}[f(x_i)=y_i]}.
\]

Compute AUC with the trapezoidal rule and divide by the maximum evaluated
radius, yielding a value in `[0, 1]`. Input AUC spans `[0, 16/255]`; latent AUC
spans `[0, 0.30]`. "Minimum successful perturbation" means the first successful
radius on the stated evaluation grid and is therefore interval-censored. Treat
samples with no success as right-censored beyond the maximum radius and use a
Kaplan-Meier median; if the median is not reached, report it as greater than the
maximum radius. Do not report a median over successful attacks only or present
the grid estimate as an exact continuous minimum.

Store one row per `sample_id`, radius, restart summary, and checkpoint. Aggregate
tables are generated from these per-sample records rather than directly inside
the attack loop.

## 18.4 Attack Correctness Checks

Every final attack run must pass these checks:

- `epsilon=0` prediction equals the clean prediction;
- adversarial examples satisfy both the norm bound and `[0, 1]` bounds within
  numerical tolerance;
- the retained loss is no smaller than the initial-point loss;
- robust accuracy is non-increasing across nested radii after retaining the
  strongest previously found adversarial example per sample;
- increasing steps or restarts on a diagnostic subset does not materially
  increase reported robust accuracy;
- BPDA/STE and black-box results for discrete models are shown together;
- EoT sample count is increased on a subset to check convergence for VIB.

Failure of a check is an evaluation bug or an unresolved masking warning, not a
robustness result.

---

# 19. Exact Geometry and Invariance Estimators

## 19.1 Latent Extraction

Run latent extraction in evaluation mode with no augmentation and the fixed
tuning/final preprocessing. Store float32 latents but accumulate covariance and
pairwise summaries in float64. For VIB store `mu`, `logvar`, and optionally a
fixed set of posterior samples; the primary geometry tables use `mu`.

Normalize each latent with:

\[
\bar z = z / \max(\|z\|_2, 10^{-12}).
\]

Report the number of near-zero vectors. Compute exact pairwise quantities in
blocks so the complete pairwise matrix does not need to reside in memory.

For intra- and inter-class distance, first compute the relevant mean for each
anchor sample and then average anchors. This gives each image equal weight and
prevents a larger class from dominating. Exclude self-pairs. Also report
per-class values and their macro average.

For covariance, center latents using the evaluation-set mean. Drop eigenvalues
smaller than `1e-12 * largest_eigenvalue` before computing effective rank, and
report both nominal dimension and effective rank. Estimate encoder spectral
norm with 20 JVP/VJP power iterations on a fixed stratified set of 500 sample
IDs. Use the same IDs for every model.

For empirical contraction, pair each anchor with 10 deterministic same-class
and 10 deterministic cross-class partners chosen by a hash of the two sample
IDs. Report same-class and cross-class ratios separately. Input distances are
computed on raw center-cropped pixels before ImageNet normalization. Use the
raw, unnormalized canonical latent in the contraction equation from Section
7.7; report the same calculation with unit-normalized latents as a separate
scale-free diagnostic.

## 19.2 Natural Collision Threshold

Calibrate continuous collision thresholds only on the development-tuning set.
For each tuning example compute its nearest *other* example from the same
class. Set the primary threshold `tau` to the 5th percentile of these distances
for that checkpoint. Apply the frozen threshold to opposing-class nearest
neighbors in the final-evaluation set. Include sensitivity results for the 1st
and 10th percentiles and always report the threshold-free distribution of
nearest opposing-class distance. These primary collision distances use
Euclidean distance between unit-normalized canonical latents.

For VQ report all of:

- exact equality of the complete 16-token code sequence;
- fraction of matching token positions for the nearest opposing-class sample;
- Hamming distance between code sequences;
- Euclidean distance between flattened codebook vectors.

Exact whole-sequence equality alone may be too rare to characterize collision
behavior.

## 19.3 Invariance Transform Set

Apply transforms in raw `[0, 1]` space to the deterministic center-cropped
image. Use the following fixed variants:

```text
translation: four directions by 4 pixels, reflection padding
brightness: factors 0.9 and 1.1
contrast: factors 0.9 and 1.1
Gaussian noise: sigma=0.02, two fixed noise seeds per sample
horizontal flip: one variant
```

Clamp after photometric transforms. Compute the latent distance for every
variant, average variants within each transform type, and then macro-average
the five transform types so Gaussian noise or translation does not receive
extra weight. Report each transform separately as well as the macro value.
Also report prediction consistency and accuracy under each benign transform;
a low latent distance is not useful invariance if the clean prediction was
already wrong.

---

# 20. Collision Attack Protocol

Choose source-target pairs deterministically from examples that both the
bottleneck system and the frozen reference classifier classify correctly. Each
source is paired with one correctly classified target from every opposing
class, capped at 1,000 source-target pairs per configuration for the initial
study. The same sample-ID pairs are reused across bottleneck strengths whenever
all correctness conditions hold. Report results both on this shared
intersection and on each model's full eligible set. For VIB, pair selection and
continuous collision distance use the posterior mean.

Optimize a targeted latent-collision objective with a source-preservation term
in raw pixel space:

\[
L = d(E(x_{\rm adv}), E(x_t))
+ \lambda_{\rm sem}\,\operatorname{CE}(g(x_{\rm adv}), y_s),
\]

where `g` is a separately trained, frozen reference classifier. Minimize this
loss with projected Adam or normalized projected gradient descent for 200
steps and 5 restarts. Sweep `lambda_sem` on the development-tuning pairs only,
then freeze it. Use the same input epsilon grid as Experiment 1. A collision is
successful only when all of the following hold:

1. the input perturbation is within its specified bound;
2. the reference classifier still predicts the source class;
3. the bottleneck-specific collision condition is met;
4. before the attack, the bottleneck system and reference classifier correctly
   classified both source and target.

For continuous models, the primary collision condition is distance below the
frozen tuning-set threshold from Section 19.2. Also report target-class
prediction and achieved latent-distance percentile. For VQ require exact target
code sequence for "exact collision" and separately report token-match rates.
For scalar quantization, target the center of each target bin and require exact
equality of the full quantized vector for "exact collision"; separately report
the fraction of matched scalar bins. These graded metrics prevent an all-zero
success table from hiding meaningful movement toward collision.

---

# 21. Statistical Analysis and Reporting

The independent replication unit is the training seed, not an individual
image. Report each seed as a point and report configuration summaries as
mean plus standard deviation across the three seeds. Use paired seed-level
contrasts when comparing bottleneck strengths trained with the same seed.

For final-evaluation accuracy and distance summaries, add a 95% hierarchical
bootstrap interval that resamples training seeds and then samples within seed.
If three seeds are too few for a stable seed-level inferential claim, describe
the result as exploratory and add seeds before making a strong claim.

The preregistered Phase 1 primary outcomes are:

1. input robust-accuracy AUC versus `dz`;
2. latent robust-accuracy AUC versus `dz`;
3. median nearest opposing-class distance versus `dz`.

Fit both a monotonic ordinal trend and a quadratic ordinal trend, because the
stated hypothesis permits a non-monotonic relationship. Treat Spearman
correlations between robustness and the geometry metrics as explanatory and
report confidence intervals. Correct the family of geometry-correlation
p-values using Benjamini-Hochberg. Do not describe a correlation as mediation
or causation without a separate justified analysis.

Always show clean accuracy beside robustness. In addition to conditional ASR,
perform a sensitivity analysis restricted to configurations whose mean clean
accuracy differs by no more than two percentage points, if such a subset
exists. Do not discard low-accuracy configurations merely to make a trend
cleaner.

---

# 22. Repository, Configuration, and Artifact Contract

Use a package layout rather than placing importable code directly in experiment
scripts:

```text
ribs/
├── pyproject.toml
├── uv.lock or equivalent exact lock file
├── configs/
│   ├── model/
│   ├── train/
│   ├── attack/
│   └── experiment/
├── src/ribs/
│   ├── data/
│   ├── models/
│   ├── attacks/
│   ├── metrics/
│   ├── evaluation/
│   └── cli/
├── tests/
│   ├── unit/
│   └── smoke/
├── scripts/
└── outputs/                 # ignored by git
```

Provide separate commands for preparing data, training, extracting latents,
attacking, computing metrics, and rendering figures. A representative command
contract is:

```bash
python -m ribs.cli.prepare_data data=imagenette
python -m ribs.cli.train model=dimensional model.dz=128 seed=0
python -m ribs.cli.evaluate run_id=<run-id> evaluation=clean
python -m ribs.cli.attack run_id=<run-id> attack=input_pgd
python -m ribs.cli.extract_latents run_id=<run-id> split=final
python -m ribs.cli.analyze experiment=geometry run_id=<run-id>
python -m ribs.cli.render experiment=phase1
```

Every run receives a stable ID derived from the resolved configuration plus an
explicit attempt number. Write outputs to a temporary run directory and add a
`COMPLETED` marker only after all required files have been flushed. Downstream
jobs must reject incomplete runs.

Each completed training run contains:

```text
resolved_config.yaml
environment.json
data_manifest_hash.txt
checkpoints/{last,best_tune_accuracy}.pt
history.parquet
metrics/tune_clean.json
logs.txt
COMPLETED
```

Evaluation adds per-sample Parquet files keyed by `sample_id`, summary JSON,
and an evaluation configuration containing attack seeds and software versions.
Store dense latent arrays in safetensors with a parallel Parquet index rather
than serializing one Python dictionary per sample. Figures and paper tables
must be regenerated from stored metrics by scripts; manually copied numbers
are not source data.

Do not overwrite a completed run. A changed configuration or rerun creates a
new run ID. Keep large raw datasets, checkpoints, and outputs out of git, while
committing manifests, configs, schemas, analysis code, and small test fixtures.

---

# 23. Testing and Milestone Acceptance Criteria

Before launching the Phase 1 training runs, the implemented Phase 1 components
must pass:

## Unit tests

- every implemented model obeys the common shapes and output contract;
- L-infinity and L2 projections satisfy their bounds per sample;
- raw-pixel normalization does not change the meaning of attack epsilon;
- AUC and conditional ASR match hand-computed examples;
- blockwise geometry equals a direct small-matrix implementation;
- latent artifacts round-trip with sample IDs in the same order.

## Smoke tests

- the dimensional model can overfit a fixed batch of 32 examples;
- a two-epoch CPU or single-GPU run can train, resume, evaluate, and extract
  latents;
- PGD raises loss on a differentiable toy model and `epsilon=0` is identical
  to clean evaluation;
- changing only the run seed changes stochastic state while preserving the
  data split;
- the figure pipeline runs from stored fixture metrics without a checkpoint.

## Phase 1 acceptance criteria

Phase 1 is ready for analysis only when:

1. all 18 expected runs have a completion marker and matching data-manifest
   hash;
2. the seed-0 identity diagnostic is complete and has been compared with the
   learned 512-dimensional projection;
3. no run has NaN loss, an unresolved resume mismatch, or a failed attack
   correctness check;
4. clean, input-robust, and latent-robust per-sample records exist for every
   final sample;
5. geometry and invariance metrics can be regenerated from saved latents;
6. all primary figures can be produced by one documented command;
7. exclusions, failed attempts, and configuration changes are recorded rather
   than silently removed.

Before each Phase 2 family is added to the experimental matrix, add and pass
its family-specific tests: the hand-computed VIB KL and zero-KL case, VQ
nearest-code and straight-through-gradient checks, scalar-quantizer level and
rounding checks, and fixed-batch overfitting for the newly implemented family.

---

# 24. Decision Gates Between Phases

At the end of Phase 1, proceed to all Phase 2 families if at least one of the
following is reproducible across seeds:

- a meaningful change in input or latent robust AUC across `dz`;
- a meaningful change in semantic margin or invariance even if robust accuracy
  is flat;
- a well-supported null result whose generality across bottleneck mechanisms is
  scientifically worth testing.

"Meaningful" must be judged using effect sizes and uncertainty, not only a
p-value. If the Phase 1 result is dominated by clean-accuracy collapse, first
run the prespecified optimization/accuracy-matching ablation. If attack sanity
checks fail, resolve evaluation validity before training additional families.

Before Phase 2 begins, write a short frozen decision record containing:

```text
included model families and strengths
family-specific training changes
primary Phase 2 outcomes
final attack budgets
any deviations from this implementation specification
```

This makes the distinction between planned analysis and post hoc exploration
auditable while still allowing the project to respond sensibly to Phase 1.
