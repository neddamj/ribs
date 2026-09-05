# Phase 1 Results

## Scope and analysis unit

The dimensional matrix contains (d_z \in \{16, 32, 64, 128, 256, 512\})
with training seeds 0, 1, and 2. The seed-0 identity model is reported as a
diagnostic rather than as another dimensional replicate. Training seed is the
independent replication unit. With only three seeds, all inferential
statements are exploratory.

## Primary robustness results

All accuracy and AUC values below are proportions. Values are mean ± SD over
the three training seeds.

| (d_z) | Clean accuracy | Input-PGD AUC | Latent-PGD AUC |
|---:|---:|---:|---:|
| 16 | 0.9037 ± 0.0033 | 0.0565 ± 0.0018 | 0.8115 ± 0.0027 |
| 32 | 0.9001 ± 0.0042 | 0.0587 ± 0.0013 | 0.8082 ± 0.0050 |
| 64 | 0.9011 ± 0.0045 | 0.0584 ± 0.0018 | 0.8064 ± 0.0086 |
| 128 | 0.9009 ± 0.0040 | 0.0602 ± 0.0027 | 0.8141 ± 0.0071 |
| 256 | 0.9027 ± 0.0065 | 0.0588 ± 0.0014 | 0.8135 ± 0.0013 |
| 512 | 0.9011 ± 0.0095 | 0.0614 ± 0.0026 | 0.8169 ± 0.0048 |

The input-PGD AUC spans only 0.0565–0.0614 across dimensionalities, and the
latent-PGD AUC spans 0.8064–0.8169. The (d_z=512) model has the highest
mean AUC for both attacks, but the differences are small relative to the
seed-level variation. The results therefore do not support a strong monotonic
capacity-to-robustness effect.

Paired AUC differences relative to (d_z=512), with the same training seed
paired across dimensionalities, are:

| (d_z) | Input-PGD difference | Latent-PGD difference |
|---:|---:|---:|
| 16 | −0.004889 ± 0.001203 | −0.005364 ± 0.002573 |
| 32 | −0.002630 ± 0.002080 | −0.008723 ± 0.008015 |
| 64 | −0.002943 ± 0.001415 | −0.010450 ± 0.004430 |
| 128 | −0.001131 ± 0.000199 | −0.002807 ± 0.003302 |
| 256 | −0.002537 ± 0.003696 | −0.003374 ± 0.004919 |

These contrasts are consistent with a small advantage for the learned
512-dimensional model, but they are not uniformly large or precise enough to
justify a strong claim.

## Robustness curves

Input-PGD robustness falls sharply at small input radii. Across the
dimensional models, mean robust accuracy is approximately 0.37–0.40 at
(\epsilon=0.003922), 0.05–0.08 at \(\epsilon=0.007843), and effectively
zero by \(\epsilon=0.031373\). Differences between dimensionalities are
small compared with this overall drop.

Latent-PGD robustness degrades more gradually: mean robust accuracy remains
approximately 0.88–0.90 at \(\rho=0.05\), 0.85–0.86 at \(\rho=0.10\),
0.78–0.79 at \(\rho=0.20\), and 0.68–0.70 at \(\rho=0.30\). The curves
show modest dimensionality differences but no clean monotonic ordering.

The complete per-radius curves are in
[`outputs/phase1_curves.parquet`](outputs/phase1_curves.parquet), and the
regenerated figures are in [`outputs/figures/`](outputs/figures/).

## Geometry and invariance interpretation

Several geometry measurements change systematically with representation
dimension:

- effective rank increases from 8.21 ± 0.16 at (d_z=16) to 9.21 ± 0.07 at
  (d_z=512);
- median nearest opposing-class distance increases from 0.500 ± 0.010 to
  0.541 ± 0.018;
- separation ratio decreases from 1.917 ± 0.012 to 1.832 ± 0.013 because
  intra-class distance grows while inter-class distance remains nearly flat;
- normalized same-class contraction increases from 2.13 ± 0.02 to
  11.41 ± 0.12, and normalized different-class contraction increases from
  1.07 ± 0.00 to 6.05 ± 0.02.

Thus, lower-dimensional representations have smaller effective rank and
smaller opposing-class margins, but this geometric change translates into
only a modest change in the measured robustness AUCs.

Invariance is comparatively stable and non-monotonic. Mean prediction
consistency ranges are 0.974–0.977 for brightness, 0.977–0.979 for contrast,
0.947–0.951 for flips, 0.968–0.973 for noise, and 0.976–0.978 for
translations. Nuisance distance ranges from 0.1065 to 0.1132, semantic
distance from 1.411 to 1.423, and the semantic-to-nuisance ratio from 12.55
to 13.32. These results do not show a clear monotonic invariance trend.

## Geometry–robustness correlations

The correlations use the 18 dimensional configurations per attack. Confidence
intervals are seed-cluster bootstrap intervals, and BH correction is applied
across the 16 geometry-correlation tests.

The strongest latent-PGD associations after BH correction are:

- median nearest opposing-class distance: \(\rho=0.740\), 95% CI
  [0.543, 0.943], BH \(q=0.007\);
- encoder Jacobian median: \(\rho=0.645\), 95% CI [0.564, 0.886], BH
  \(q=0.031\).

Other latent associations are suggestive but do not survive BH correction:
effective rank \(\rho=0.552, q=0.093\), same-class contraction
\(\rho=0.492, q=0.093\), and different-class contraction
\(\rho=0.498, q=0.093\). Input-PGD contraction associations are positive
but also do not survive correction: same-class \(\rho=0.519, q=0.093\) and
different-class \(\rho=0.486, q=0.093\).

These are exploratory correlations. They should not be described as
mediation, causation, or evidence for a universal bottleneck effect.

## Identity versus learned 512-dimensional model

In the seed-0 diagnostic, the identity representation has clean accuracy
0.8976 versus 0.8910 for the learned (d_z=512) projection. Its input-PGD
AUC is 0.0580 versus 0.0635, while its latent-PGD AUC is 0.6674 versus
0.8116.

The identity representation has higher effective rank (20.18 versus 9.19),
but a smaller opposing-class distance (0.461 versus 0.521) and lower
separation ratio (1.575 versus 1.843). This shows that latent robustness is
not determined by nominal dimension or effective rank alone; learned
projection geometry and training materially affect the result.

## Interpretation

### Main scientific conclusion

Phase 1 does not support the simple claim that reducing representation
dimension improves adversarial robustness. Clean accuracy is effectively
matched across the dimensional sweep, but the learned 512-dimensional model
has the highest mean input- and latent-PGD AUC. All paired mean AUC contrasts
against that model are negative. The size and ordering of the differences are
not strong enough, with three training seeds, to establish a large monotonic
effect. The evidence therefore leans toward a small robustness cost from
stronger dimensional compression rather than a robustness benefit.

The input-PGD AUC values near 0.06 should not be read as 6% accuracy at one
attack strength. Each value is robust accuracy integrated over the full
\([0,16/255]\) radius range and normalized to lie in \([0,1]\). Accuracy is
already approximately 0.37--0.40 at \(1/255\), approximately 0.05--0.08 at
\(2/255\), and nearly zero from \(4/255\) onward. The principal input-space
finding is therefore that all of these normally trained models are highly
vulnerable, while bottleneck width makes only a modest difference relative to
that shared vulnerability.

Input- and latent-PGD AUC magnitudes must not be compared directly. The two
attacks operate in different spaces, use different norms, and span different
radius scales. The high latent AUC only describes behavior over the specified
relative-L2 latent budget; it does not mean that the model is intrinsically
more robust in latent space than in input space.

### Nominal width is a weak proxy for used capacity

The effective rank changes only from 8.21 at \(d_z=16\) to 9.21 at
\(d_z=512\), despite the 32-fold change in nominal width. Under this
covariance-based measure, the wide representations occupy a low-dimensional
subspace and the 16-dimensional representation still accommodates nearly all
of the effective dimensionality used by the wider models. This is a plausible
explanation for the flat clean-accuracy and robustness curves: changing the
size of the latent container does not necessarily impose a comparably large
change in the information or degrees of freedom actually used by the network.

Consequently, these results are evidence about deterministic dimensional
bottlenecks, not about information restriction in general. Subsequent phases
must measure realized geometry and effective capacity instead of treating a
configured bottleneck parameter as a direct measurement of retained
information.

### Global class compactness and local semantic margin diverge

Compression changes different aspects of geometry in different directions.
From \(d_z=512\) to \(d_z=16\), mean inter-class distance remains near 1.42
and intra-class distance decreases from approximately 0.775 to 0.741. This
produces the higher separation ratio at lower dimension. At the same time,
median nearest opposing-class distance decreases from 0.541 to 0.500. Thus,
the compressed models form somewhat tighter classes on average while bringing
the closest examples from different classes nearer together.

This distinction explains why an apparently favorable global separation ratio
does not yield improved robustness. Average class compactness can improve
while the local cross-class regions most relevant to semantic confusion become
less well separated. The correlation analysis is consistent with this
interpretation: nearest opposing-class distance is the strongest geometry
correlate of latent robustness.

The correlation does not establish a causal or mediating pathway. Dimension,
training dynamics, geometry, and robustness all vary together in only 18
configurations from three independent training seeds. The positive association
between encoder Jacobian magnitude and latent robustness is particularly
unsuitable for a causal interpretation because the latent attack bypasses the
encoder; it is more plausibly a correlated feature of the learned solution.

### The proposed nuisance-invariance mechanism is not observed

The nuisance invariance measurements are stable and non-monotonic across
dimension. Phase 1 therefore provides no evidence for the proposed pathway in
which stronger dimensional compression removes the tested nuisance variation
and thereby improves robustness. Other bottleneck mechanisms could still
produce that behavior, but it should now be treated as a hypothesis to test
rather than the expected explanation of the dimensional results.

### Meaning of the identity diagnostic

The large seed-0 latent-AUC difference between the identity representation and
the learned 512-dimensional projection shows that the learned projection is
not a neutral no-bottleneck control. A learned full-width linear map can
materially change Euclidean latent geometry and therefore the outcome of a
relative-L2 latent attack. Latent robustness is partly a property of the
chosen representation coordinates, not just the end-to-end classifier's
semantic behavior.

The learned \(d_z=512\) configuration should therefore be called the weakest
member of the learned dimensional sweep, not a no-bottleneck baseline. Because
the identity difference is material, the prespecified identity follow-up
should be expanded to seeds 1 and 2 before making a seed-level baseline claim.

### Implication for the research direction

Phase 2 remains warranted because Phase 1 shows a reproducible geometry change
and because nominal dimensionality may have been a weak intervention on actual
capacity. The purpose of Phase 2 is now sharper: determine whether mechanisms
that restrict information through stochastic regularization, discreteness,
precision, or reconstruction produce robustness effects that the dimensional
sweep did not, and whether any such effects track local semantic margin across
families.

A shared cross-family relationship between measured geometry and robustness
would support a broader bottleneck account. Effects that appear in only one
family should instead be reported as mechanism-specific. Flat results across
all families would be evidence against the proposed general connection between
bottleneck strength and adversarial robustness.

## Phase 2 decision gate

The Phase 2 gate is met under the research plan's qualitative criterion,
based on a reproducible semantic-margin/geometry change rather than a strong
robustness-AUC effect. The (d_z=512) opposing-class margin exceeds the
(d_z=16) margin for all three training seeds. Effective rank and normalized
contraction show the same broad dimensional trend.

This supports proceeding to the planned Phase 2 families under the frozen
decision record in Section 25 of `RIBS_RESEARCH_PLAN.md`. It does not justify
claiming a universal bottleneck phenomenon, and no Phase 2 training was
performed here.

## Result locations

- [Aggregate results](outputs/phase1_summary.csv)
- [Seed-level summary](outputs/phase1_seed_summary.csv)
- [Paired AUC contrasts](outputs/phase1_paired_contrasts.csv)
- [Robustness curves](outputs/phase1_curves.parquet)
- [Geometry correlations](outputs/phase1_geometry_correlations.csv)
- [Identity diagnostic](outputs/phase1_identity_diagnostic.parquet)

# Phase 2 execution status (2026-09-02)

The Phase 2 implementation, frozen decision record, family-specific tests,
artifact checks, postprocessing script, aggregation, plotting, and acceptance
validation are present. The timestamped record is
[`configs/phase2_decision_20260902T120000.yaml`](configs/phase2_decision_20260902T120000.yaml).
It was written before any Phase 2 final-evaluation results were inspected.

No Phase 2 training or final evaluation was launched because this environment
has no usable GPU: `nvidia-smi` reported that it could not communicate with
the NVIDIA driver, and PyTorch reported `cuda_available=False` with zero CUDA
devices. A CPU benchmark of one dimensional-model training step at the
production image and batch settings took approximately 2.60 seconds; the
prescribed 200-epoch run is approximately ten hours before Phase 2's attack
and analysis workload. Running the full 69-run matrix on CPU would therefore
not be a safe or reproducible completion path within this environment.

The required identity follow-up is correspondingly incomplete. The existing
seed-0 identity run remains valid, while seeds 1 and 2 are missing. The
immutable audit is
[`outputs/phase1_identity_followup_20260902T120000Z.json`](outputs/phase1_identity_followup_20260902T120000Z.json),
which records `status: blocked` and confirms that the prior identity
diagnostic was preserved. The Phase 2 acceptance report is
[`outputs/phase2_acceptance.json`](outputs/phase2_acceptance.json): it expects
69 runs and observes zero completed Phase 2 runs.

Therefore there are no Phase 2 scientific results to interpret, no Phase 2
figures or aggregate tables to claim, and no Phase 2 samples or failed runs
were discarded. Phase 1 artifacts and its `status: ready` acceptance report
remain intact. The safest next action is to run the identity follow-up and the
frozen Phase 2 matrix in the intended CUDA environment, then execute
`scripts/run_phase2_postprocess.sh` and require `validate-phase2` to report
`status: ready` before updating the scientific interpretation.
