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

## Phase 2 live execution checkpoint (2026-09-07)

The preceding 2026-09-02 section is retained as a historical record of the
pre-GPU execution state. Authorized CUDA execution has since been started
without changing the frozen decision record or Phase 1 artifacts. Canonical
deduplication currently finds all 69 Phase 2 training configurations complete
(VIB 18, VQ 18, quantized 18, and autoencoder 15). Three configurations have
completed the primary final robustness evaluation, and one configuration has
passed the attack-diagnostic checks. The remaining postprocessing is still
running in four GPU workers (VIB EoT attack, VQ transfer attack, quantized
transfer attack, and autoencoder collision tuning).

This is an execution checkpoint, not a Phase 2 scientific result. Aggregate
tables, figures, acceptance validation, and interpretation remain deferred
until every configuration has valid final artifacts and the prescribed
diagnostics pass or are explicitly documented as evaluation failures.

The current retry provenance is also preserved in the run directories: VIB
robustness attempts `attempt0` and `attempt1` are incomplete; autoencoder
collision-tuning attempts `attempt0`, `attempt1`, and `attempt2` are preserved,
with the latest attempt using the memory-safe per-pair fallback; and the VQ
and quantized transfer stages each have a unique active attempt directory.
These incomplete attempts are not included in any aggregate or scientific
result.

## Phase 2 live execution checkpoint (2026-09-07, continued)

The autoencoder geometry analysis for
`autoencoder-seed2-0b1d033221fc-attempt0` completed successfully and produced
its immutable `COMPLETED` marker. Canonical geometry coverage is now 20/69
(VIB 6, VQ 1, quantized 1, autoencoder 12). The next missing autoencoder
geometry analysis, `autoencoder-seed2-74592e513eb3-attempt0`, has been launched
on GPU0 and is active.

The four-process quantized seed-1 recovery batch tagged
`20260907T024000Z_recovery6` was verified as failed from its saved logs: each
attempt repeatedly raised `multiprocessing.resource_sharer.PermissionError:
[Errno 1] Operation not permitted`. Those stale failed workers were stopped
after preserving their logs; no valid checkpoint or completed artifact was
removed, and the canonical completed training set remains unchanged.

## Phase 2 live execution checkpoint (2026-09-07, geometry queue continued)

The autoencoder geometry queue is complete for all 15 canonical autoencoder
runs. Canonical geometry coverage has advanced to 23/69: VIB 6, VQ 1,
quantized 1, and autoencoder 15. The next canonical geometry run,
`vib-seed1-13c26a0e5b54-attempt0`, is active on GPU0. The canonical run
selection audit for this queue is saved at
`outputs/phase2_canonical_runs_20260907T_monitor.json`.

Primary robustness remains 3/69 and accepted attack diagnostics remain 1/69;
the VIB EoT, VQ transfer, quantized transfer, and autoencoder collision jobs
remain active on the other GPUs. No aggregate or final scientific result has
been produced from this incomplete set.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed1-13c26a0e5b54-attempt0` also
completed successfully. Geometry coverage is now 24/69 (VIB 7, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB geometry run,
`vib-seed1-524cf03621e5-attempt0`, is active on GPU0. Primary robustness is
still 3/69, with no aggregate or final interpretation generated before the
remaining prescribed evaluations are complete.

## Phase 2 live execution checkpoint (2026-09-07, continued VIB geometry)

The canonical geometry analysis for `vib-seed1-524cf03621e5-attempt0`
completed successfully. Geometry coverage is now 25/69 (VIB 8, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB geometry run,
`vib-seed1-7cd8a44642a3-attempt0`, is active on GPU0. Primary robustness
remains 3/69, and all four long-running robustness/collision workers remain
in progress.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry queue)

The canonical geometry analysis for `vib-seed1-7cd8a44642a3-attempt0`
completed successfully. Geometry coverage is now 26/69 (VIB 9, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB geometry run,
`vib-seed1-8af36c1cf7da-attempt0`, is active on GPU0. Primary robustness
remains 3/69; the four robustness/collision workers remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed1-8af36c1cf7da-attempt0`
completed successfully. Geometry coverage is now 27/69 (VIB 10, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB geometry run,
`vib-seed1-a1c0c996201d-attempt0`, is active on GPU0. Primary robustness
remains 3/69, with VIB EoT, VQ transfer, quantized transfer, and autoencoder
collision workers still active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry queue)

The canonical geometry analysis for `vib-seed1-a1c0c996201d-attempt0`
completed successfully. Geometry coverage is now 28/69 (VIB 11, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB geometry run,
`vib-seed1-c7deeac79eff-attempt0`, is active on GPU0. Primary robustness
remains 3/69; the four robustness/collision workers remain active.

## Phase 2 live execution checkpoint (2026-09-07, transfer transitions)

The VQ transfer attack for `vq-seed0-08c408be9bfb-attempt0` and the
quantized transfer attack for `quantized-seed0-32a29c89b9ae-attempt0` both
completed successfully with immutable `COMPLETED` markers. Their postprocess
wrappers advanced to collision tuning, which is now active for both runs;
the autoencoder collision tuning and VIB EoT attack remain active as well.
The VIB geometry run `vib-seed1-c7deeac79eff-attempt0` is still active on
GPU0. Primary robustness remains 3/69 and geometry remains 28/69.

## Phase 2 live execution checkpoint (2026-09-07, VIB seed-2 geometry)

The canonical geometry analysis for `vib-seed1-c7deeac79eff-attempt0`
completed successfully, completing the canonical VIB seed-1 geometry set.
Geometry coverage is now 29/69 (VIB 12, VQ 1, quantized 1, autoencoder 15).
The next canonical run, `vib-seed2-23f3658dcdd9-attempt0`, is active on GPU0.
VQ and quantized transfer artifacts are complete and their collision-tuning
workers are active; primary robustness remains 3/69.

## Phase 2 live execution checkpoint (2026-09-07, VIB seed-2 geometry)

The canonical geometry analysis for `vib-seed2-23f3658dcdd9-attempt0`
completed successfully. Geometry coverage is now 30/69 (VIB 13, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB run,
`vib-seed2-396e90e1a020-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed2-396e90e1a020-attempt0`
completed successfully. Geometry coverage is now 31/69 (VIB 14, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB run,
`vib-seed2-78961eb4f046-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed2-78961eb4f046-attempt0`
completed successfully. Geometry coverage is now 32/69 (VIB 15, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB run,
`vib-seed2-a62a2333b504-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed2-a62a2333b504-attempt0`
completed successfully. Geometry coverage is now 33/69 (VIB 16, VQ 1,
quantized 1, autoencoder 15). The next canonical VIB run,
`vib-seed2-d01bc345e1af-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry continued)

The canonical geometry analysis for `vib-seed2-d01bc345e1af-attempt0`
completed successfully. Geometry coverage is now 34/69 (VIB 17, VQ 1,
quantized 1, autoencoder 15). The final canonical VIB geometry run,
`vib-seed2-f2c608dc1a5e-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VIB geometry complete)

The final canonical VIB geometry analysis for
`vib-seed2-f2c608dc1a5e-attempt0` completed successfully. Canonical geometry
coverage is now 35/69: all 18 VIB configurations, the existing VQ seed-0
codebook-32 configuration, the existing quantized seed-0 FP32 configuration,
and all 15 autoencoder configurations are complete. The next canonical
geometry run, `vq-seed0-249cb22c2ac9-attempt0`, is active on GPU0. Primary
robustness remains 3/69; VIB EoT and collision tuning for VQ, quantized, and
autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed0-249cb22c2ac9-attempt0`
completed successfully. Canonical geometry coverage is now 36/69 (VIB 18,
VQ 2, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed0-4028de1d4968-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed0-4028de1d4968-attempt0`
completed successfully. Canonical geometry coverage is now 37/69 (VIB 18,
VQ 3, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed0-664584b28f3b-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed0-664584b28f3b-attempt0`
completed successfully. Canonical geometry coverage is now 38/69 (VIB 18,
VQ 4, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed0-708798915fbe-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed0-708798915fbe-attempt0`
completed successfully. Canonical geometry coverage is now 39/69 (VIB 18,
VQ 5, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed0-f22f404a4f07-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed0-f22f404a4f07-attempt0`
completed successfully. Canonical geometry coverage is now 40/69 (VIB 18,
VQ 6, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-a3847a172642-attempt0`, is active on GPU0. Primary robustness
remains 3/69; VIB EoT and collision tuning for VQ, quantized, and autoencoder
remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-a3847a172642-attempt0`
completed successfully. Canonical geometry coverage is now 41/69 (VIB 18,
VQ 7, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-b389c430e266-attempt0`, is active on GPU0. Canonical primary
robustness coverage is 3/69 (VQ 1, quantized 1, autoencoder 1); this count
includes the family-specific `autoencoder-robustness-*` marker. VIB EoT and
collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-b389c430e266-attempt0`
completed successfully. Canonical geometry coverage is now 42/69 (VIB 18,
VQ 8, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-bc716f6d968a-attempt0`, is active on GPU0. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-bc716f6d968a-attempt0`
completed successfully. Canonical geometry coverage is now 43/69 (VIB 18,
VQ 9, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-dda167111c69-attempt0`, is active on GPU0. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-dda167111c69-attempt0`
completed successfully. Canonical geometry coverage is now 44/69 (VIB 18,
VQ 10, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-e2a2df854657-attempt0`, is active on GPU0. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-e2a2df854657-attempt0`
completed successfully. Canonical geometry coverage is now 45/69 (VIB 18,
VQ 11, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed1-fbad3fefe2c8-attempt0`, is active on GPU0. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed1-fbad3fefe2c8-attempt0`
completed successfully. Canonical geometry coverage is now 46/69 (VIB 18,
VQ 12, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-41ca0ecf676c-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed2-41ca0ecf676c-attempt0`
completed successfully. Canonical geometry coverage is now 47/69 (VIB 18,
VQ 13, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-46365e60118c-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed2-46365e60118c-attempt0`
completed successfully. Canonical geometry coverage is now 48/69 (VIB 18,
VQ 14, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-5737ad2b9af1-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). VIB EoT
and collision tuning for VQ, quantized, and autoencoder remain active.

The next canonical quantized geometry run, `quantized-seed0-3a051827f0bc-attempt0`,
is also active on GPU3 alongside the quantized collision tuner with sufficient
memory headroom. Both analyses retain independent sessions and immutable run
directories.

## Phase 2 live execution checkpoint (2026-09-07, VQ and quantized geometry)

The canonical VQ geometry analysis for `vq-seed2-5737ad2b9af1-attempt0`
completed successfully. Canonical geometry coverage is now 49/69 (VIB 18,
VQ 15, quantized 1, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-cfdb81a38d33-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. The canonical
quantized geometry run `quantized-seed0-3a051827f0bc-attempt0` remains active on
GPU3. Canonical primary robustness coverage remains 3/69 (VQ 1, quantized 1,
autoencoder 1). VIB EoT and collision tuning for VQ, quantized, and
autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, quantized geometry completed)

The canonical quantized geometry analysis for `quantized-seed0-3a051827f0bc-attempt0`
completed successfully. Canonical geometry coverage is now 50/69 (VIB 18,
VQ 15, quantized 2, autoencoder 15). The next canonical quantized geometry run,
`quantized-seed0-707596ba87fb-attempt0`, is active on GPU3 alongside the
quantized collision tuner with sufficient memory headroom. The canonical VQ
geometry run `vq-seed2-cfdb81a38d33-attempt0` remains active on GPU0. Canonical
primary robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1).
VIB EoT and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry completed)

The canonical VQ geometry analysis for `vq-seed2-cfdb81a38d33-attempt0`
completed successfully. Canonical geometry coverage is now 51/69 (VIB 18,
VQ 17, quantized 2, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-e835014b162d-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. The canonical
quantized geometry run `quantized-seed0-707596ba87fb-attempt0` remains active on
GPU3. Canonical primary robustness coverage remains 3/69 (VQ 1, quantized 1,
autoencoder 1). VIB EoT and collision tuning for VQ, quantized, and
autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, quantized geometry continued)

The canonical quantized geometry analysis for `quantized-seed0-707596ba87fb-attempt0`
completed successfully. Canonical geometry coverage is now 52/69 (VIB 18,
VQ 17, quantized 3, autoencoder 15). The next canonical quantized geometry run,
`quantized-seed0-be6b4f9d89b3-attempt0`, is active on GPU3 alongside the
quantized collision tuner with sufficient memory headroom. The canonical VQ
geometry run `vq-seed2-e835014b162d-attempt0` remains active on GPU0. Canonical
primary robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1).
VIB EoT and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry continued)

The canonical VQ geometry analysis for `vq-seed2-e835014b162d-attempt0`
completed successfully. Canonical geometry coverage is now 53/69 (VIB 18,
VQ 18, quantized 3, autoencoder 15). The next canonical VQ geometry run,
`vq-seed2-ff6367c1f152-attempt0`, is active on GPU0 alongside the existing
autoencoder collision tuner with sufficient memory headroom. The canonical
quantized geometry run `quantized-seed0-be6b4f9d89b3-attempt0` remains active on
GPU3. Canonical primary robustness coverage remains 3/69 (VQ 1, quantized 1,
autoencoder 1). VIB EoT and collision tuning for VQ, quantized, and
autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, quantized geometry continued)

The canonical quantized geometry analysis for `quantized-seed0-be6b4f9d89b3-attempt0`
completed successfully. Canonical geometry coverage is now 54/69 (VIB 18,
VQ 18, quantized 4, autoencoder 15). The next canonical quantized geometry run,
`quantized-seed0-d505f26491f8-attempt0`, is active on GPU3 alongside the
quantized collision tuner with sufficient memory headroom. The canonical VQ
geometry run `vq-seed2-ff6367c1f152-attempt0` remains active on GPU0. Canonical
primary robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1).
VIB EoT and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, VQ geometry complete)

The canonical VQ geometry analysis for `vq-seed2-ff6367c1f152-attempt0`
completed successfully. Canonical geometry coverage is now 55/69 (VIB 18,
VQ 18, quantized 4, autoencoder 15); all canonical VQ geometry analyses are
complete. The canonical quantized geometry run
`quantized-seed0-d505f26491f8-attempt0` remains active on GPU3. Canonical
primary robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1).
VIB EoT and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, geometry complete)

The canonical quantized geometry analysis for `quantized-seed0-d505f26491f8-attempt0`
completed successfully. Canonical geometry coverage is now 56/69 (VIB 18,
VQ 18, quantized 5, autoencoder 15); all canonical geometry analyses are
complete for VIB, VQ, and autoencoder; 13 quantized geometry analyses remain.
The next canonical quantized geometry run, `quantized-seed0-fff63c8e4822-attempt0`,
is active on GPU3 alongside the quantized collision tuner with sufficient memory
headroom. Canonical primary robustness coverage remains 3/69 (VQ 1, quantized
1, autoencoder 1). The VIB EoT attack and collision tuning for VQ, quantized,
and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, quantized geometry continued)

The canonical quantized geometry analysis for `quantized-seed0-fff63c8e4822-attempt0`
completed successfully. Canonical geometry coverage is now 57/69 (VIB 18,
VQ 18, quantized 6, autoencoder 15). The next canonical quantized geometry run,
`quantized-seed1-48b755d18993-attempt0`, is active on GPU3 alongside the
quantized collision tuner with sufficient memory headroom. Canonical primary
robustness coverage remains 3/69 (VQ 1, quantized 1, autoencoder 1). The VIB
EoT attack and collision tuning for VQ, quantized, and autoencoder remain active.

## Phase 2 live execution checkpoint (2026-09-07, quantized geometry continued)

The canonical quantized geometry analysis for `quantized-seed1-48b755d18993-attempt0`
completed successfully. Its effective rank was 15.2654 and its median nearest
opposing-class distance was 0.5742. Canonical geometry coverage is now 58/69
(VIB 18, VQ 18, quantized 7, autoencoder 15). The next canonical quantized
geometry run, `quantized-seed1-5f687e2a7df1-attempt0`, is being advanced on GPU3.

An initial direct invocation of the generic `evaluate` command for
`autoencoder-seed0-78952625a171-attempt0` was rejected by the implementation's
explicit autoencoder safeguard because it lacked the reference classifier; it
created no evaluation artifact. The prescribed `evaluate-autoencoder` command
with the frozen seed-0 identity reference is being used for the retry. Existing
collision, VIB EoT, and all prior artifacts are preserved.

The corrected autoencoder clean evaluation for
`autoencoder-seed0-78952625a171-attempt0` completed with the frozen identity
seed-0 reference. Canonical clean-evaluation coverage is now 5/69. The next
quantized geometry analysis remains active on GPU3; the VIB EoT attack and
collision tuners remain active, and no prior artifacts were overwritten.

Two further prescribed autoencoder clean evaluations completed successfully:
`autoencoder-seed0-8254d43e96fe-attempt0` and
`autoencoder-seed0-a32175a9812e-attempt0`. Canonical clean-evaluation coverage
is now 7/69. The next quantized geometry analysis remains active on GPU3.

Two additional prescribed autoencoder clean evaluations completed successfully:
`autoencoder-seed1-c2c92a92b603-attempt0` and
`autoencoder-seed1-c6c3002bb1ff-attempt0`. Canonical clean-evaluation coverage
is now 13/69. The next quantized geometry analysis remains active on GPU3.

Two further prescribed autoencoder clean evaluations completed successfully:
`autoencoder-seed2-0af1a3e18640-attempt0` and
`autoencoder-seed2-0b1d033221fc-attempt0`. The autoencoder family now has
15/15 canonical clean evaluations, and overall clean-evaluation coverage is
15/69. The next quantized geometry analysis remains active on GPU3.

The prescribed clean evaluations for
`quantized-seed0-3a051827f0bc-attempt0` and
`vib-seed0-3b91e4ae81ac-attempt0` completed successfully. Canonical
clean-evaluation coverage is now 17/69. The next quantized geometry analysis
remains active on GPU3.

The prescribed clean evaluations for
`quantized-seed0-be6b4f9d89b3-attempt0` and
`vib-seed0-706eefdb59f5-attempt0` completed successfully. Canonical
clean-evaluation coverage is now 21/69. The next quantized geometry analysis
remains active on GPU3.

Two additional prescribed autoencoder clean evaluations completed successfully:
`autoencoder-seed1-5a921dff5d06-attempt0` and
`autoencoder-seed1-8a450388c714-attempt0`. Canonical clean-evaluation coverage
is now 11/69. The next quantized geometry analysis remains active on GPU3.

## Phase 2 live execution checkpoint (2026-09-07, latest verified)

The canonical quantized geometry analysis for `quantized-seed1-5f687e2a7df1-attempt0`
completed successfully. Its effective rank was 15.7107 and its median nearest
opposing-class distance was 0.6136. Canonical geometry coverage is now 59/69
(VIB 18, VQ 18, quantized 9, autoencoder 15). Canonical clean-evaluation
coverage is 19/69. The next canonical quantized geometry run is
`quantized-seed1-99701c2dfb6b-attempt0`; it is being advanced on GPU3. The VIB
EoT attack and collision tuners remain active.

The canonical quantized geometry analysis for `quantized-seed2-233d68935e71-attempt0`
completed successfully. Its effective rank was 15.4658 and its median nearest
opposing-class distance was 0.5865. Canonical geometry coverage is now 65/69
(VIB 18, VQ 18, quantized 14, autoencoder 15). Canonical clean-evaluation
coverage is 69/69 and invariance coverage is 45/69. The next canonical
quantized geometry run is `quantized-seed2-a189989cfbb0-attempt0`; the VIB EoT
attack and collision tuners remain active.

The canonical quantized geometry analysis for `quantized-seed1-e0992bbbee3b-attempt0`
completed successfully. Its effective rank was 14.7087 and its median nearest
opposing-class distance was 0.5541. Canonical geometry coverage is now 62/69
(VIB 18, VQ 18, quantized 11, autoencoder 15). Canonical clean-evaluation
coverage is 59/69 after the two latest VQ clean evaluations. The next canonical
quantized geometry run is `quantized-seed1-e6a5929852b4-attempt0`; it is being
advanced on GPU3. The VIB EoT and collision tuners remain active.

The canonical quantized geometry analysis for `quantized-seed1-9a74f7658339-attempt0`
completed successfully. Its effective rank was 15.0142 and its median nearest
opposing-class distance was 0.5594. Canonical geometry coverage is now 61/69
(VIB 18, VQ 18, quantized 11, autoencoder 15). Canonical clean-evaluation
coverage is 42/69. The next canonical quantized geometry run is
`quantized-seed1-e0992bbbee3b-attempt0`; it is being advanced on GPU3. The VIB
EoT attack and collision tuners remain active.

The prescribed clean evaluations for
`quantized-seed0-fff63c8e4822-attempt0` and
`vib-seed0-d5cb4d5a0cec-attempt0` completed successfully. Canonical
clean-evaluation coverage is now 25/69. The next canonical quantized geometry
analysis remains active on GPU3.

The prescribed clean evaluations for
`quantized-seed1-48b755d18993-attempt0` and
`vib-seed1-13c26a0e5b54-attempt0` completed successfully. Canonical
clean-evaluation coverage is now 27/69. The next canonical quantized geometry
analysis remains active on GPU3.

## Phase 2 live execution checkpoint (2026-09-07, latest verified)

The canonical quantized geometry analysis for `quantized-seed1-99701c2dfb6b-attempt0`
completed successfully. Its effective rank was 14.7973 and its median nearest
opposing-class distance was 0.5396. Canonical geometry coverage is now 60/69
(VIB 18, VQ 18, quantized 10, autoencoder 15). Canonical clean-evaluation
coverage is 30/69. The next canonical quantized geometry run is
`quantized-seed1-9a74f7658339-attempt0`; it is being advanced on GPU3. The VIB
EoT attack and collision tuners remain active.

## Phase 2 live execution checkpoint (2026-09-07, canonical marker reconciliation)

The canonical manifest currently contains 69/69 completed training runs and
69/69 completed clean evaluations. Geometry coverage is 65/69, with
`quantized-seed2-a189989cfbb0-attempt0` still running on GPU3 and three further
quantized geometry analyses queued. Invariance coverage is 51/69 after two
quantized analyses completed; the next pair,
`quantized-seed1-e6a5929852b4-attempt0` and
`quantized-seed2-03decf1791f9-attempt0`, is running on GPUs 0 and 2.
Primary robustness coverage remains 3/69, collision artifacts 3/69, square
checks 2/69, and transfer checks 2/69. The long VIB EoT attack and the
autoencoder collision tuner remain active. These counts are computed only from
the frozen canonical manifest and completed artifact markers; retries and
noncanonical directories are preserved but excluded from progress counts.

## Phase 2 live execution checkpoint (2026-09-07, latest verified)

The canonical quantized geometry analysis for
`quantized-seed2-a189989cfbb0-attempt0` completed successfully. Its effective
rank was 16.9126, median nearest opposing-class distance was 0.6177, and
separation ratio was 1.9134. Geometry coverage is now 66/69. Invariance
coverage is now 55/69 after four additional quantized analyses completed; the
next pair is running on GPUs 0 and 2. The next canonical geometry run,
`quantized-seed2-a83593a8c81c-attempt0`, is running on GPU3. Training and clean
evaluation remain 69/69; primary robustness remains 3/69, collision artifacts
3/69, square checks 2/69, and transfer checks 2/69. The VIB EoT attack and
collision attack/tuning jobs remain active.

## Phase 2 live execution checkpoint (2026-09-07, invariance stage complete)

All canonical invariance analyses are now complete: 69/69, including the
final VQ seed-2 pair. Training and clean evaluation remain 69/69. Geometry is
66/69, with the quantized seed-2 geometry analysis still active and two more
quantized geometry analyses queued. Primary robustness remains 3/69;
collision artifacts remain 3/69, with the VQ and quantized collision attacks
still running and the autoencoder collision tuner active. Square and transfer
diagnostics remain 2/69 each. No earlier run directories were overwritten.

## Phase 2 live execution checkpoint (2026-09-07, robustness queue active)

The canonical quantized geometry analysis for
`quantized-seed2-a83593a8c81c-attempt0` completed successfully. Its effective
rank was 15.1170, median nearest-opposing-class distance was 0.5946,
separation ratio was 1.9157, and median encoder-Jacobian norm was 6.5306.
Geometry coverage is now 67/69; the next quantized geometry run is active on
GPU3. Invariance remains complete at 69/69. Two new primary robustness jobs
are active: autoencoder seed-0 `autoencoder-seed0-78952625a171-attempt0` and
VQ seed-0 `vq-seed0-249cb22c2ac9-attempt0`. Primary robustness remains 3/69
until their completed markers are written. Collision artifacts remain 3/69;
the VIB EoT and collision workloads remain active.

## Phase 2 live execution checkpoint (2026-09-07, geometry nearly complete)

The canonical quantized geometry analysis for
`quantized-seed2-e613a58e70ab-attempt0` completed successfully. Its effective
rank was 15.2694, median nearest-opposing-class distance was 0.5674,
separation ratio was 1.9195, and median encoder-Jacobian norm was 6.9906.
Geometry coverage is now 68/69; the final missing geometry analysis,
`quantized-seed2-f18282ff7628-attempt0`, is running on GPU3. Invariance is
complete at 69/69. The VQ seed-0 and autoencoder seed-0 primary robustness
evaluations remain active; primary robustness is still 3/69. Collision
artifacts remain 3/69, and the VIB EoT attack remains active.

## Phase 2 live execution checkpoint (2026-09-07, geometry complete)

The final canonical quantized geometry analysis,
`quantized-seed2-f18282ff7628-attempt0`, completed successfully. Its effective
rank was 14.9162, median nearest-opposing-class distance was 0.5414,
separation ratio was 1.9745, and median encoder-Jacobian norm was 7.0331.
Geometry and invariance are now complete at 69/69. Primary robustness remains
3/69: the VQ seed-0 and autoencoder seed-0 attacks are still active, and a
quantized seed-0 primary attack has now started on GPU3. Collision artifacts
remain 3/69, with the VIB EoT and collision workloads still active.

## Phase 2 execution recovery checkpoint (2026-09-08)

The prior command-session processes did not survive the overnight monitoring
pause; their saved logs were empty and no completed artifacts were present for
the four unfinished primary attacks. No completed run was overwritten. The
unfinished VIB seed-0, VQ seed-0, autoencoder seed-0, and quantized seed-0
primary evaluations were restarted in persistent tmux sessions, along with
the unfinished autoencoder seed-0 collision-tuning job. Sessions are named
`ribs_vib_s0_20260908`, `ribs_vq_s0_20260908`, `ribs_ae_s0_20260908`,
`ribs_q_s0_20260908`, and `ribs_ae_collision_s0_20260908`; their logs are in
`outputs/phase2_tmux_logs/`. At restart, canonical coverage was training,
clean, geometry, and invariance 69/69; primary robustness 3/69; collision
3/69; square 2/69; and transfer 2/69.

## Phase 2 execution checkpoint (2026-09-08 10:30 EDT)

The canonical primary robustness evaluations for
`quantized-seed0-3a051827f0bc-attempt0` and
`vq-seed0-249cb22c2ac9-attempt0` completed without overwriting earlier
attempts. Their primary robustness directories are respectively
`evaluations/robustness-fde9c52f558a-attempt2/` and
`evaluations/robustness-d41148c88f73-attempt2/`. Both attack-correctness
diagnostics completed but reported a one-sample (3.125 percentage-point)
convergence drop on the pre-quantization latent surface at one radius; these
are retained as evaluation warnings and are not interpreted as robustness
findings. The quantized diagnostic failure is at rho=0.05 and the VQ failure
is at rho=0.20.

The required quantized Square Attack check was started after its primary
process released GPU3, in persistent session
`ribs_quantized_s0_square_20260908`, with log
`outputs/phase2_tmux_logs/quantized-seed0-3a051827f0bc-square.log`. At this
checkpoint, canonical coverage is training, clean, geometry, and invariance
69/69; primary robustness 5/69; collision 3/69; square 2/69; and transfer
2/69. The remaining active workloads are VIB seed-0 primary attack,
autoencoder seed-0 primary attack, and autoencoder seed-0 collision tuning.

The completed VQ seed-0 primary released GPU2, so the next canonical VQ
configuration, `vq-seed0-4028de1d4968-attempt0` (codebook size 64), was
started there in persistent session `ribs_vq_s0_k64_20260908`, with log
`outputs/phase2_tmux_logs/vq-seed0-4028de1d4968.log`. The quantized Square
Attack remains active on GPU3; the VIB, autoencoder, and autoencoder
collision-tuning workloads remain active on GPUs 1 and 0.

## Phase 2 diagnostic-audit tooling checkpoint (2026-09-08)

The discrete convergence diagnostics at the registered 32-sample size have
3.125 percentage-point resolution, so the observed one-sample drops are
retained as failed correctness diagnostics. To support an independent audit
without changing the frozen primary attack budgets, the CLI now provides
`attack-diagnostics` with explicit, provenance-recorded sample-count and
tolerance overrides. This command creates a new immutable diagnostic artifact;
it does not overwrite or reinterpret the original 32-sample failures. The
new parser and evaluation paths pass the focused tests, Ruff, and Black.

## Phase 2 live execution checkpoint (2026-09-08 11:34 EDT)

The VQ codebook-64 primary evaluation wrote its completed artifact at
`outputs/vq/vq-seed0-4028de1d4968-attempt0/evaluations/robustness-3df366939bda-attempt0/`
and released GPU2. The next pending canonical VQ evaluation,
`vq-seed0-664584b28f3b-attempt0` (codebook size 16), was started on GPU2 in
persistent session `ribs_vq_s0_k16_20260908`; its log is
`outputs/phase2_tmux_logs/vq-seed0-664584b28f3b.log`.

At this checkpoint, canonical coverage is training, clean, geometry, and
invariance 69/69; primary robustness 6/69; collision 3/69; Square Attack
2/69; and transfer 2/69. The VIB seed-0 primary attack, autoencoder seed-0
primary attack, autoencoder collision tuning, and quantized seed-0 Square
Attack remain active.

## Phase 2 evaluation recovery checkpoint (2026-09-08 12:15 EDT)

The VQ codebook-16 seed-0 primary evaluation attempt
`evaluations/robustness-17d09e45d548-attempt0/` failed before writing result
tables. The failure was the retained-loss check at rho=0.30. Reproduction
identified the exact sample as
`imagenette2-320/val/n03028079/n03028079_9371.JPEG`: a misclassified
candidate with CE 1.0371 was selected over a correctly classified clean point
with CE 1.1274, as required by the frozen success-priority rule. The prior
attempt, log, checkpoint, clean results, geometry, and invariance results are
preserved.

The attack result contract now records the highest CE encountered separately
from the success-priority selected candidate. This resolves the plan's
selection/retention edge case without weakening the primary attack or changing
its budgets. A regression test covers the case; the focused attack and
bottleneck suite passes 15/15 and Ruff passes. A corrected retry was started
on idle GPU2 in session `ribs_vq_s0_k16_retry_20260908`, with log
`outputs/phase2_tmux_logs/vq-seed0-664584b28f3b-retry.log`.

At 12:20 EDT, after GPU3 became idle, the next pending autoencoder primary
evaluation, `autoencoder-seed0-8254d43e96fe-attempt0` (dz=128), was started in
session `ribs_ae_s0_dz128_20260908`, with log
`outputs/phase2_tmux_logs/autoencoder-seed0-8254d43e96fe-dz128.log`.

## Phase 2 partial aggregation checkpoint (2026-09-08 13:08 EDT)

While the five remaining attack/tuning workers continue running, the
Phase 2-namespaced aggregation command completed successfully:

```text
PYTHONPATH=src .venv/bin/python -m ribs.cli aggregate --output-root outputs --experiment phase2 --include-families dimensional vib vq quantized autoencoder
```

The current checkpoint contains 20 summary rows, 20 seed-summary rows, 140
curve rows, 20 shared-clean-summary rows, and 48 geometry-correlation rows.
These files are an explicitly partial aggregation of valid completed
artifacts; they are not the final Phase 2 analysis and will be regenerated
after all canonical robustness, diagnostic, Square/transfer, collision, and
validation artifacts are complete:
`outputs/phase2_summary.parquet`, `outputs/phase2_summary.csv`,
`outputs/phase2_seed_summary.parquet`, `outputs/phase2_seed_summary.csv`,
`outputs/phase2_curves.parquet`, `outputs/phase2_shared_clean_summary.parquet`,
`outputs/phase2_shared_clean_correct.parquet`,
`outputs/phase2_geometry_correlations.parquet`,
`outputs/phase2_geometry_correlations.csv`, and
`outputs/phase2_inferential_exclusions.json`.

## Phase 2 live execution checkpoint (2026-09-08 13:08 EDT)

The corrected VQ codebook-16 seed-0 primary evaluation completed and wrote
the valid robustness artifact
`outputs/vq/vq-seed0-664584b28f3b-attempt0/evaluations/robustness-17d09e45d548-attempt1/`.
The canonical primary robustness count is now 7/69. The same process remains
active while it finishes the prescribed post-evaluation diagnostics, so GPU2
has not yet been reassigned. The remaining active workloads are VIB seed-0,
autoencoder dz=512 seed-0, autoencoder collision tuning, and autoencoder
dz=128 seed-0.

The VQ codebook-16 diagnostic process then exited cleanly and released GPU2.
Its required Square Attack follow-up was launched at 13:09 EDT in session
`ribs_vq_s0_k16_square_20260908_1309`, with log
`outputs/phase2_tmux_logs/vq-seed0-664584b28f3b-square.log`. The primary
evaluation and diagnostics remain immutable; this is a separate evaluation
attempt.

## Phase 2 external execution interruption checkpoint (2026-09-08 13:42 EDT)

The five workers that were active during the 13:38 monitoring interval are no
longer present: VIB beta=0 primary,
`autoencoder-seed0-78952625a171-attempt0` primary (dz=512),
`autoencoder-seed0-8254d43e96fe-attempt0` primary (dz=128), autoencoder
collision tuning for `autoencoder-seed0-10fedce2cf4e-attempt0`, and VQ
codebook-16 Square Attack for
`vq-seed0-664584b28f3b-attempt0`. The process IDs were absent from a direct
process check at 13:42 EDT, and the corresponding required final result
markers were not written. Their prior valid artifacts and all earlier failed
or partial attempts remain preserved.

GPU monitoring was unstable at the same time: `nvidia-smi` first reported
that it could not communicate with the NVIDIA driver, briefly returned a
utilization snapshot, and then failed again. The experiment logs for these
workers contain no terminal error text. This is recorded as an external
execution interruption, not as a robustness result. No worker was restarted
until GPU availability can be verified stable.

## Phase 2 execution-monitoring correction (2026-09-08 13:48 EDT)

The 13:42 process-absence conclusion above was a false negative caused by the
sandbox process namespace. A host-side read-only `nvidia-smi` and process check
at 13:48:49 EDT showed that all five workers were still alive:

* PID 1109786: autoencoder dz=512 primary attack on GPU 0;
* PID 1109790: VIB beta=0 primary attack on GPU 1;
* PID 1110313: autoencoder collision tuning on GPU 0;
* PID 1216030: VQ codebook-16 Square Attack on GPU 2; and
* PID 1189176: autoencoder dz=128 primary attack on GPU 3.

All four GPUs were actively occupied. A subsequent workspace-visible artifact
poll found no new completion markers or result files during the next interval,
so these jobs remained in progress. No jobs were relaunched and no prior
artifacts were changed. The earlier checkpoint is retained as an audit trail,
but should not be interpreted as evidence that the workers terminated.

## Phase 2 filesystem-recomputed recovery checkpoint (2026-09-09 12:12 EDT)

The canonical manifest was recomputed from completed run directories with
`phase2-run-dirs` and a new duplicate-provenance audit. It selects 69/69
registered Phase 2 training configurations. Clean evaluation, latent
artifacts, geometry, and invariance are complete for 69/69 canonical runs.
The current canonical evaluation coverage is 12/69 primary robustness runs,
1/69 collision runs, 4/36 Square Attack runs, and 3/36 nearest-capacity
transfer runs. These are filesystem counts, not historical estimates, and
the aggregate tables remain partial.

The five registered failed convergence diagnostics are exactly three VQ
seed-0 runs (`08c408be9bfb`, `249cb22c2ac9`, `664584b28f3b`) and two
quantized seed-0 runs (`3a051827f0bc`, `707596ba87fb`). Their original
32-sample artifacts remain untouched. Independent 128-sample audits with the
frozen 0.02 tolerance are queued in
`outputs/phase2_tmux_logs/phase2-diagnostic-audit-20260909T161200Z.log` and
will be interpreted separately from the primary results.

Persistent family queues are active in tmux and wait on the host-visible
workers before using a GPU: VIB on GPU1, VQ on GPU3, quantized on GPU2, and
autoencoder on GPU0. The existing workers continue to occupy all four GPUs
(including the pre-existing two VIB processes on GPU1); no duplicate
experiment was launched and no result directory was deleted or overwritten.
Validation remains `not_ready` until the missing canonical evaluations and
postprocessing artifacts are completed.
