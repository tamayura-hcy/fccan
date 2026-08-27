# SVDNA (Singular Value Decomposition Noise Adaptation)

Koch et al., "Noise transfer for unsupervised domain adaptation of retinal OCT images",
MICCAI 2022.
Official code: https://github.com/ValentinKoch/SVDNA

**Paradigm**: non-adversarial, OCT-specific. Transfers noise structure (via SVD) from
unlabeled target to labeled source images as augmentation; no network architecture change.

## Status
- [ ] TODO: implement train.py
- [ ] Core: decompose target images with SVD into components; synthesize source-style
      noisy images; train classifier on source + synthesized target-noise images.
- [ ] Most relevant to our low-frequency shortcut story (noise is low-frequency).

## Key idea
```
target SVD → noise components → apply to source images → train classifier
```
