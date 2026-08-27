# SHOT (Source Hypothesis Transfer)

Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer
for Unsupervised Domain Adaptation", ICML 2020.
Official code: https://github.com/tim-learn/SHOT

**Paradigm**: non-adversarial, source-free: freeze source hypothesis, adapt target
encoder via information entropy minimization + pseudo-label self-supervised learning.

## Status
- [ ] TODO: implement train.py (source-free: train source classifier first, then adapt
      target encoder only; do NOT use target labels).
- [ ] Two-stage: (1) train source CE; (2) freeze clf, adapt encoder with
      `L = entropy + pseudo-label CE (weighted by confidence)`.
- [ ] Reuse common/data_loader.py + common/evaluate.py.

## Key formula (stage 2)
```
L = sum_H(-p log p) + CE(softmax(feat), pseudo_label) * weight
pseudo_label = argmax of source-hypothesis prediction
```
