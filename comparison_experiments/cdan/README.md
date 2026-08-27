# CDAN (Conditional Adversarial Domain Adaptation)

Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018.
Official code: https://github.com/thuml/CDAN (also in thuml/Transfer-Learning-Library)

**Paradigm**: adversarial (conditional discriminator with multi-linear map).

## Status
- [ ] TODO: implement train.py (mirror comparison_experiments/dann/train.py structure)
- [ ] Add conditional adversarial net: discriminator takes `feature x classifier_output` via
      multi-linear map (outer product).
- [ ] Reuse common/data_loader.py + common/evaluate.py (same as DANN).

## Key formula
```
D_cond = Discriminator( vec(feat ⊗ softmax(logit)) )
L_adv = CE(D_cond(concat_src), 0) + CE(D_cond(concat_tgt), 1)
```
