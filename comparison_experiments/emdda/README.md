# EM-DDA (Entropy Minimization + semantic-aligned Domain Adaptation)

Luo et al. (referenced in DAGCN paper as [30]).
Semantic alignment with entropy minimization for OCT domain adaptation.

**Paradigm**: non-adversarial (entropy min + semantic/class alignment).

## Status
- [ ] TODO: implement train.py
- [ ] No official repo; implement from DAGCN paper description:
      `L = L_cls(source) + λ·L_EM(target) + γ·L_semantic_alignment`
- [ ] This is the closest "EM-only" ancestor to our method; useful as an ablation anchor.

## Note
Not a strict requirement; implement only if time permits. DAGCN's EM-DDA comparison
numbers can be cited from the paper if we cannot reproduce.
