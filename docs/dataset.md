# Dataset

JIGSAWS Suturing (https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/), monocular (capture1) videos only.

## Split

Subjects are split 4 / 4 (`configs/split.yaml`). Each set has one expert, one intermediate and two novices
(self-reported), and the two sets have similar GRS distributions.

- Development set (few-shot examples, method development, hyperparameter tuning): B, E, F, I
- Test set: C, D, G, H
