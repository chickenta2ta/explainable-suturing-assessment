# Dataset

JIGSAWS Suturing (https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/), monocular (capture1) videos only.

## Split

Subjects are split 4 / 4 (`configs/split.yaml`). Each set has one expert, one intermediate and two novices
(self-reported), and the two sets have similar GRS distributions.

- Development set: B, E, F, I
- Test set: C, D, G, H

## Use of the development set

- Few-shot examples: `Suturing_E003`. Segmentation masks of the instruments, needle and thread on 20 frames.
  This video is not used for anything else.
- Segmentation evaluation: `Suturing_I002`. Segmentation masks of the instruments, needle and thread on frames
  sampled every 3 s, used to measure the segmentation IoU. A different subject from the few-shot examples.
- Method development and hyperparameter tuning: all development videos except `Suturing_E003` (19 videos).
  Hyperparameters are selected by leave-one-subject-out cross-validation over B, E, F and I.

The test set is evaluated once, with all settings fixed on the development set.
