# Dataset

JIGSAWS Suturing (https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/), monocular (capture1) videos only.

## Split

Subjects are split 4 / 4 (`configs/split.yaml`). Each set has one expert, one intermediate and two novices
(self-reported), and the two sets have similar GRS distributions.

- Development set: B, E, F, I
- Test set: C, D, G, H

## Videos used

- Few-shot examples: `Suturing_E003`. Segmentation masks of the instruments, needle and thread on 20 frames.
  This video is not used for anything else.
- Development videos: two per subject, the lowest- and highest-GRS trials of each subject (ties broken by the lower
  trial number; 8 videos, `configs/split.yaml`). Used for method development and hyperparameter
  tuning, with hyperparameters selected by leave-one-subject-out cross-validation over B, E, F and I.
- Segmentation evaluation: `Suturing_I002`, one of the development videos and a different subject from the few-shot
  examples. Segmentation masks of the instruments, needle and thread on frames sampled every 3 s, used to measure
  the segmentation IoU.
- Test videos: all 19 videos of C, D, G and H, evaluated once with all settings fixed on the development set.

Videos recorded at 320×240 (`Suturing_G001`–`G005`, `Suturing_H001`) are upscaled 2× to 640×480.
