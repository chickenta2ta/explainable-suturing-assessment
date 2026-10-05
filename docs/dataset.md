# Dataset

JIGSAWS Suturing (https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/), monocular (capture1) videos only.

## Split

Subjects are split 4 / 4 (`configs/split.yaml`). Each set has one expert, one intermediate and two novices
(self-reported), and the two sets have similar GRS distributions.

- Development set: B, E, F, I
- Test set: C, D, G, H

## Videos used

The few-shot examples and the segmentation evaluation use development subjects' trials that are not development
videos, one trial for each role.

- Segmentation few-shot examples: `Suturing_E003`. Segmentation masks of the instruments, needle and thread on 20
  frames. This video is not used for anything else.
- Event detection few-shot examples: `Suturing_B001`. Short clips around annotated events, covering all event types
  (`docs/events.md`). This video is not used for anything else.
- Development videos: two per subject, the lowest- and highest-GRS trials of each subject (ties broken by the lower
  trial number; 8 videos, `configs/split.yaml`). Used for method development and hyperparameter
  tuning, with hyperparameters selected by leave-one-subject-out cross-validation over B, E, F and I.
- Segmentation evaluation: `Suturing_I003`, a different subject from the few-shot examples. Segmentation masks of
  the instruments, needle and thread on frames sampled every 3 s, used to measure the segmentation IoU. This video is
  not used for anything else.
- Test videos: all 19 videos of C, D, G and H, evaluated once with all settings fixed on the development set.

Videos recorded at 320×240 (`Suturing_G001`–`G005`, `Suturing_H001`) are upscaled 2× to 640×480.

## Frame indexing and sampling

- `frame_index` is the 0-based order in which OpenCV (`cv2.VideoCapture.read()`) decodes the capture1 video from the
  start. It does not match the frame numbers of the JIGSAWS gesture labels or kinematics.
- Segmentation is run at 10 fps on the frames with `frame_index % 3 == 0`. Frames added later (e.g. at a higher rate
  around an event) keep their own `frame_index`, so existing results are unchanged.
