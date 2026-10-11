# Keypoints

The needle and the instruments are annotated with keypoints on each frame. The instrument keypoints follow Du et al.
(IEEE TMI 2018) and the needle keypoints follow Chiu et al. (IROS 2022) and PinPoint (2026).

## Keypoint definitions

### Needle

| name | definition |
|---|---|
| needle_tip | The sharp end of the needle |
| needle_tail | The end of the needle where the thread is attached |

### Instruments

Each visible instrument (at most two, in no particular order) is annotated separately.

| name | definition |
|---|---|
| clasper_1, clasper_2 | Tips of the two jaws, in no particular order. When the jaws are closed, both are at almost the same place |
| head | Joint at the base of the jaws, around which the jaws open and close |
| grasp | Where the center line of the needle crosses the center line of the jaws (the line along the jaws through the middle of their width). Only when the jaws grip the needle |

The `shaft` and `end` keypoints of Du et al. are not used. The jaws can bend from the shaft at the wrist, and the
Needle Hold sub-skills of EASE are defined by the jaws, whose direction is given by the jaw tips and `head`.

## Annotation rules

- Only visible keypoints are annotated. Keypoints hidden by a jaw, the pad or the other instrument, or outside the
  image, are left out; their positions are not guessed.
- `grasp` is annotated when the jaws are seen gripping the needle, even though the gripped section itself is hidden
  by the jaws.
- `grasp` is annotated for every instrument that grips the needle, so both instruments have one during a reposition.
  Touching or being near the needle is not gripping.
- When only one jaw tip is visible, it is annotated as `clasper_1`.
