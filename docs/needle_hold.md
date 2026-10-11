# Needle Hold

The Needle Hold sub-skills of EASE are measured from the keypoints (`docs/keypoints.md`) in the Needle Hold
observation window of each stitch (`docs/events.md`).

## Frames

A frame is used when exactly one instrument has `grasp`. That instrument is the one holding the needle.

## Needle Hold Angle

- Keypoints: `grasp`, `needle_tip` and `head` of the holding instrument.
- Angle: between `grasp` → `needle_tip` and `grasp` → `head` (0–180°).
- Frames: the five frames with the largest |`grasp` → `needle_tip`| × |`grasp` → `head`|, where the needle and the
  jaws are least foreshortened.
- Value of a stitch: the median angle of these frames. The stitch is not measured if no frame has the keypoints.
