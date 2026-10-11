# Needle Hold

The Needle Hold sub-skills of EASE are measured from the keypoints (`docs/keypoints.md`) in the Needle Hold
observation window of each stitch (`docs/events.md`).

## Frames

A frame is used when exactly one instrument has `grasp`. That instrument is the one holding the needle.

## Needle Hold Angle

- Keypoints: `grasp`, `needle_tip` and `head` of the holding instrument.
- Angle: the angle between the line from `grasp` to `needle_tip` and the line from `grasp` to `head` (0–180°).
- Frames: the five frames with the largest product of the distance from `grasp` to `needle_tip` and the distance
  from `grasp` to `head`, where the needle and the jaws are least foreshortened.
- Value of a stitch: the median angle of these frames. The stitch is not measured if no frame has the keypoints.

## Depth of Needle Hold

- Keypoints: `grasp`, `head` and at least one of `clasper_1` and `clasper_2` of the holding instrument.
- Jaw tip: the midpoint of `clasper_1` and `clasper_2`, or the visible one of them.
- Depth: the distance from the jaw tip to the projection of `grasp` onto the line through the jaw tip and `head`,
  divided by the distance from the jaw tip to `head` (0 at the jaw tip, 1 at `head`).
- Frames: the five frames with the largest distance from the jaw tip to `head`.
- Value of a stitch: the median depth of these frames. The stitch is not measured if no frame has the keypoints.
