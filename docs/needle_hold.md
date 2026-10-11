# Needle Hold

The Needle Hold sub-skills of EASE are measured from the keypoints (`docs/keypoints.md`) in the Needle Hold
observation window of each stitch (`docs/events.md`).

## Frames

A frame is used when exactly one instrument has `grasp`. That instrument is the one holding the needle.
The jaw tip is the midpoint of `clasper_1` and `clasper_2` of the holding instrument, or the visible one of them.

## Needle Hold Angle

The angle between the needle and the jaws in 3D, following the two arrows of the EASE figure (from the grasp point
to the needle tip and along the jaws): 90° when the needle is perpendicular to the jaws and 180° when it extends
along them.

- Keypoints: `grasp`, `needle_tip`, `head` and the jaw tip of the holding instrument.
- Vectors in the image: the needle from `grasp` to `needle_tip`, and the jaws from the jaw tip to `head`.
- True lengths: the 95th percentile of the length in the image, over all frames of the video for the jaws (their
  length is fixed by the instrument) and over the frames of the stitch for the needle (the grasp position does not
  change in the window). Foreshortening only shortens a vector, so a high percentile gives the true length while
  ignoring a few erroneous keypoints.
- Depth: with weak perspective, the depth component of each vector is the square root of the true length squared
  minus the length in the image squared (0 if negative). The camera looks down on the pad, so the needle tip is
  farther from the camera than `grasp`, and `head` is nearer than the jaw tip.
- Angle: the angle between the two 3D vectors (0–180°).
- Value of a stitch: the median angle of all frames with the keypoints. The stitch is not measured if no frame has
  them.

## Depth of Needle Hold

- Keypoints: `grasp`, `head` and the jaw tip of the holding instrument.
- Depth: the distance from the jaw tip to the projection of `grasp` onto the line through the jaw tip and `head`,
  divided by the distance from the jaw tip to `head` (0 at the jaw tip, 1 at `head`).
- Frames: the frames in which the jaws look at least 0.8 times their true length (the 95th percentile of the jaw
  length in the image over all frames of the video, as for Needle Hold Angle), where the jaws are not strongly
  foreshortened.
- Value of a stitch: the median depth of these frames. The stitch is not measured if there is no such frame.
