# Needle Hold

The Needle Hold sub-skills of EASE are measured from the keypoints and the needle mask (`docs/keypoints.md`) in the
Needle Hold observation window of each stitch (`docs/events.md`).

## Frames

A frame is used when exactly one instrument has `grasp`. That instrument is the one holding the needle.
The jaw tip is the midpoint of `clasper_1` and `clasper_2` of the holding instrument, or the visible one of them.

The true length of the jaws is the 95th percentile of the distance from the jaw tip to `head` in the image, over the
frames of the Needle Hold observation windows of all stitches of the video. The length is fixed by the
instrument, and foreshortening only shortens it, so a high percentile gives the true length while ignoring a few
erroneous keypoints. The windows are used because the instrument then works at the pad, at a similar distance from
the camera.

## Needle Hold Ratio

The grasp ratio s is the length along the needle from `needle_tip` to the grasp point divided by the length of the
needle (0 at the tip, 1 at the tail). In EASE, a hold between the 1:1 ratio hold and the 3/4:1/4 ratio hold is
acceptable, and the score 3 differs from 2 by whether the surgeon adjusts the grasp position to the depth of the
bite, which the grasp position alone does not show. Following AutoEASE, the score is in two classes: good (EASE 2
or 3) when 1/2 ≤ s ≤ 3/4, and poor (EASE 1) otherwise.

The needle is registered to each frame as a rigid circular arc.

- Model: one needle per video, a planar circular arc with radius r (pixels) and central angle Θ, with the tip at
  angle 0 and the tail at angle Θ. The grasp point of a stitch is at angle sΘ.
- Observations of a frame: `needle_tip`, `needle_tail` (when visible) and the needle polygons, and `grasp`,
  `clasper_1`, `clasper_2` and `head` of the holding instrument.
- Registration of a frame: with weak perspective, the arc is placed so that angle 0 is at `needle_tip` and angle sΘ
  is at `grasp`. The chord between them has the 3D length 2r sin(sΘ/2), which fixes its depth up to sign; s is
  impossible in the frame if the chord in the image is longer. The rotation of the needle plane about the chord and
  the sign of the depth are chosen to minimize the cost.
- Cost of a frame: the sum of three mean distances in pixels, without truncation:
  - from the points of the projected arc to the needle mask, leaving out the points inside the triangle of
    `clasper_1`, `clasper_2` and `head`, where the jaws hide the needle;
  - from the points of the needle mask to the projected arc;
  - from the projected tail to `needle_tail`, when it is visible.
- Cost of a stitch: the mean cost of the frames of its window that have `needle_tip` and the needle mask (Frames).
- Estimation: r and Θ of the video minimize the sum over the stitches of the least cost over s. Then s of each
  stitch minimizes its cost.
- Search: 64 rotations of the needle plane, 32 points on the arc, 64 points of the needle mask, r from 16 to 256
  pixels in steps of 1/8 octave, Θ from 10° to 350° in steps of 10°, and s in steps of 1/32 for r and Θ and 1/128
  for the final value.
- Value of a stitch: s. The stitch is not measured if no frame has the observations.

## Needle Hold Angle

The angle between the needle and the jaws in 3D, following the two arrows of the EASE figure (from the grasp point
to the needle tip and along the jaws): 90° when the needle is perpendicular to the jaws and 180° when it extends
along them.

- Keypoints: `grasp`, `needle_tip`, `head` and the jaw tip of the holding instrument.
- Vectors in the image: the needle from `grasp` to `needle_tip`, and the jaws from the jaw tip to `head`.
- True lengths: the true length of the jaws (Frames) and, for the needle, the 95th percentile of its length in the
  image over the frames of the stitch (the grasp position does not change in the window).
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
- Frames: the frames of the stitch in which the jaws look at least 0.8 times their true length (Frames), where
  the jaws are not strongly foreshortened.
- Value of a stitch: the median depth of these frames. The stitch is not measured if there is no such frame.

## Results

`results/dev/needle_hold_ratio.json`, `results/dev/needle_hold_angle.json` and
`results/dev/depth_of_needle_hold.json` hold the value and the EASE score of each stitch
(`suturing_assessment/needle_hold.py`, with the annotated events). The score of Needle Hold Ratio is good or poor.

```json
{
  "videos": [
    {
      "video": "Suturing_B002",
      "stitches": [
        {"entry_frame_index": 625, "withdrawal_end_frame_index": 951, "frames": 116, "needle_hold_angle": 147.8, "score": 2}
      ]
    }
  ]
}
```

- `frames`: the number of frames used. The value and the score are null when the stitch is not measured.
