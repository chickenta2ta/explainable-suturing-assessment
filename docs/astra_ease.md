# Astra EASE Baseline

As a baseline, Astra scores the Needle Hold sub-skills of EASE and measures their quantities directly from the video
frames, with few-shot examples. It is compared with the measurements of `docs/needle_hold.md` on the same observation
windows.

## Requests

One request scores one stitch of the test videos. Stitches whose answer is not valid are submitted again.

- Target frames: the Needle Hold observation window of the stitch (`docs/events.md`), given by its frame indices.
- Context frames: 10 s before and after the target frames. They show the situation and are not scored.
- Frames: every frame with `frame_index % 6 == 0` (5 fps), 640×480, labeled with the frame index.
- Sub-skills: Needle Hold Ratio, Needle Hold Angle and Depth of Needle Hold, each scored 1, 2 or 3.

## Prompt

- Criteria: the descriptions of the three sub-skills in Figure 2 of the EASE paper (Haque et al., Urol Pract 2023),
  verbatim. The images of the figure are not given.
- Notes on the criteria:
  - Needle Hold Ratio: the position along the needle is 0 at the tip and 1 at the tail, and 1/2 to 3/4 is
    acceptable.
  - Needle Hold Angle: the angle in 3D between the needle from the grasp point to the tip and the jaws, 90° when
    perpendicular and 180° along the jaws.
  - Depth of Needle Hold: the distance of the needle from the jaw tip divided by the length of the jaws.
- Instructions:
  - Score the hold with which the needle enters the tissue. If the hold seems to change in the target frames, score
    the hold at the entry.
  - Score only the instrument holding the needle that enters the tissue, even if the other instrument touches the
    needle.
  - Besides the score, measure the quantity of each sub-skill in the notes: the position along the needle (0 to 1,
    to 0.001), the angle in degrees (0 to 180, to 0.1°) and the depth in the jaws (0 at the jaw tip, 1 at the base of
    the jaws, to 0.001).
  - Answer all three sub-skills. When the needle is hard to see, give the most likely value and express the
    uncertainty in the probabilities instead of declining.

## Examples

Three development stitches: the smallest set in which each score present in the development videos appears at least
once for each sub-skill, with at most one stitch per surgeon, and the first such set in (video, entry) order. No
development stitch has Needle Hold Angle 2.

| Video | Entry | Needle Hold Ratio | Needle Hold Angle | Depth of Needle Hold |
|---|---|---|---|---|
| Suturing_B003 | 246 | 3 | 1 | 1 |
| Suturing_E001 | 354 | 1 | 3 | 3 |
| Suturing_I004 | 3342 | 2 | 3 | 2 |

- Frames: every frame with `frame_index % 6 == 0` from 1 s before the Needle Hold observation window to 1 s after
  the entry, as the examples of event detection show 1 s before and after the event.
- Each example gives only the EASE scores, without values or reasons.

## Output

```json
{
  "needle_hold_ratio": {"value": 0.853, "score": 1, "probabilities": {"1": 0.6, "2": 0.3, "3": 0.1}, "reason": "..."},
  "needle_hold_angle": {"value": 121.5, "score": 3, "probabilities": {"1": 0.1, "2": 0.2, "3": 0.7}, "reason": "..."},
  "depth_of_needle_hold": {"value": 0.304, "score": 2, "probabilities": {"1": 0.2, "2": 0.5, "3": 0.3}, "reason": "..."}
}
```

- `value`: the measured quantity: the position along the needle (0 to 1, to 0.001), the angle in degrees (0 to 180,
  to 0.1°) or the depth in the jaws (0 at the jaw tip, 1 at the base of the jaws, to 0.001). It is answered
  separately from the score.
- `score`: 1, 2 or 3.
- `probabilities`: the probabilities of the three scores, summing to 1.
- `reason`: one or two sentences.

## Scores

`results/test/astra_ease/{video}.json` hold the scores given by Astra (`suturing_assessment/astra_ease.py`), one entry
per stitch with its entry frame (`entry_frame_index`).
