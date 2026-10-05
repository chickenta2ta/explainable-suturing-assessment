# Events

## Sub-skills

The following sub-skills of EASE (Haque et al., Urol Pract 2023) are assessed.

- Needle Repositions
- Needle Hold Ratio
- Needle Hold Angle
- Depth of Needle Hold
- Wrist Rotation (Needle Driving and Needle Withdrawal are assessed together)

## Event definitions

An event is an instant (one frame) in a video, as in event spotting (e.g. Precise Event Spotting, ECCV 2022) and
GolfDB (CVPR Workshops 2019).

| id | name | definition |
|---|---|---|
| 1 | needle_reposition_start | Both instruments grasp the needle while it is outside the tissue |
| 2 | needle_reposition_end | One instrument releases the needle |
| 3 | needle_entry | The needle tip enters the tissue. Annotated for every insertion |
| 4 | needle_withdrawal_end | The whole needle leaves the tissue after passing through to the other side |
| 5 | needle_retraction_end | The whole needle leaves the tissue after being pulled back to the entry side |

- The needle is either outside the tissue or in the tissue. It is outside the tissue at the start of the video.
- The parking place where the needle is stuck at the start and end of the video is not tissue; the needle there is
  outside the tissue.
- needle_entry puts the needle in the tissue, and needle_withdrawal_end or needle_retraction_end puts it back outside.
- Repositions are annotated only while the needle is outside the tissue.

When the instant is occluded, the occluded period is treated as in the tissue or held by both instruments, and the
event is placed as follows.

| event | frame |
|---|---|
| needle_reposition_start | Last frame in which the needle is seen held by one instrument |
| needle_reposition_end | First frame in which the needle is seen held by one instrument |
| needle_entry | Last frame in which the needle is seen outside the tissue |
| needle_withdrawal_end / needle_retraction_end | First frame in which the needle is seen outside the tissue |

## Observation windows

A stitch ends with needle_withdrawal_end, and the needle_entry immediately before it is the entry of the stitch.

Example with a retraction (`*` is the entry of the stitch):

```
reposition_start → reposition_end → entry → retraction_end
→ reposition_start → reposition_end → entry* → withdrawal_end
```

| Sub-skill | Observation window |
|---|---|
| Needle Repositions | Previous needle_withdrawal_end (start of the video for the first stitch) to the entry of the stitch |
| Needle Hold Ratio / Needle Hold Angle / Depth of Needle Hold | Last needle_reposition_end before the entry of the stitch to the entry of the stitch |
| Wrist Rotation | Entry of the stitch to needle_withdrawal_end |

- Needle Repositions is assessed by the number of needle_reposition_start in the window.
- If there is no needle_reposition_end between the previous needle_withdrawal_end (start of the video for the first
  stitch) and the entry of the stitch, the Needle Hold window starts at the previous needle_withdrawal_end (start of
  the video).
- The Needle Hold sub-skills are fixed once the needle is repositioned, so they can be measured at any frame in the
  window.
