import json
import sys
from pathlib import Path

import numpy as np

# Percentile of the length in the image taken as the true length (docs/needle_hold.md)
TRUE_LENGTH_PERCENTILE = 95


def stitches(events):
    """(window start, entry, withdrawal_end) of each stitch, from the events of a video (docs/events.md)"""
    events = sorted(events, key=lambda e: e["frame_index"])
    out, previous_end, entry = [], 0, None
    for event in events:
        if event["event"] == "needle_entry":
            entry = event["frame_index"]
        elif event["event"] == "needle_withdrawal_end" and entry is not None:
            repositions = [
                e["frame_index"]
                for e in events
                if e["event"] == "needle_reposition_end"
                and previous_end <= e["frame_index"] <= entry
            ]
            start = max(repositions) if repositions else previous_end
            out.append((start, entry, event["frame_index"]))
            previous_end, entry = event["frame_index"], None
    return out


def holding_instrument(frame):
    """(grasp, head, jaw tip) of the only instrument with grasp, or None"""
    holders = [i for i in frame["instruments"] if i["grasp"] is not None]
    if len(holders) != 1 or holders[0]["head"] is None:
        return None
    holder = holders[0]
    claspers = [holder[k] for k in ("clasper_1", "clasper_2") if holder[k] is not None]
    if not claspers:
        return None
    return (
        np.array(holder["grasp"], float),
        np.array(holder["head"], float),
        np.mean(np.array(claspers, float), axis=0),
    )


def angle(a, b):
    """Angle between two vectors in degrees"""
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def needle_hold_angle(frames, windows):
    """Needle Hold Angle of each stitch: (median angle or None, number of frames)"""
    rows = []
    for start, entry, _ in windows:
        stitch_rows = []
        for index, frame in frames.items():
            if not start <= index <= entry or frame["needle"]["needle_tip"] is None:
                continue
            holder = holding_instrument(frame)
            if holder is None:
                continue
            grasp, head, jaw_tip = holder
            needle = np.array(frame["needle"]["needle_tip"], float) - grasp
            stitch_rows.append((needle, head - jaw_tip))
        rows.append(stitch_rows)
    jaws = [np.linalg.norm(jaw) for stitch_rows in rows for _, jaw in stitch_rows]
    if not jaws:
        return [(None, 0) for _ in windows]
    jaw_length = np.percentile(jaws, TRUE_LENGTH_PERCENTILE)
    out = []
    for stitch_rows in rows:
        if not stitch_rows:
            out.append((None, 0))
            continue
        needle_length = np.percentile(
            [np.linalg.norm(needle) for needle, _ in stitch_rows],
            TRUE_LENGTH_PERCENTILE,
        )
        angles = []
        for needle, jaw in stitch_rows:
            # The needle tip is farther from the camera than grasp; head is nearer than the jaw tip
            needle_depth = np.sqrt(max(needle_length**2 - needle @ needle, 0))
            jaw_depth = -np.sqrt(max(jaw_length**2 - jaw @ jaw, 0))
            angles.append(angle(np.r_[needle, needle_depth], np.r_[jaw, jaw_depth]))
        out.append((float(np.median(angles)), len(stitch_rows)))
    return out


def angle_score(value):
    """EASE score of a Needle Hold Angle: 3 (90-135°), 2 (135-150°) or 1"""
    if value is None:
        return None
    if 90 <= value <= 135:
        return 3
    return 2 if 135 < value <= 150 else 1


def main():
    """Needle Hold Angle of the stitches of the given videos, from keypoints and annotated events"""
    if len(sys.argv) < 5:
        print("Usage: needle_hold.py KEYPOINT_DIR ANNOTATIONS OUTPUT VIDEO [VIDEO ...]")
        sys.exit(1)
    keypoint_dir, annotations, output = map(Path, sys.argv[1:4])
    events = {
        v["video"]: v["events"] for v in json.loads(annotations.read_text())["videos"]
    }
    videos = []
    for video in sys.argv[4:]:
        result = json.loads((keypoint_dir / video / f"{video}.json").read_text())
        frames = {int(index): frame for index, frame in result["frames"].items()}
        windows = stitches(events[video])
        values = needle_hold_angle(frames, windows)
        videos.append(
            {
                "video": video,
                "stitches": [
                    {
                        "entry_frame_index": entry,
                        "withdrawal_end_frame_index": end,
                        "frames": count,
                        "needle_hold_angle": None if value is None else round(value, 1),
                        "score": angle_score(value),
                    }
                    for (_, entry, end), (value, count) in zip(windows, values)
                ],
            }
        )
    lines = ["{", '  "videos": [']
    for i, video in enumerate(videos):
        lines.append("    {")
        lines.append(f'      "video": {json.dumps(video["video"])},')
        lines.append('      "stitches": [')
        lines += [
            f"        {json.dumps(s)}" + ("," if j < len(video["stitches"]) - 1 else "")
            for j, s in enumerate(video["stitches"])
        ]
        lines.append("      ]")
        lines.append("    }" + ("," if i < len(videos) - 1 else ""))
    lines += ["  ]", "}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"{output}: {sum(len(v['stitches']) for v in videos)} stitches")


if __name__ == "__main__":
    main()
