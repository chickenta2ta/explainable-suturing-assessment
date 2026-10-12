import json
import sys
from pathlib import Path

import numpy as np
from needle_hold_ratio import needle_hold_ratio, only_holder, ratio_score

# Percentile of the length in the image taken as the true length (docs/needle_hold.md)
TRUE_LENGTH_PERCENTILE = 95
# Depth of Needle Hold uses frames in which the jaws look at least this fraction of their true length
DEPTH_MIN_JAW = 0.8

SUB_SKILLS = {
    "needle_hold_ratio": (
        lambda f, w: needle_hold_ratio(f, w),
        lambda v: ratio_score(v),
        3,
    ),
    "needle_hold_angle": (
        lambda f, w: needle_hold_angle(f, w),
        lambda v: angle_score(v),
        1,
    ),
    "depth_of_needle_hold": (
        lambda f, w: depth_of_needle_hold(f, w),
        lambda v: depth_score(v),
        3,
    ),
}


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
    holder = only_holder(frame)
    if holder is None or holder["head"] is None:
        return None
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


def jaw_length(frames, windows):
    """True length of the jaws: the percentile of the jaw length in the image over the windows of all stitches"""
    lengths = []
    for index, frame in frames.items():
        if not any(start <= index <= entry for start, entry, _ in windows):
            continue
        holder = holding_instrument(frame)
        if holder is not None:
            lengths.append(np.linalg.norm(holder[1] - holder[2]))
    return np.percentile(lengths, TRUE_LENGTH_PERCENTILE) if lengths else None


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
    jaw_true = jaw_length(frames, windows)
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
            jaw_depth = -np.sqrt(max(jaw_true**2 - jaw @ jaw, 0))
            angles.append(angle(np.r_[needle, needle_depth], np.r_[jaw, jaw_depth]))
        out.append((float(np.median(angles)), len(stitch_rows)))
    return out


def depth_of_needle_hold(frames, windows):
    """Depth of Needle Hold of each stitch: (median depth or None, number of frames)"""
    jaw_true = jaw_length(frames, windows)
    out = []
    for start, entry, _ in windows:
        depths = []
        for index, frame in frames.items():
            if not start <= index <= entry:
                continue
            holder = holding_instrument(frame)
            if holder is None:
                continue
            grasp, head, jaw_tip = holder
            jaw = head - jaw_tip
            # Frames in which the jaws are not strongly foreshortened
            if np.linalg.norm(jaw) < DEPTH_MIN_JAW * jaw_true:
                continue
            depths.append(float((grasp - jaw_tip) @ jaw / (jaw @ jaw)))
        out.append((float(np.median(depths)), len(depths)) if depths else (None, 0))
    return out


def depth_score(value):
    """EASE score of a Depth of Needle Hold: 3 (distal 1/4 of the jaws), 2 (1/4-1/2) or 1"""
    if value is None:
        return None
    if value <= 0.25:
        return 3
    return 2 if value <= 0.5 else 1


def angle_score(value):
    """EASE score of a Needle Hold Angle: 3 (90-135°), 2 (135-150°) or 1"""
    if value is None:
        return None
    if 90 <= value <= 135:
        return 3
    return 2 if 135 < value <= 150 else 1


def write_results(path, videos, name):
    """Write the values of a sub-skill, one stitch per line"""
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
    path.write_text("\n".join(lines) + "\n")
    print(f"{path}: {sum(len(v['stitches']) for v in videos)} stitches")


def main():
    """Needle Hold sub-skills of the stitches of the given videos, from keypoints, needle masks and annotated events"""
    if len(sys.argv) < 5:
        print(
            "Usage: needle_hold.py KEYPOINT_DIR ANNOTATIONS OUTPUT_DIR VIDEO [VIDEO ...]"
        )
        sys.exit(1)
    keypoint_dir, annotations, output_dir = map(Path, sys.argv[1:4])
    events = {
        v["video"]: v["events"] for v in json.loads(annotations.read_text())["videos"]
    }
    results = {name: [] for name in SUB_SKILLS}
    for video in sys.argv[4:]:
        result = json.loads((keypoint_dir / f"{video}.json").read_text())
        frames = {int(index): frame for index, frame in result["frames"].items()}
        windows = stitches(events[video])
        for name, (measure, score, digits) in SUB_SKILLS.items():
            values = measure(frames, windows)
            results[name].append(
                {
                    "video": video,
                    "stitches": [
                        {
                            "entry_frame_index": entry,
                            "withdrawal_end_frame_index": end,
                            "frames": count,
                            name: None if value is None else round(value, digits),
                            "score": score(value),
                        }
                        for (_, entry, end), (value, count) in zip(windows, values)
                    ],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, videos in results.items():
        write_results(output_dir / f"{name}.json", videos, name)


if __name__ == "__main__":
    main()
