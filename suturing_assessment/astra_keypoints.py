import json
import sys
from pathlib import Path

import cv2
import requests
from astra import API_URL, MODEL, REASONING_EFFORT, headers, image_url, submit
from frames import FRAME_SIZE, count_frames, original_size, read_frames, video_path

# Keypoint detection runs on frame_index % KEYPOINT_STEP == 0
KEYPOINT_STEP = 3
FPS = 30

WINDOW = 20
# Frames shared by consecutive windows; the answer nearer the window center is kept
OVERLAP = 2

EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent
    / "annotations"
    / "keypoint_fewshot_E003.json"
)

NEEDLE_KEYPOINTS = ["needle_tip", "needle_tail"]
INSTRUMENT_KEYPOINTS = ["clasper_1", "clasper_2", "head", "grasp"]

PROMPT = """# Task
In frames of a suturing training video (JIGSAWS) recorded with the da Vinci Surgical System on a bench-top model
(640x480 pixels), output the keypoints of the needle and the instruments as integer pixel coordinates [x, y].

# Objects
## Instruments
- The robotic instruments. Each instrument is the jaws, wrist and shaft together.
- At most two, in no particular order.
- The static wire-like metal object in the top-left background is not an instrument.

## Needle
- There is only one needle. It is a rigid, thin (a few pixels) silver curved needle shaped as part of a circle. Its
  shape is the same in every frame.
- The thread comes out of the tail of the needle. The thread is not part of the needle. The needle has a constant
  curvature, so the point where the curvature changes (e.g. where the curve starts to bend the other way) is the
  boundary between the needle and the thread.

## Instrument gripping the needle
- Gripping: an instrument holds the needle between its jaws. Touching or being near is not gripping.

# Keypoints
## Needle
- needle_tip: the sharp end of the needle.
- needle_tail: the end of the needle where the thread is attached.

## Each visible instrument
- clasper_1, clasper_2: the tips of the two jaws, in no particular order. When the jaws are closed, both are at almost
  the same place.
- head: the joint at the base of the jaws, around which the jaws open and close.
- grasp: where the center line of the needle crosses the center line of the jaws (the line along the jaws through the
  middle of their width). Only when the instrument grips the needle. Both instruments have one while both grip the
  needle. The gripped section of the needle is always covered by the jaws, so locate grasp from the needle seen on
  either side of the jaws.

## Hidden keypoints
- Give a keypoint only when it is visible. When it is hidden (by a jaw, the pad or the other instrument) or outside
  the image, give null. Do not guess hidden positions.
- The only exception is grasp, which is given whenever the instrument is seen gripping the needle.
- When only one jaw tip is visible, give it as clasper_1.

# Output
For every given frame, the needle keypoints and the keypoints of each visible instrument.

# Examples
Frames from another video, each with its keypoints. Keypoints that are not given are null."""

WINDOW_PROMPT = (
    "The following are {count} consecutive frames of one video, {interval:.1f} s apart, in time order, each "
    "preceded by its frame index. The needle is rigid and keeps its shape between frames, while the thread changes "
    "its shape as it moves. Use this difference as well. While an instrument grips the needle, the needle keeps its "
    "position relative to the jaws between frames, so gripping can be told over time even when the needle and the "
    "jaws overlap. Answer for every frame."
)

POINT_SCHEMA = {
    "anyOf": [
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        {"type": "null"},
    ]
}
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frames"],
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["frame_index", "needle", "instruments"],
                "properties": {
                    "frame_index": {"type": "integer"},
                    "needle": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": NEEDLE_KEYPOINTS,
                        "properties": {k: POINT_SCHEMA for k in NEEDLE_KEYPOINTS},
                    },
                    "instruments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": INSTRUMENT_KEYPOINTS,
                            "properties": {
                                k: POINT_SCHEMA for k in INSTRUMENT_KEYPOINTS
                            },
                        },
                    },
                },
            },
        }
    },
}


def windows(num_frames):
    """Split the sampled frames into windows starting at frame 0; the last one may be shorter"""
    sampled = list(range(0, num_frames, KEYPOINT_STEP))
    starts = range(0, max(len(sampled) - OVERLAP, 1), WINDOW - OVERLAP)
    return [sampled[start : start + WINDOW] for start in starts]


def upload_example_images(video_dir, examples, file_ids_path):
    """Upload the example frames once and return their file IDs"""
    # Every request must start with the same file IDs for the examples to be cached
    file_ids = json.loads(file_ids_path.read_text()) if file_ids_path.exists() else {}
    indices = [frame["frame_index"] for frame in examples["frames"]]
    images = read_frames(video_path(video_dir, examples["video"]), indices)
    for index in indices:
        name = f"{examples['video']}_{index}"
        if name in file_ids:
            continue
        _, buffer = cv2.imencode(".png", images[index])
        response = requests.post(
            f"{API_URL}/files",
            headers=headers(),
            data={"purpose": "vision"},
            files={"file": (f"{name}.png", buffer.tobytes())},
        )
        response.raise_for_status()
        file_ids[name] = response.json()["id"]
        file_ids_path.write_text(json.dumps(file_ids, indent=1))
    return file_ids


def example_label(frame):
    """Keypoints of an example frame in the output format, with missing keypoints as null"""
    return {
        "needle": {k: frame["needle"].get(k) for k in NEEDLE_KEYPOINTS},
        "instruments": [
            {k: instrument.get(k) for k in INSTRUMENT_KEYPOINTS}
            for instrument in frame["instruments"]
        ],
    }


def examples_content(video_dir, output_dir):
    """Prompt and few-shot examples, ending with a cache breakpoint"""
    examples = json.loads(EXAMPLES_PATH.read_text())
    file_ids = upload_example_images(
        video_dir, examples, output_dir / "keypoint_example_files.json"
    )
    content = [{"type": "input_text", "text": PROMPT}]
    for frame in examples["frames"]:
        name = f"{examples['video']}_{frame['frame_index']}"
        content.append(
            {
                "type": "input_text",
                "text": f"{name}: {json.dumps(example_label(frame))}",
            }
        )
        content.append(
            {"type": "input_image", "file_id": file_ids[name], "detail": "high"}
        )
    content.append(
        {
            "type": "input_text",
            "text": "End of examples.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    )
    return content


def request_body(examples, frame_indices, images):
    """Responses API request for one window"""
    window_content = [
        {
            "type": "input_text",
            "text": WINDOW_PROMPT.format(
                count=len(frame_indices), interval=KEYPOINT_STEP / FPS
            ),
        }
    ]
    for index in frame_indices:
        window_content.append({"type": "input_text", "text": f"frame {index}"})
        window_content.append(
            {
                "type": "input_image",
                "image_url": image_url(images[index]),
                "detail": "high",
            }
        )
    return {
        "model": MODEL,
        "reasoning": {"effort": REASONING_EFFORT},
        "prompt_cache_options": {"mode": "explicit"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "keypoints",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "input": [
            {"role": "user", "content": examples},
            {"role": "user", "content": window_content},
        ],
    }


def build(video_dir, video, output_dir):
    """Write the batch requests of a video"""
    path = video_path(video_dir, video)
    video_windows = windows(count_frames(path))
    images = read_frames(path, [index for window in video_windows for index in window])
    examples = examples_content(video_dir, output_dir)
    with open(output_dir / f"{video}.requests.jsonl", "w") as f:
        for window in video_windows:
            line = {
                "custom_id": f"{video}:{window[0]}",
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body(examples, window, images),
            }
            f.write(json.dumps(line) + "\n")
    meta = {
        "video": video,
        "original_size": original_size(path),
        "windows": video_windows,
    }
    (output_dir / f"{video}.windows.json").write_text(json.dumps(meta))
    print(f"{video}: {len(video_windows)} windows")


def parse_answers(line, window):
    """{frame_index: answer} of a batch output line, or None if it does not cover the window"""
    try:
        output = line["response"]["body"]["output"]
        text = "".join(
            c.get("text", "") for o in output for c in o.get("content") or []
        )
        answers = {a["frame_index"]: a for a in json.loads(text)["frames"]}
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return answers if set(answers) == set(window) else None


def fetch(video, output_dir):
    """Merge the window answers into per-frame keypoints in the original resolution"""
    batch_ids = json.loads((output_dir / f"{video}.batches.json").read_text())
    lines = []
    for batch_id in batch_ids:
        batch = requests.get(f"{API_URL}/batches/{batch_id}", headers=headers()).json()
        if batch["status"] != "completed":
            print(f"Error: batch {batch_id} is {batch['status']}")
            sys.exit(1)
        content = requests.get(
            f"{API_URL}/files/{batch['output_file_id']}/content", headers=headers()
        ).text
        lines += [json.loads(line) for line in content.splitlines()]

    meta = json.loads((output_dir / f"{video}.windows.json").read_text())
    scale_x = meta["original_size"][0] / FRAME_SIZE[0]
    scale_y = meta["original_size"][1] / FRAME_SIZE[1]
    windows_by_id = {f"{video}:{w[0]}": w for w in meta["windows"]}

    def to_original(points, keys):
        return {
            k: (
                None
                if points[k] is None
                else [round(points[k][0] * scale_x), round(points[k][1] * scale_y)]
            )
            for k in keys
        }

    results = {}
    answered = set()
    for line in lines:
        window = windows_by_id[line["custom_id"]]
        answers = parse_answers(line, window)
        if answers is None:
            continue
        answered.add(line["custom_id"])
        center = (len(window) - 1) / 2
        for position, index in enumerate(window):
            distance = abs(position - center)
            if index in results and results[index][0] <= distance:
                continue
            answer = answers[index]
            results[index] = (
                distance,
                {
                    "needle": to_original(answer["needle"], NEEDLE_KEYPOINTS),
                    "instruments": [
                        to_original(instrument, INSTRUMENT_KEYPOINTS)
                        for instrument in answer["instruments"]
                    ],
                },
            )
    failed = sorted(set(windows_by_id) - answered)
    output = {
        "video": video,
        "model": MODEL,
        "failed_windows": failed,
        "frames": {str(i): result for i, (_, result) in sorted(results.items())},
    }
    (output_dir / f"{video}.json").write_text(json.dumps(output))
    print(f"{video}: {len(results)} frames, {len(failed)} failed windows")


def main():
    """Build, submit or fetch the Astra keypoint detection batches of a video"""
    if len(sys.argv) < 4 or sys.argv[1] not in ("build", "submit", "fetch"):
        print(
            "Usage: astra_keypoints.py build VIDEO_DIR VIDEO OUTPUT_DIR\n"
            "       astra_keypoints.py {submit,fetch} VIDEO OUTPUT_DIR"
        )
        sys.exit(1)
    command = sys.argv[1]
    if command == "build":
        output_dir = Path(sys.argv[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        build(Path(sys.argv[2]), sys.argv[3], output_dir)
    elif command == "submit":
        submit(sys.argv[2], Path(sys.argv[3]))
    else:
        fetch(sys.argv[2], Path(sys.argv[3]))


if __name__ == "__main__":
    main()
