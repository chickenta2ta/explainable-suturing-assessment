import base64
import json
import os
import sys
from pathlib import Path

import cv2
import requests

from frames import (
    FRAME_SIZE,
    FRAME_STEP,
    count_frames,
    original_size,
    read_frames,
    video_path,
)

API_URL = "https://api.openai.com/v1"
MODEL = "gpt-6-astra"
REASONING_EFFORT = "medium"

WINDOW = 20
# Frames shared by consecutive windows; the answer nearer the window center is kept
OVERLAP = 2

JPEG_QUALITY = 95
# The Batch API accepts input files up to 200 MB
MAX_BATCH_BYTES = 150_000_000

EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent
    / "annotations"
    / "segmentation_fewshot_E003.json"
)

PROMPT = """# Task
In frames of a suturing training video (JIGSAWS) recorded with the da Vinci Surgical System on a bench-top model
(640x480 pixels), output the outlines of the needle and the instruments as polygons (lists of integer pixel coordinates [x, y]).

# Objects
## Instruments
- The robotic instruments. Each instrument (jaws, wrist and shaft together) is one polygon. Include the shaft up to the image border.
- At most two.
- The static wire-like metal object in the top-left background is not an instrument.

## Needle
- There is only one needle. It is a rigid, thin (a few pixels) silver curved needle shaped as part of a circle. Its shape is the same in every frame.
- Give one polygon per visible part (at most two). Do not include hidden parts.
  - When held in a jaw, it can appear as two parts, one on each side of the jaw.
  - When inserted in the pad, the tip side and the tail side can appear as two parts.
- The thread comes out of the tail of the needle. Do not include the thread. The needle has a constant curvature, so the point where the
  curvature changes (e.g. where the curve starts to bend the other way) is the boundary between the needle and the thread.

## Thread
- The examples also show thread polygons ("thread") to help tell it from the needle. Do not output the thread.

# Needle state
## needle_state: whether the needle is in the pad
- inserted: part of the needle is inside the pad (while being driven or pulled out, left in the pad, or stuck in the pad as its parking
  place at the start or end)
- not_inserted: the needle is visible and no part of it is inside the pad (in the air, or lying on the pad)
- not_visible: no part of the needle is visible (out of view, or completely hidden by a jaw or the pad)

## holder: which instrument grips the needle between its jaws
- left / right: the instrument whose shaft enters from the left / right border of the image
- both: both instruments (while it is handed over)
- none: no jaw grips it (touching or being near is not gripping). Also none when needle_state is not_visible

# Examples
Each example gives the needle, instruments and thread polygons of the frame."""

WINDOW_PROMPT = (
    "The following are {count} consecutive frames of the same video, 0.1 s apart, in time order. "
    "The needle is rigid and keeps its shape between frames, while the thread changes its shape as it moves. "
    "Use this difference as well. Answer for every frame."
)

POLYGON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    },
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
                "required": [
                    "frame_index",
                    "needle_state",
                    "holder",
                    "needle",
                    "instruments",
                ],
                "properties": {
                    "frame_index": {"type": "integer"},
                    "needle_state": {
                        "type": "string",
                        "enum": ["inserted", "not_inserted", "not_visible"],
                    },
                    "holder": {
                        "type": "string",
                        "enum": ["left", "right", "both", "none"],
                    },
                    "needle": {"type": "array", "items": POLYGON_SCHEMA},
                    "instruments": {"type": "array", "items": POLYGON_SCHEMA},
                },
            },
        }
    },
}


def headers():
    """Authorization header from OPENAI_API_KEY"""
    return {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}


def windows(num_frames):
    """Split the sampled frames into windows starting at frame 0; the last one may be shorter"""
    sampled = list(range(0, num_frames, FRAME_STEP))
    starts = range(0, max(len(sampled) - OVERLAP, 1), WINDOW - OVERLAP)
    return [sampled[start : start + WINDOW] for start in starts]


def image_url(image):
    """JPEG data URL of the image"""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


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


def examples_content(video_dir, output_dir):
    """Prompt and few-shot examples, ending with a cache breakpoint"""
    examples = json.loads(EXAMPLES_PATH.read_text())
    file_ids = upload_example_images(
        video_dir, examples, output_dir / "example_files.json"
    )
    content = [{"type": "input_text", "text": PROMPT}]
    for frame in examples["frames"]:
        name = f"{examples['video']}_{frame['frame_index']}"
        labels = {key: frame[key] for key in ("needle", "instruments", "thread")}
        content.append({"type": "input_text", "text": f"{name}: {json.dumps(labels)}"})
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
        {"type": "input_text", "text": WINDOW_PROMPT.format(count=len(frame_indices))}
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
                "name": "segmentation",
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


def submit(video, output_dir):
    """Submit the requests as batches whose input files stay under MAX_BATCH_BYTES"""
    chunks = [[]]
    size = 0
    for line in open(output_dir / f"{video}.requests.jsonl", "rb"):
        if size + len(line) > MAX_BATCH_BYTES and chunks[-1]:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line)
    batch_ids = []
    for number, lines in enumerate(chunks):
        response = requests.post(
            f"{API_URL}/files",
            headers=headers(),
            data={"purpose": "batch"},
            files={"file": (f"{video}.{number}.jsonl", b"".join(lines))},
        )
        response.raise_for_status()
        response = requests.post(
            f"{API_URL}/batches",
            headers=headers(),
            json={
                "input_file_id": response.json()["id"],
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            },
        )
        response.raise_for_status()
        batch_ids.append(response.json()["id"])
    (output_dir / f"{video}.batches.json").write_text(json.dumps(batch_ids))
    print(f"{video}: submitted {len(batch_ids)} batches")


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
    """Merge the window answers into per-frame results in the original resolution"""
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

    def to_original(polygons):
        return [
            [[round(x * scale_x), round(y * scale_y)] for x, y in p] for p in polygons
        ]

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
                    "needle_state": answer["needle_state"],
                    "holder": answer["holder"],
                    "needle": to_original(answer["needle"]),
                    "instruments": to_original(answer["instruments"]),
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
    """Build, submit or fetch the Astra segmentation batches of a video"""
    if len(sys.argv) < 4 or sys.argv[1] not in ("build", "submit", "fetch"):
        print(
            "Usage: astra_segmentation.py build VIDEO_DIR VIDEO OUTPUT_DIR\n"
            "       astra_segmentation.py {submit,fetch} VIDEO OUTPUT_DIR"
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
