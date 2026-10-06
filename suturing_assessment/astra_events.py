import json
import sys
from pathlib import Path

import cv2
import requests
from astra_segmentation import (
    API_URL,
    MODEL,
    REASONING_EFFORT,
    headers,
    image_url,
    submit,
)
from frames import count_frames, read_frames, video_path

# Event detection runs at 5 fps on frame_index % 6 == 0, a subset of the segmentation frames
EVENT_STEP = 6
FPS = 30

# Each window answers the events of its target frames and sees context frames before and after
TARGET_SECONDS = 20
CONTEXT_SECONDS = 10

EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent / "annotations" / "event_fewshot_B001.json"
)

EVENTS = [
    "needle_reposition_start",
    "needle_reposition_end",
    "needle_entry",
    "needle_withdrawal_end",
    "needle_retraction_end",
]

PROMPT = """# Task
In frames of a suturing training video (JIGSAWS) recorded with the da Vinci Surgical System on a bench-top model
(640x480 pixels), find the needle handling events and the frames in which they happen.

# Objects
## Instruments
- The robotic instruments. Each instrument is the jaws, wrist and shaft together.
- At most two. The left instrument is the one whose shaft enters from the left border of the image, and the right
  instrument the one whose shaft enters from the right border.
- The static wire-like metal object in the top-left background is not an instrument.

## Needle
- There is only one needle. It is a rigid, thin (a few pixels) silver curved needle shaped as part of a circle. Its
  shape is the same in every frame.
- The thread comes out of the tail of the needle.

# States
## Whether the needle is in the tissue
- The needle is either in the tissue or outside it. It is outside at the start of the video.
- In the tissue: part of the needle is inside the pad at the suture line (around the incision and its entry and exit
  marks). The needle is in the tissue only when part of its tip is actually hidden inside the pad. Placing the tip on
  the surface, pressing it against the surface or sliding it along the surface is still outside.
- Parking place: a needle stuck in the pad away from the suture line, as its parking place at the start or end, is
  outside the tissue (the parking place is not tissue).

## Instrument gripping the needle
- Gripping: an instrument holds the needle between its jaws. Touching or being near is not gripping.

# Events
- needle_reposition_start: both instruments grip the needle while it is outside the tissue.
- needle_reposition_end: one instrument releases the needle after a reposition.
- needle_entry: the needle tip enters the tissue. The tip often rests on the target point for a while; those frames
  are before the entry.
- needle_withdrawal_end: the needle, having passed through to the exit side, leaves the tissue completely.
- needle_retraction_end: after an entry, the needle is pulled back to the entry side without its tip coming out on
  the exit side, and leaves the tissue completely. Without an entry (the tip going into the tissue), there is no
  retraction.

# Order
- needle_entry and the end of a pass (needle_withdrawal_end or needle_retraction_end) alternate. After an entry the
  needle stays in the tissue until exactly one withdrawal_end or retraction_end, and only then can the next entry
  happen.
- Repositions happen only while the needle is outside the tissue. Each needle_reposition_start is followed by its
  needle_reposition_end before the next reposition_start or entry.
- One instrument pulls the needle out on the exit side and the other one drives it, so after a
  needle_withdrawal_end there is at least one reposition before the next entry.
- After the last stitch the needle is carried to the parking place. No events are given after the last
  needle_withdrawal_end.

# Frames to answer
- The frames are 0.2 s apart, so the instant of an event often falls between two given frames.
- Answer only with frame indices of the given frames: the first frame showing the new state.
- When the instant is occluded, treat the occluded period as in the tissue or gripped by both instruments, and answer
  the following frame among the given frames:
  - needle_reposition_start: the last frame in which the needle is seen gripped by one instrument
  - needle_reposition_end: the first frame in which the needle is seen gripped by one instrument
  - needle_entry: the last frame in which the needle is seen outside the tissue
  - needle_withdrawal_end / needle_retraction_end: the first frame in which the needle is seen outside the tissue

# Input
- Consecutive frames of one video, 0.2 s apart, in time order, each preceded by its frame index.
- Answer events only for the frames in the given range (the target frames). The frames before and after are context,
  to tell what happened before and after.
- The events detected from the start of the video up to the target frames are given, with whether the needle is in
  the tissue at the start of the target frames (derived from them). They are earlier results and may contain errors;
  if the images clearly show otherwise, follow the images.
- An event in the list may be visible in the context frames. Do not answer it again as an event of the target
  frames. Answer an event that newly happens in the target frames even if the list has one of the same type (for
  example, two repositions can happen in a row).

# Output
All events that happen in the target frames, each as a type and a frame index. Empty if none.

# Examples
Clips from another video, one per event type: frames 0.2 s apart from 1 s before to 1 s after the annotated events.
Each clip gives the annotated frame and the frame to answer by the rules above."""

WINDOW_PROMPT = """# Events detected so far
From the start of the video up to frame {start}: {events}.
At frame {start} the needle is {state} the tissue.

# Target frames
Answer only the events from frame {first} to frame {last}. The other frames are context."""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event", "frame_index"],
                "properties": {
                    "event": {"type": "string", "enum": EVENTS},
                    "frame_index": {"type": "integer"},
                },
            },
        }
    },
}


def sampled(start, end):
    """Frames with frame_index % EVENT_STEP == 0 in [start, end)"""
    return list(range(-(-start // EVENT_STEP) * EVENT_STEP, end, EVENT_STEP))


def windows(num_frames):
    """[context start, target start, target end, context end] of each window; targets do not overlap"""
    target, context = TARGET_SECONDS * FPS, CONTEXT_SECONDS * FPS
    return [
        [
            max(0, start - context),
            start,
            min(start + target, num_frames),
            min(start + target + context, num_frames),
        ]
        for start in range(0, num_frames, target)
    ]


def upload_example_images(video_dir, examples, file_ids_path):
    """Upload the example frames once and return their file IDs"""
    # Every request must start with the same file IDs for the examples to be cached
    file_ids = json.loads(file_ids_path.read_text()) if file_ids_path.exists() else {}
    indices = [
        i for clip in examples["clips"] for i in sampled(clip["start"], clip["end"] + 1)
    ]
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
        video_dir, examples, output_dir / "event_example_files.json"
    )
    content = [{"type": "input_text", "text": PROMPT}]
    for clip in examples["clips"]:
        indices = sampled(clip["start"], clip["end"] + 1)
        content.append(
            {
                "type": "input_text",
                "text": f"Example from {examples['video']}, frames {indices[0]}-{indices[-1]}:",
            }
        )
        for index in indices:
            content.append({"type": "input_text", "text": f"frame {index}"})
            content.append(
                {
                    "type": "input_image",
                    "file_id": file_ids[f"{examples['video']}_{index}"],
                    "detail": "high",
                }
            )
        annotated = "; ".join(
            f"{e['event']} at frame {e['frame_index']} "
            f"(answer: frame {sampled(e['frame_index'], e['frame_index'] + EVENT_STEP)[0]})"
            for e in clip["events"]
        )
        content.append(
            {"type": "input_text", "text": f"Annotated events: {annotated}."}
        )
    content.append(
        {
            "type": "input_text",
            "text": "End of examples.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    )
    return content


def window_text(window, events):
    """Events detected so far, the needle state at the target start and the target range"""
    _, start, end, _ = window
    inside = False
    for event in events:
        if event["event"] == "needle_entry":
            inside = True
        elif event["event"] in ("needle_withdrawal_end", "needle_retraction_end"):
            inside = False
    target = sampled(start, end)
    return WINDOW_PROMPT.format(
        start=target[0],
        events="; ".join(f"{e['event']} at frame {e['frame_index']}" for e in events)
        or "none",
        state="in" if inside else "outside",
        first=target[0],
        last=target[-1],
    )


def request_body(examples, window, events, images):
    """Responses API request for one window"""
    content = [{"type": "input_text", "text": window_text(window, events)}]
    for index in sampled(window[0], window[3]):
        content.append({"type": "input_text", "text": f"frame {index}"})
        content.append(
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
                "name": "events",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "input": [
            {"role": "user", "content": examples},
            {"role": "user", "content": content},
        ],
    }


def load_result(video_dir, video, output_dir):
    """Result of the video so far, created with its windows on the first call"""
    path = output_dir / f"{video}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "video": video,
        "model": MODEL,
        "windows": windows(count_frames(video_path(video_dir, video))),
        "done": 0,
        "failed_windows": [],
        "events": [],
    }


def build(video_dir, video, output_dir):
    """Write the batch request of the next window of a video; False when all windows are done"""
    result = load_result(video_dir, video, output_dir)
    (output_dir / f"{video}.json").write_text(json.dumps(result))
    if result["done"] == len(result["windows"]):
        print(f"{video}: all {result['done']} windows done")
        return False
    window = result["windows"][result["done"]]
    images = read_frames(video_path(video_dir, video), sampled(window[0], window[3]))
    examples = examples_content(video_dir, output_dir)
    line = {
        "custom_id": f"{video}:{window[1]}",
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body(examples, window, result["events"], images),
    }
    (output_dir / f"{video}.requests.jsonl").write_text(json.dumps(line) + "\n")
    print(f"{video}: window {result['done'] + 1}/{len(result['windows'])}")
    return True


def parse_events(line, window):
    """Events of the target frames in a batch output line, or None if it has no valid answer"""
    try:
        output = line["response"]["body"]["output"]
        text = "".join(
            c.get("text", "") for o in output for c in o.get("content") or []
        )
        events = json.loads(text)["events"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return sorted(
        (e for e in events if window[1] <= e["frame_index"] < window[2]),
        key=lambda e: e["frame_index"],
    )


def record(video, output_dir, line):
    """Add the events of a window answer to the result; False if the answer is not valid"""
    path = output_dir / f"{video}.json"
    result = json.loads(path.read_text())
    window = result["windows"][result["done"]]
    events = parse_events(line, window)
    if line["custom_id"] != f"{video}:{window[1]}" or events is None:
        result["failed_windows"].append(window[1])
        path.write_text(json.dumps(result))
        return False
    result["events"] += events
    result["done"] += 1
    path.write_text(json.dumps(result))
    return True


def fetch(video, output_dir):
    """Add the answer of the submitted window to the result"""
    for batch_id in json.loads((output_dir / f"{video}.batches.json").read_text()):
        batch = requests.get(f"{API_URL}/batches/{batch_id}", headers=headers()).json()
        if batch["status"] != "completed":
            print(f"Error: batch {batch_id} is {batch['status']}")
            sys.exit(1)
        content = requests.get(
            f"{API_URL}/files/{batch['output_file_id']}/content", headers=headers()
        ).text
        for line in content.splitlines():
            ok = record(video, output_dir, json.loads(line))
            print(
                f"{video}: {'recorded' if ok else 'failed'} {json.loads(line)['custom_id']}"
            )


def main():
    """Build, submit or fetch the Astra event detection batch of the next window of a video"""
    if len(sys.argv) < 4 or sys.argv[1] not in ("build", "submit", "fetch"):
        print(
            "Usage: astra_events.py build VIDEO_DIR VIDEO OUTPUT_DIR\n"
            "       astra_events.py {submit,fetch} VIDEO OUTPUT_DIR"
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
