import json
import sys
from pathlib import Path

import cv2
import requests
from astra import (
    API_URL,
    MODEL,
    REASONING_EFFORT,
    dumps_lines,
    headers,
    image_url,
    submit,
)
from frames import read_frames, video_path
from needle_hold import stitches

# Frames are sampled at 5 fps on frame_index % 6 == 0, as in event detection
FRAME_STEP = 6
FPS = 30

# Context frames before and after the Needle Hold observation window, and margin of the examples
CONTEXT_SECONDS = 10
EXAMPLE_MARGIN_SECONDS = 1

ANNOTATIONS = Path(__file__).resolve().parent.parent / "annotations"
ANNOTATIONS_PATH = ANNOTATIONS / "events_and_ease.json"
EXAMPLES_PATH = ANNOTATIONS / "ease_fewshot.json"

SUB_SKILLS = ["needle_hold_ratio", "needle_hold_angle", "depth_of_needle_hold"]
SCORES = ["1", "2", "3"]

PROMPT = """# Task
In frames of a suturing training video (JIGSAWS) recorded with the da Vinci Surgical System on a bench-top model
(640x480 pixels), score how the needle is held for one stitch with three sub-skills of EASE (End-to-End Assessment
of Suturing Expertise): Needle Hold Ratio, Needle Hold Angle and Depth of Needle Hold.

# Objects
## Instruments
- The robotic instruments. Each instrument is the jaws, wrist and shaft together. The instrument that holds the
  needle is the needle driver.
- At most two. The static wire-like metal object in the top-left background is not an instrument.

## Needle
- There is only one needle. It is a rigid, thin (a few pixels) silver curved needle shaped as part of a circle. Its
  shape is the same in every frame.
- The thread comes out of the tail of the needle.

# Criteria
Each sub-skill is scored 3 (ideal), 2 (adequate) or 1 (fail) by the following criteria of EASE.

## Needle Hold Ratio
- 3: Surgeon adjusts needle grasp position based on depth of bite (3/4:1/4 ratio hold or 1:1 ratio hold)
- 2: Surgeon does not adjust needle grasp position based on depth of bite AND Grasps needle within acceptable range
  (Between 1:1 Ratio Hold and 3/4:1/4 Ratio Hold)
- 1: Surgeon grasps needle outside of the acceptable range (< 1:1 Ratio Hold or > 3/4:1/4 Ratio Hold)

## Needle Hold Angle
- 3: Needle tip to Driver angle is slightly "obtuse" when grasping needle (90°-135°)
- 2: Needle tip to Driver angle is "obtuse" when grasping needle, but is approaching "flush" (135°-150°)
- 1: Needle tip to Driver angle is "flush" (150°-180°) or "acute" (<90°) when grasping needle OR Driver is angled and
  occludes target field during needle handling or driving

## Depth of Needle Hold
- 3: Needle is held at distal 1/4 jaws of needle driver
- 2: Needle is neither at tip nor goes beyond middle of jaws of driver (1/4-1/2 of driver)
- 1: Needle is held deeper than halfway in the jaws of the driver

## Notes on the criteria
- Needle Hold Ratio: the position along the needle is 0 at the tip and 1 at the tail, and 1/2 to 3/4 is acceptable.
- Needle Hold Angle: the angle in 3D between the needle from the grasp point to the tip and the jaws, 90° when
  perpendicular and 180° along the jaws.
- Depth of Needle Hold: the distance of the needle from the jaw tip divided by the length of the jaws.

# Instructions
- Score the hold with which the needle enters the tissue. If the hold seems to change in the target frames, score
  the hold at the entry.
- Score only the instrument holding the needle that enters the tissue, even if the other instrument touches the
  needle.
- Answer all three sub-skills. When the needle is hard to see, express the uncertainty in the probabilities instead
  of declining.

# Input
- Consecutive frames of one video, 0.2 s apart, in time order, each preceded by its frame index.
- The target frames run from the end of the last needle reposition before the entry (or, without one, the end of the
  previous stitch or the start of the video) to the needle entry, the frame in which the needle tip enters the
  tissue. The frames before and after are context, to tell what happens before and after, and are not scored.

# Output
For each sub-skill, the score, the probabilities of the three scores (summing to 1) and the reason in one or two
sentences.

# Examples
Stitches from other videos, each shown from up to 1 s before its target frames (none when they start at the start of
the video) to 1 s after them, with the EASE scores."""

STITCH_PROMPT = """# Target frames
Score the hold of the needle from frame {first} to frame {last} (the needle entry). The other frames are context."""

SUB_SKILL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "probabilities", "reason"],
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3]},
        "probabilities": {
            "type": "object",
            "additionalProperties": False,
            "required": SCORES,
            "properties": {s: {"type": "number"} for s in SCORES},
        },
        "reason": {"type": "string"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": SUB_SKILLS,
    "properties": {k: SUB_SKILL_SCHEMA for k in SUB_SKILLS},
}


def sampled(start, end):
    """Frames with frame_index % FRAME_STEP == 0 in [start, end)"""
    return list(range(-(-max(start, 0) // FRAME_STEP) * FRAME_STEP, end, FRAME_STEP))


def annotated_stitches(video):
    """{entry: (window start, EASE scores)} of the annotated stitches of a video"""
    videos = {v["video"]: v for v in json.loads(ANNOTATIONS_PATH.read_text())["videos"]}
    scores = {
        s["entry_frame_index"]: s["ease_scores"] for s in videos[video]["stitches"]
    }
    return {
        entry: (start, scores[entry])
        for start, entry, _ in stitches(videos[video]["events"])
    }


def example_frames(video, entry):
    """Window start and the frames of an example stitch, from 1 s before its window to 1 s after its entry"""
    start, _ = annotated_stitches(video)[entry]
    margin = EXAMPLE_MARGIN_SECONDS * FPS
    return start, sampled(start - margin, entry + margin + 1)


def upload_example_images(video_dir, examples, file_ids_path):
    """Upload the example frames once and return their file IDs"""
    # Every request must start with the same file IDs for the examples to be cached
    file_ids = json.loads(file_ids_path.read_text()) if file_ids_path.exists() else {}
    for example in examples["stitches"]:
        video = example["video"]
        _, indices = example_frames(video, example["entry_frame_index"])
        indices = [i for i in indices if f"{video}_{i}" not in file_ids]
        images = read_frames(video_path(video_dir, video), indices)
        for index in indices:
            _, buffer = cv2.imencode(".png", images[index])
            response = requests.post(
                f"{API_URL}/files",
                headers=headers(),
                data={"purpose": "vision"},
                files={"file": (f"{video}_{index}.png", buffer.tobytes())},
            )
            response.raise_for_status()
            file_ids[f"{video}_{index}"] = response.json()["id"]
            file_ids_path.write_text(json.dumps(file_ids, indent=1))
    return file_ids


def examples_content(video_dir, output_dir):
    """Prompt and few-shot examples, ending with a cache breakpoint"""
    examples = json.loads(EXAMPLES_PATH.read_text())
    file_ids = upload_example_images(
        video_dir, examples, output_dir / "ease_example_files.json"
    )
    content = [{"type": "input_text", "text": PROMPT}]
    for example in examples["stitches"]:
        video, entry = example["video"], example["entry_frame_index"]
        start, indices = example_frames(video, entry)
        content.append(
            {
                "type": "input_text",
                "text": f"Example from {video}, target frames {start}-{entry}:",
            }
        )
        for index in indices:
            content.append({"type": "input_text", "text": f"frame {index}"})
            content.append(
                {
                    "type": "input_image",
                    "file_id": file_ids[f"{video}_{index}"],
                    "detail": "high",
                }
            )
        scores = annotated_stitches(video)[entry][1]
        content.append(
            {
                "type": "input_text",
                "text": "Scores: "
                + ", ".join(f"{k} {scores[k]}" for k in SUB_SKILLS)
                + ".",
            }
        )
    content.append(
        {
            "type": "input_text",
            "text": "End of examples.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    )
    return content


def request_body(examples, start, entry, images):
    """Responses API request for one stitch"""
    content = [
        {"type": "input_text", "text": STITCH_PROMPT.format(first=start, last=entry)}
    ]
    for index in sorted(images):
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
                "name": "ease_scores",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "input": [
            {"role": "user", "content": examples},
            {"role": "user", "content": content},
        ],
    }


def load_result(video, output_dir):
    """Result of the video so far"""
    path = output_dir / f"{video}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"video": video, "model": MODEL, "failed_stitches": [], "stitches": []}


def build(video_dir, video, output_dir):
    """Write the batch requests of the stitches of a video that have no answer yet"""
    done = {s["entry_frame_index"] for s in load_result(video, output_dir)["stitches"]}
    todo = {e: s for e, (s, _) in annotated_stitches(video).items() if e not in done}
    if not todo:
        print(f"{video}: all stitches done")
        return
    context = CONTEXT_SECONDS * FPS
    frames = {e: sampled(s - context, e + context + 1) for e, s in todo.items()}
    images = read_frames(
        video_path(video_dir, video), [i for f in frames.values() for i in f]
    )
    examples = examples_content(video_dir, output_dir)
    with open(output_dir / f"{video}.requests.jsonl", "w") as f:
        for entry, start in sorted(todo.items()):
            stitch_images = {i: images[i] for i in frames[entry] if i in images}
            line = {
                "custom_id": f"{video}:{entry}",
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body(examples, start, entry, stitch_images),
            }
            f.write(json.dumps(line) + "\n")
    print(f"{video}: {len(todo)} stitches")


def parse_scores(line):
    """Scores of a batch output line, or None if it has no valid answer"""
    try:
        output = line["response"]["body"]["output"]
        text = "".join(
            c.get("text", "") for o in output for c in o.get("content") or []
        )
        answer = json.loads(text)
        return {
            k: {
                "score": int(answer[k]["score"]),
                "probabilities": {
                    s: float(answer[k]["probabilities"][s]) for s in SCORES
                },
                "reason": str(answer[k]["reason"]),
            }
            for k in SUB_SKILLS
        }
    except (KeyError, TypeError, ValueError):
        return None


def fetch(video, output_dir):
    """Add the answers of the submitted stitches to the result"""
    result = load_result(video, output_dir)
    failed = []
    for batch_id in json.loads((output_dir / f"{video}.batches.json").read_text()):
        batch = requests.get(f"{API_URL}/batches/{batch_id}", headers=headers()).json()
        if batch["status"] != "completed":
            print(f"Error: batch {batch_id} is {batch['status']}")
            sys.exit(1)
        content = requests.get(
            f"{API_URL}/files/{batch['output_file_id']}/content", headers=headers()
        ).text
        for line in map(json.loads, content.splitlines()):
            entry = int(line["custom_id"].split(":")[1])
            scores = parse_scores(line)
            if scores is None:
                failed.append(entry)
            else:
                result["stitches"].append({"entry_frame_index": entry, **scores})
    result["stitches"] = sorted(
        {s["entry_frame_index"]: s for s in result["stitches"]}.values(),
        key=lambda s: s["entry_frame_index"],
    )
    result["failed_stitches"] = sorted(failed)
    (output_dir / f"{video}.json").write_text(dumps_lines(result, "stitches"))
    print(f"{video}: {len(result['stitches'])} stitches, {len(failed)} failed")


def main():
    """Build, submit or fetch the Astra EASE scoring batch of the stitches of a video"""
    if len(sys.argv) < 4 or sys.argv[1] not in ("build", "submit", "fetch"):
        print(
            "Usage: astra_ease.py build VIDEO_DIR VIDEO OUTPUT_DIR\n"
            "       astra_ease.py {submit,fetch} VIDEO OUTPUT_DIR"
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
