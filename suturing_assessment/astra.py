import base64
import json
import os

import cv2
import requests

API_URL = "https://api.openai.com/v1"
MODEL = "gpt-6-astra"
REASONING_EFFORT = "medium"

JPEG_QUALITY = 95
# The Batch API accepts input files up to 200 MB
MAX_BATCH_BYTES = 150_000_000


def headers():
    """Authorization header from OPENAI_API_KEY"""
    return {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}


def image_url(image):
    """JPEG data URL of the image"""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def dumps_lines(result, key):
    """JSON text of a result, one field per line and one item of result[key] per line"""
    fields = []
    for name, value in result.items():
        if name == key and value:
            items = (
                [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()]
                if isinstance(value, dict)
                else [json.dumps(v) for v in value]
            )
            brackets = "{}" if isinstance(value, dict) else "[]"
            body = ",\n".join(f"    {item}" for item in items)
            fields.append(
                f"  {json.dumps(name)}: {brackets[0]}\n{body}\n  {brackets[1]}"
            )
        else:
            fields.append(f"  {json.dumps(name)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(fields) + "\n}\n"


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
