"""
Stage 1 — Video -> HTML generation.

Scans the input video directory, sends each recording to the generation model,
and — the moment a response comes back — extracts the HTML and saves it to
`config.GEN_HTML_DIR/<id>.html`. Raw responses are also appended to per-worker
.jsonl files (for resume + later re-extraction) and merged at the end.

This merges the former `video2code_inference_call_video.py` (inference) and
`extract_html.py` (HTML extraction) into a single pass: every generated output
flows straight into the save logic.

Run:
    python pipeline/stage1_generate.py
    # or: python -m pipeline.stage1_generate
"""

import os
import sys
import json
import base64
import math
import time
import random
from multiprocessing import Process

from openai import OpenAI

# Make the repo root importable whether run as a script or a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.main.config as config
from evaluation.main.pipeline_full_tasks_editted.html_extractor import save_html


def load_prompt() -> str:
    with open(config.GEN_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(video_path: str, prompt: str):
    """Call the generation model for a single video. Returns (success, text)."""
    client = OpenAI(api_key=config.require_api_key(), base_url=config.API_BASE)

    base64_video = encode_video_to_base64(video_path)
    data_url = f"data:video/mp4;base64,{base64_video}"

    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]

    for attempt in range(config.GEN_MAX_RETRIES):
        try:
            if config.GEN_STREAM:
                response = client.chat.completions.create(
                    model=config.GENERATION_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=config.GEN_MAX_TOKENS,
                    stream=True,
                    timeout=600,
                )
                collected = []
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            collected.append(delta.content)
                return True, "".join(collected).strip()
            else:
                response = client.chat.completions.create(
                    model=config.GENERATION_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=config.GEN_MAX_TOKENS,
                    timeout=600,
                )
                return True, response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < config.GEN_MAX_RETRIES - 1:
                delay = min(
                    config.GEN_INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    config.GEN_MAX_RETRY_DELAY,
                )
                time.sleep(delay)

    return False, f"Failed after {config.GEN_MAX_RETRIES} attempts"


def get_response(item, worker_id, output_file, index_file, error_ids):
    video_id = item["id"]
    video_path = item["video_path"]
    prompt = item["prompt"]

    success, result = call_model(video_path, prompt)

    record = {
        "id": video_id,
        "video_path": video_path,
        "success": success,
        "response": result,
    }

    # 1) Persist the raw response (for resume + later re-extraction).
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(index_file, "w") as f:
        f.write(str(video_id))

    # 2) Each output immediately flows into the HTML save logic.
    if success:
        html_path = save_html(video_id, result, str(config.GEN_HTML_DIR))
        if html_path:
            print(f"[Worker {worker_id}] Done: {video_id} -> {html_path}")
        else:
            print(f"[Worker {worker_id}] Done (no valid HTML extracted): {video_id}")
    else:
        print(f"[Worker {worker_id}] Failed: {video_id} - {result}")
        error_ids.append(video_id)


def worker_fn(worker_id, items):
    output_file = os.path.join(config.GEN_OUTPUT_DIR, f"generate_worker_{worker_id}.jsonl")
    index_file = os.path.join(config.GEN_OUTPUT_DIR, f"data_index_worker_{worker_id}.txt")

    # Resume: skip already-processed ids.
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        done_ids.add(obj.get("id"))
                    except Exception:
                        pass

    remaining = [item for item in items if item["id"] not in done_ids]
    print(f"[Worker {worker_id}] Resume: {len(done_ids)} done, {len(remaining)} remaining")

    error_ids = []
    for item in remaining:
        get_response(item, worker_id, output_file, index_file, error_ids)

    if error_ids:
        print(f"[Worker {worker_id}] Errors: {error_ids}")


def build_items(prompt: str):
    items = []
    for fname in sorted(os.listdir(config.VIDEO_DIR)):
        if not fname.lower().endswith(".mp4"):
            continue
        video_id = os.path.splitext(fname)[0]
        items.append({
            "id": video_id,
            "video_path": os.path.join(str(config.VIDEO_DIR), fname),
            "prompt": prompt,
        })
    return items


def run_multiprocess(items):
    total = len(items)
    chunk_size = math.ceil(total / config.GEN_NUM_WORKERS)
    processes = []

    for i in range(config.GEN_NUM_WORKERS):
        start = i * chunk_size
        if start >= total:
            break
        chunk = items[start: start + chunk_size]
        p = Process(target=worker_fn, args=(i, chunk))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def merge_results():
    merged = []
    for fname in sorted(os.listdir(config.GEN_OUTPUT_DIR)):
        if fname.endswith(".jsonl") and "generate_worker" in fname:
            fpath = os.path.join(config.GEN_OUTPUT_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            merged.append(json.loads(line))
                        except Exception:
                            pass

    def sort_key(x):
        try:
            return int(x["id"])
        except (ValueError, KeyError):
            return x.get("id", "")

    merged.sort(key=sort_key)

    merged_output = os.path.join(config.GEN_OUTPUT_DIR, "merged.jsonl")
    with open(merged_output, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    success_count = sum(1 for x in merged if x.get("success"))
    print(f"\nMerged {len(merged)} records -> {merged_output}")
    print(f"Success: {success_count} / {len(merged)}")
    print(f"HTML files: {config.GEN_HTML_DIR}")


def main():
    os.makedirs(config.GEN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.GEN_HTML_DIR, exist_ok=True)

    prompt = load_prompt()
    items = build_items(prompt)
    print(f"Total videos: {len(items)}, Workers: {config.GEN_NUM_WORKERS}")
    print(f"Model: {config.GENERATION_MODEL}")

    run_multiprocess(items)
    merge_results()


if __name__ == "__main__":
    main()
