"""Stage 1 — generate synthesis HTML responses from annotated video URLs."""

import os
import sys
import json
import math
import time
import random
from multiprocessing import Process
from openai import OpenAI

# Make the synthesis package root importable when run from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline.html_extractor import save_html, save_jsonl_htmls

DATA_JSONL = str(config.URLS_FILE)
PROMPT_FILE = str(config.GEN_PROMPT_FILE)

OUTPUT_DIR = str(config.GEN_OUTPUT_DIR)
OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, "generate_worker")
INDEX_PREFIX = os.path.join(OUTPUT_DIR, "data_index_worker")
MERGED_OUTPUT = str(config.GEN_MERGED_OUTPUT)

NUM_WORKERS = config.GEN_NUM_WORKERS
MODEL_NAME = config.GENERATION_MODEL
MAX_TOKEN = config.GEN_MAX_TOKENS
BASE_URL = config.API_BASE

MAX_RETRIES = config.GEN_MAX_RETRIES
INITIAL_RETRY_DELAY = config.GEN_INITIAL_RETRY_DELAY
MAX_RETRY_DELAY = config.GEN_MAX_RETRY_DELAY

STREAM = config.GEN_STREAM


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_model(video_url, prompt):
    client = OpenAI(api_key=config.require_api_key(), base_url=BASE_URL)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": f"{video_url}"
                },
                "fps": 2
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]

    for attempt in range(MAX_RETRIES):
        try:
            if STREAM:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKEN,
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
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKEN,
                    timeout=600,
                )

                # print(response)
                return True, response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = min(
                    INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    MAX_RETRY_DELAY
                )
                time.sleep(delay)

    return False, f"Failed after {MAX_RETRIES} attempts"


def get_response(item, worker_id, output_file, index_file, error_ids):
    video_id = item["id"]
    video_url = item["video_url"]
    prompt = item["prompt"]

    success, result = call_model(video_url, prompt)

    record = {
        "id": video_id,
        "video_url": video_url,
        "success": success,
        "response": result,
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(str(video_id))

    if success:
        html_path = save_html(video_id, result, str(config.GEN_HTML_DIR))
        if html_path:
            print(f"[Worker {worker_id}] Done: {video_id} -> {html_path}")
        else:
            print(f"[Worker {worker_id}] Done (no valid HTML extracted): {video_id}")
    else:
        print(f"[Worker {worker_id}] Failed: {video_id} — {result}")
        error_ids.append(video_id)

def worker_fn(worker_id, items):
    output_file = f"{OUTPUT_PREFIX}_{worker_id}.jsonl"
    index_file = f"{INDEX_PREFIX}_{worker_id}.txt"

    # Resume: skip already-processed ids
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

    # ← 新增：断点续连时打印剩余数量
    remaining = [item for item in items if item["id"] not in done_ids]
    print(f"[Worker {worker_id}] Resume: {len(done_ids)} done, {len(remaining)} remaining")

    error_ids = []
    for item in remaining:  # ← 直接遍历 remaining，省去每次 if 判断
        get_response(item, worker_id, output_file, index_file, error_ids)

    if error_ids:
        print(f"[Worker {worker_id}] Errors: {error_ids}")

def build_items(prompt):
    items = []

    with open(DATA_JSONL, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Warning] JSON parse failed at line {line_no}: {e}")
                continue

            video_id = obj.get("id")
            video_url = obj.get("video_url")

            if video_id is None:
                print(f"[Warning] Missing id at line {line_no}")
                continue

            if not video_url:
                print(f"[Warning] Missing video_url at id={video_id}, line={line_no}")
                continue

            items.append({
                "id": str(video_id),
                "video_url": video_url,
                "prompt": prompt,
            })

    return items


def run_multiprocess(items):
    total = len(items)
    if total == 0:
        print("No items to process.")
        return

    chunk_size = math.ceil(total / NUM_WORKERS)
    processes = []

    for i in range(NUM_WORKERS):
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

    if not os.path.exists(OUTPUT_DIR):
        print(f"Output dir does not exist: {OUTPUT_DIR}")
        return

    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".jsonl") and "generate_worker" in fname:
            fpath = os.path.join(OUTPUT_DIR, fname)

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
        except Exception:
            return str(x.get("id", ""))

    merged.sort(key=sort_key)

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    success_count = sum(1 for x in merged if x.get("success"))
    print(f"\nMerged {len(merged)} records → {MERGED_OUTPUT}")
    print(f"Success: {success_count} / {len(merged)}")
    # Also covers resumed records generated before immediate extraction existed.
    save_jsonl_htmls(MERGED_OUTPUT, str(config.GEN_HTML_DIR))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompt = load_prompt()
    items = build_items(prompt)

    print(f"Total videos: {len(items)}, Workers: {NUM_WORKERS}")

    run_multiprocess(items)
    merge_results()
