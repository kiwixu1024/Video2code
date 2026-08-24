# Video2Code data pipeline

This directory contains the reproducible data-preparation pipeline used to turn
recorded browser interactions into packaged Video2Code training samples. Scripts
are numbered in execution order and use verb-based names so that their purpose is
visible without opening the source.

## Pipeline overview

| Step | Script | Input | Output |
|---:|---|---|---|
| 00 | `00_normalize_filtered_videos.py` | filtered videos mixed into `videos/` | normalized `videos_filtered/` |
| 01 | `01_build_timeline_dataset.py` | interaction JSON and source videos | `raw_dataset.jsonl` |
| 02 | `02_split_long_videos.py` | `raw_dataset.jsonl` | split videos and `raw_dataset_splitted.jsonl` |
| 03 | `03_extract_clips_and_screenshots.py` | split dataset and videos | operation clips, screenshots, and enriched JSONL |
| 04 | `04_add_synthetic_urls.py` | enriched JSONL | JSONL with synthetic public video URLs |
| 05 | `05_generate_tool_calls.py` | screenshots and enriched JSONL | video-understanding/tool-call generations |
| 06 | `06_generate_reasoning.py` | screenshots and enriched JSONL | reasoning generations |
| 07 | `07_generate_html.py` | screenshots and enriched JSONL | HTML generations |
| 08 | `08_merge_generations.py` | outputs from steps 05–07 | `merged_results.jsonl` |
| 09 | `09_extract_html_previews.py` | merged results | standalone HTML previews |
| 10 | `10_filter_invalid_html.py` | merged results and HTML previews | `merged_results_clean.jsonl` |
| 11 | `11_build_training_records.py` | cleaned merged results | final `data.jsonl` |
| 12 | `12_package_media_shards.py` | final JSONL and media | JSONL/TAR shards |
| 13 | `13_build_unistore.py` | JSONL/TAR shards | indexed UniStore dataset |

Steps 05, 06, and 07 are independent branches and may run in parallel. Step 08
is their synchronization point.

```text
00 -> 01 -> 02 -> 03 -> 04 -+-> 05 -+
                             +-> 06 -+-> 08 -> 09 -> 10 -> 11 -> 12 -> 13
                             +-> 07 -+
```

## Setup

Python 3.10 or newer is recommended. FFmpeg and ffprobe must be available on
`PATH`; Chromium is required by the HTML-validation step.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # run from the repository root
playwright install chromium
cp .env.example .env
```

Export the variables from `.env` in your shell before running steps 05–07. Never
commit a real API key.

Set `VIDEO2CODE_DATA_ROOT` to the shared dataset workspace. All stages resolve
their input and output paths from that root. Run scripts from the repository root:

```bash
python data/data_process/01_build_timeline_dataset.py
```

Large stages overwrite their declared JSONL output. Keep source data immutable,
and test a small copy before processing a full dataset. The model-call stages use
multiple processes; lower `NUM_WORKERS` if the provider rate-limits requests.

## Data layout

A practical workspace layout is:

```text
dataset_root/
├── results/                       # interaction timeline JSON files
├── videos/                        # source MP4 files
├── splitted_videos/               # step 02
├── clipped_videos/                # step 03
├── operation_screenshots/         # step 03
├── long_video_understanding_results/  # step 05
├── think_results/                 # step 06
├── code_results/                  # step 07
├── generated_htmls/               # step 09
└── unistore/
    ├── raw/                        # step 12
    └── final/                      # step 13
```

## Naming convention

- Two-digit prefixes encode stable execution order.
- Names describe the transformation, not an experiment or model version.
- Dataset/model versions belong in configuration or the output directory, not in
  script filenames.
- Generated data and logs should stay outside this source directory.
