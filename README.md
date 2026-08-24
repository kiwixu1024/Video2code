# Video2Code

Official data-construction and evaluation code for
**[Video2Code: Generating Interactive Webpages from UI Videos via Action-Aware Revisit](https://arxiv.org/abs/2606.20711)**.

**Paper:** [Repository PDF](assets/video2code.pdf) · [arXiv](https://arxiv.org/abs/2606.20711)

Video2Code treats UI video-to-code generation as executable state-transition
recovery. A model first reads the full interaction video coarsely, predicts
action-critical temporal regions, revisits those regions at higher temporal
resolution with a clipping tool, and then generates executable HTML, CSS, and
JavaScript. This repository contains the pipeline for constructing the
action-aligned training data and the browser-based evaluation for real-world and
synthetic benchmarks.

## Overview

UI videos capture not only how a webpage looks, but also how it changes after a
click, text input, selection, or scroll. Sparse video sampling can miss these
short transitions. Video2Code therefore centers each training and evaluation
example on a complete **state-action-state** transition.

```text
                  Action-aligned training data

 URL or local HTML
        |
        v
 data/data_construct
 Automated browser exploration, interaction recording, action timelines
        |
        |  results/<id>.json + videos/<id>.mp4
        v
 data/data_process
 Timeline alignment, splitting, clipping, annotation, filtering, packaging
        |
        v
 Action-Aware Revisit supervision
 (full video -> temporal tool call -> focused clips -> webpage code)


                         Evaluation

 UI interaction video -> generated webpage -> browser replay -> scores
                                             |
                         +-------------------+-------------------+
                         |                                       |
             evaluation/real-world                  evaluation/synthesis
             sequential website tasks               segment/operation tasks
```

The paper formulates an action-critical segment as the interval containing the
UI state before an operation, the operation itself, and the resulting UI state.
The construction code records these temporal anchors and turns them into
temporal-clipping tool-call supervision. The evaluation code checks both rendered
appearance and whether replaying the demonstrated action produces the expected
state transition.

## Repository layout

| Path | Purpose | Documentation |
|---|---|---|
| `data/data_construct/` | Automatically explore public URLs or local HTML files and record filtered interaction videos with action-aligned timelines. | [Data construction](data/data_construct/README.md) |
| `data/data_process/` | Convert recordings into Action-Aware Revisit training records, including temporal clips, model annotations, filtering, and packaged shards. | [Data processing](data/data_process/README.md) |
| `evaluation/real-world/` | Generate webpages and evaluate WebVid2Code-Real with sequential browser interaction replay and initial-page retry. | [Real-world evaluation](evaluation/real-world/README.md) |
| `evaluation/synthesis/` | Evaluate WebVid2Code-Syn using its segment/operation hierarchy and segment-level state reset rules. | [Synthesis evaluation](evaluation/synthesis/README.md) |

Detailed commands, configuration variables, expected data layouts, and output
formats are intentionally kept in the README of each component.

## Data pipeline

### 1. Collect UI interaction videos

`data/data_construct` accepts either executable local HTML files or a list of
webpage URLs. An exploration agent launches each page in a Playwright browser,
discovers useful interactions, executes and validates them, and records the
browser viewport. A lightweight color marker provides precise temporal anchors
for action boundaries.

For every successful page, construction produces:

```text
dataset_root/
├── results/<id>.json       # operations and action_frames_timeline_filtered
└── videos/<id>.mp4         # filtered interaction video
```

Only the filtered video is retained, and its timestamps match
`action_frames_timeline_filtered`. See the
[construction guide](data/data_construct/README.md) for HTML and URL modes.

### 2. Build Action-Aware Revisit supervision

`data/data_process` consumes the videos and timelines, then:

1. converts action anchors into a normalized timeline dataset;
2. splits long recordings and reduces irrelevant temporal gaps;
3. extracts action-level clips and reference screenshots;
4. constructs temporal-clipping tool-call supervision;
5. generates reasoning and executable webpage targets;
6. merges and filters annotations; and
7. writes final JSONL records and packaged media shards.

The tool-call, reasoning, and code-generation annotation branches can run in
parallel before they are merged. See the
[processing guide](data/data_process/README.md) for the complete numbered
pipeline.

## Evaluation

The evaluation pipeline follows the browser-based functional verification
described in the paper. Both benchmark subsets use three stages:

1. generate executable webpage code from each UI video;
2. load the generated page and replay annotated user operations in a browser;
3. aggregate task-level results into page-wise and global scores.

Two complementary metrics are reported:

- **Visual similarity** measures whether the rendered state after interaction
  matches the target state in the video.
- **Functional correctness** measures whether the generated page reproduces the
  expected action-triggered state change.

Scores are first averaged over interaction tasks within each webpage and then
averaged across webpages, so pages containing more tasks do not dominate the
benchmark result.

### WebVid2Code-Real

The real-world subset contains manually recorded interactions from public
websites. Tasks follow the original interaction order and share browser state.
When a task cannot be completed from the current state, evaluation can reload the
initial generated page and retry that task to reduce cascading failures.

See [evaluation/real-world/README.md](evaluation/real-world/README.md).

### WebVid2Code-Syn

The synthesis subset contains controlled webpages obtained from automated
exploration trajectories. Operations execute in `segment/operation` order. Page
state accumulates within a segment and resets between segments, matching the
synthetic benchmark annotation structure.

See [evaluation/synthesis/README.md](evaluation/synthesis/README.md).

## Installation

Python 3.10 or newer is recommended. FFmpeg/ffprobe and a Playwright Chromium
installation are required for video processing and browser execution.

```bash
git clone <repository-url>
cd Video2code_Publish_ready

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Install FFmpeg with the package manager for your operating system, for example:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

Each component supplies an `.env.example` or example JSON configuration. Copy the
relevant example, provide an OpenAI-compatible API endpoint and key, and keep the
resulting local configuration out of version control.

The root `requirements.txt` installs the union of dependencies needed by all
components. Component-specific requirement files are also available for smaller
environments.

## Quick start

Construct a dataset from local HTML files or URLs:

```bash
cp data/data_construct/config.html.example.json \
   data/data_construct/config.local.json
export OPENAI_API_KEY=your-key

python data/data_construct/00_collect_interactions.py \
  --config data/data_construct/config.local.json
```

Process the constructed recordings:

```bash
export VIDEO2CODE_DATA_ROOT=/absolute/path/to/construction/output
python data/data_process/01_build_timeline_dataset.py
```

Then continue through the numbered stages documented in
[data/data_process/README.md](data/data_process/README.md).

Run an evaluation from the corresponding directory:

```bash
cd evaluation/real-world   # or evaluation/synthesis
cp .env.example .env

python pipeline/stage1_generate.py
python pipeline/stage2_agent_replay.py
python pipeline/stage3_parse_results.py
```

## Scope

This release provides data construction, data processing, temporal-tool
supervision preparation, and benchmark evaluation. Model training infrastructure
and model weights are not included in the directories covered by this release.

The current browser verifier focuses on clicks, text entry, selection, and
scrolling. Authentication-dependent pages, file uploads, drag-and-drop, canvas
manipulation, and workflows requiring external services are outside its primary
scope.

## Citation

If you use this repository, please cite:

```bibtex
@article{xu2026video2code,
  title   = {Video2Code: Generating Interactive Webpages from UI Videos via Action-Aware Revisit},
  author  = {Xu, Mingde and Yang, Zhen and Wang, Yan and Wang, Yu and Liu, Xijun and Dou, Zijun and Hong, Wenyi and Gu, Xiaotao and Xu, Bin and Tang, Jie},
  journal = {arXiv preprint arXiv:2606.20711},
  year    = {2026}
}
```
