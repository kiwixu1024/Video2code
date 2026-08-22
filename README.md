# Video2Code

Official evaluation code for
**[Video2Code: Generating Interactive Webpages from UI Videos via Action-Aware Revisit](https://arxiv.org/abs/2606.20711)**.

**Paper:** [Repository PDF](assets/video2code.pdf) · [arXiv](https://arxiv.org/abs/2606.20711)

Video2Code treats UI video-to-code generation as executable state-transition
recovery. A model first reads the full interaction video coarsely, predicts
action-critical temporal regions, revisits those regions at higher temporal
resolution with a clipping tool, and then generates executable HTML, CSS, and
JavaScript. This repository provides browser-based evaluation for real-world
and synthetic benchmarks.

## Overview

UI videos capture not only how a webpage looks, but also how it changes after a
click, text input, selection, or scroll. Sparse video sampling can miss these
short transitions. Video2Code therefore evaluates complete
**state-action-state** transitions.

```text
UI interaction video -> generated webpage -> browser replay -> scores
                                            |
                        +-------------------+-------------------+
                        |                                       |
            evaluation/real-world                  evaluation/synthesis
            sequential website tasks               segment/operation tasks
```

The browser verifier checks both rendered appearance and whether replaying the
demonstrated action produces the expected state transition.

## Repository layout

| Path | Purpose | Documentation |
|---|---|---|
| `evaluation/real-world/` | Evaluate WebVid2Code-Real with sequential browser interaction replay and initial-page retry. | [Real-world evaluation](evaluation/real-world/README.md) |
| `evaluation/synthesis/` | Evaluate WebVid2Code-Syn using its segment/operation hierarchy and segment-level state reset rules. | [Synthesis evaluation](evaluation/synthesis/README.md) |

Detailed commands, configuration variables, expected data layouts, and output
formats are documented in each evaluation component's README.

## Evaluation

Both benchmark subsets use three stages:

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
When a task cannot be completed from the current state, evaluation can reload
the initial generated page and retry that task to reduce cascading failures.

See [evaluation/real-world/README.md](evaluation/real-world/README.md).

### WebVid2Code-Syn

The synthesis subset contains controlled webpages. Operations execute in
`segment/operation` order. Page state accumulates within a segment and resets
between segments, matching the synthetic benchmark annotation structure.

See [evaluation/synthesis/README.md](evaluation/synthesis/README.md).

## Installation

Python 3.10 or newer is recommended. FFmpeg/ffprobe and a Playwright Chromium
installation are required for video processing and browser execution.

```bash
git clone https://github.com/kiwixu1024/Video2code.git
cd Video2code

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Install FFmpeg with the package manager for your operating system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

Each evaluation component supplies an `.env.example`. Copy the relevant file,
provide an OpenAI-compatible API endpoint and key, and keep the resulting local
configuration out of version control.

## Quick start

Run an evaluation from the corresponding directory:

```bash
cd evaluation/real-world   # or evaluation/synthesis
cp .env.example .env

python pipeline/stage1_generate.py
python pipeline/stage2_agent_replay.py
python pipeline/stage3_parse_results.py
```

Refer to the component README for the required benchmark data layout and
configuration options.

## Scope

This release provides browser-based benchmark evaluation. Model training
infrastructure, model weights, and benchmark datasets are not included.

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
