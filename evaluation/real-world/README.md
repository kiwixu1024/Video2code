# Real-world evaluation

Evaluate generated webpages on the real-world subset of WebVideo2Code-Bench.
The pipeline generates HTML, replays the recorded interactions in a browser, and
reports visual similarity and functional correctness.

## Setup

Install the shared dependencies from the repository root:

```bash
pip install -r requirements.txt
playwright install chromium
cp evaluation/real-world/.env.example evaluation/real-world/.env
```

Set the API endpoint, key, models, and optional path overrides in `.env`.

## Run

```bash
cd evaluation/real-world
python pipeline/stage1_generate.py
python pipeline/stage2_agent_replay.py
python pipeline/stage3_parse_results.py
```

- Stage 1 generates and extracts `<id>.html` from each video.
- Stage 2 replays tasks sequentially and verifies the resulting browser states.
- Stage 3 aggregates task scores page-wise and then globally.

Use `python pipeline/stage2_agent_replay.py --replay` to retry pages that failed
to render. Real-world tasks preserve state in video order and use an initial-page
retry to reduce cascading failures.

Inputs belong under `data/`; runtime outputs are written under `results/`. See
`.env.example` and `config.py` for the available configuration variables.

