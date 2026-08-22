# Synthesis evaluation

Evaluate generated webpages on the synthetic subset of WebVideo2Code-Bench.
The pipeline generates HTML, replays annotated operations in a browser, and
reports visual similarity and functional correctness.



## Run

```bash
cd evaluation/synthesis
python pipeline/stage1_generate.py
python pipeline/stage2_agent_replay.py
python pipeline/stage3_parse_results.py
```

- Stage 1 generates webpages from the annotated video inputs.
- Stage 2 evaluates operations in `segment/operation` order.
- Stage 3 aggregates operation, task, directory, and global scores.

State accumulates within a segment and resets between segments. Use
`python pipeline/stage2_agent_replay.py --replay` to retry render failures.

Inputs belong under `data/`; runtime outputs are written under `results/`. See
`.env.example` and `config.py` for the available configuration variables.

