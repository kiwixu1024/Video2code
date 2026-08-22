"""Central configuration for the synthesis evaluation pipeline."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FRAMES_DIR = Path(os.getenv("VIDEO2CODE_SYNTH_FRAMES_DIR", str(DATA_DIR / "frames")))
VIDEO_DIR = DATA_DIR / "videos"
TIMELINE_FILE = DATA_DIR / "timeline.jsonl"
URLS_FILE = Path(os.getenv("VIDEO2CODE_SYNTH_URLS_FILE", str(DATA_DIR / "urls.jsonl")))

PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
WEBENV_PY = str(ROOT / "webenv-init" / "webenv.py")

API_KEY = os.getenv("VIDEO2CODE_API_KEY", "")
API_BASE = os.getenv("VIDEO2CODE_API_BASE", "")


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError(
            "No API key configured. Set VIDEO2CODE_API_KEY in the environment "
            "or in evaluation/synthesis/.env (see .env.example)."
        )
    return API_KEY


GENERATION_MODEL = os.getenv(
    "VIDEO2CODE_GENERATION_MODEL", "qwen3-vl-235b-a22b-thinking"
)
AGENT_MODEL = os.getenv("VIDEO2CODE_AGENT_MODEL", "gpt-5-2025-08-07")

# Stage 1: URL-hosted synthesis videos -> model responses.
GEN_PROMPT_FILE = PROMPTS_DIR / "generation.txt"
GEN_OUTPUT_DIR = RESULTS_DIR / "generation" / GENERATION_MODEL
GEN_MERGED_OUTPUT = GEN_OUTPUT_DIR / "merged.jsonl"
GEN_NUM_WORKERS = int(os.getenv("VIDEO2CODE_GEN_WORKERS", "10"))
GEN_MAX_TOKENS = int(os.getenv("VIDEO2CODE_GEN_MAX_TOKENS", "32768"))
GEN_STREAM = os.getenv("VIDEO2CODE_GEN_STREAM", "1") == "1"
GEN_MAX_RETRIES = 3
GEN_INITIAL_RETRY_DELAY = 1
GEN_MAX_RETRY_DELAY = 5

# Stage 2: generated HTML -> segment/operation replay.
GEN_HTML_DIR = RESULTS_DIR / "htmls" / GENERATION_MODEL
AGENT_HTMLS_DIR = Path(os.getenv("VIDEO2CODE_HTMLS_DIR", str(GEN_HTML_DIR)))
AGENT_OUTPUT_BASE = Path(
    os.getenv("VIDEO2CODE_EVAL_OUTPUT_DIR", str(RESULTS_DIR / "eval" / GENERATION_MODEL))
)
AGENT_PORT_BASE = int(os.getenv("VIDEO2CODE_PORT_BASE", "8765"))
AGENT_PORT_POOL_SIZE = int(os.getenv("VIDEO2CODE_PORT_POOL_SIZE", "120"))
AGENT_MAX_WORKERS = int(os.getenv("VIDEO2CODE_AGENT_WORKERS", "20"))
AGENT_MAX_RETRIES = 2
AGENT_FRAME_STEP = 1
AGENT_RECORD_VIDEO = True
AGENT_MODE = "batch"
SOLID_COLOR_PIXEL_RATIO = 0.97
SOLID_COLOR_QUANT_COLORS = 8

# Stage 3: aggregate one or more Stage 2 output directories. A comma-separated
# override makes multi-model evaluation possible without editing source files.
_parse_dirs = os.getenv("VIDEO2CODE_PARSE_OUTPUT_DIRS", "")
PARSE_OUTPUT_DIRS = (
    [str(Path(p.strip()).expanduser()) for p in _parse_dirs.split(",") if p.strip()]
    if _parse_dirs
    else [str(AGENT_OUTPUT_BASE)]
)
GLOBAL_SUMMARY_OUT = Path(
    os.getenv("VIDEO2CODE_GLOBAL_SUMMARY_OUT", str(RESULTS_DIR / "global_summary.json"))
)
