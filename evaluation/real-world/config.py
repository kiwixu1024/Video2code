"""
Central configuration for the Video2Code pipeline.

All secrets (API keys) and tunable settings are read from environment variables,
which are loaded from a local `.env` file if present (see `.env.example`).
Nothing in this file should contain a real API key — keep secrets in `.env`,
which is git-ignored.

Every path is anchored to the repository root so the scripts run correctly
regardless of the current working directory.
"""

import os
from pathlib import Path

# Optional: load variables from a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"          # input .mp4 recordings
FRAMES_DIR = DATA_DIR / "frames"         # per-task extracted frames (frames/<id>/timeline_*/)
TIMELINE_FILE = DATA_DIR / "timeline.jsonl"

PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

WEBENV_PY = str(ROOT / "webenv-init" / "webenv.py")


# ============================================================
# API credentials (read from environment / .env — never hard-code them here)
# ============================================================
API_KEY = os.getenv("VIDEO2CODE_API_KEY", "")
API_BASE = os.getenv("VIDEO2CODE_API_BASE", "")


def require_api_key() -> str:
    """Return the configured API key or raise a clear error if it is missing."""
    if not API_KEY:
        raise RuntimeError(
            "No API key configured. Set VIDEO2CODE_API_KEY in your environment "
            "or in a .env file (see .env.example)."
        )
    return API_KEY


# ============================================================
# Models
# ============================================================
# Stage 1: video -> HTML generation model.
GENERATION_MODEL = os.getenv("VIDEO2CODE_GENERATION_MODEL", "gemini-2.5-pro")
# Stage 2: the GUI-agent model that recognizes/verifies interactions.
AGENT_MODEL = os.getenv("VIDEO2CODE_AGENT_MODEL", "gpt-5-2025-08-07")


# ============================================================
# Stage 1 — generation (video -> HTML)
# ============================================================
GEN_PROMPT_FILE = PROMPTS_DIR / "generation.txt"
GEN_OUTPUT_DIR = RESULTS_DIR / "generation" / GENERATION_MODEL
GEN_HTML_DIR = RESULTS_DIR / "htmls" / GENERATION_MODEL   # extracted .html files land here
GEN_NUM_WORKERS = int(os.getenv("VIDEO2CODE_GEN_WORKERS", "20"))
GEN_MAX_TOKENS = int(os.getenv("VIDEO2CODE_GEN_MAX_TOKENS", "128000"))
GEN_STREAM = os.getenv("VIDEO2CODE_GEN_STREAM", "0") == "1"
GEN_MAX_RETRIES = 3
GEN_INITIAL_RETRY_DELAY = 1
GEN_MAX_RETRY_DELAY = 5


# ============================================================
# Stage 2 — GUI-agent replay & verification
# ============================================================
# Directory of generated HTML files to evaluate (defaults to Stage 1's output).
AGENT_HTMLS_DIR = Path(os.getenv("VIDEO2CODE_HTMLS_DIR", str(GEN_HTML_DIR)))
AGENT_OUTPUT_BASE = RESULTS_DIR / "eval" / GENERATION_MODEL

AGENT_PORT_BASE = int(os.getenv("VIDEO2CODE_PORT_BASE", "8765"))
AGENT_PORT_POOL_SIZE = int(os.getenv("VIDEO2CODE_PORT_POOL_SIZE", "120"))
AGENT_MAX_WORKERS = int(os.getenv("VIDEO2CODE_AGENT_WORKERS", "20"))
AGENT_MAX_RETRIES = 2
AGENT_FRAME_STEP = 1
AGENT_RECORD_VIDEO = True
AGENT_MODE = "batch"

# A near-solid-color init.png usually means the page failed to render; such tasks
# can be detected and re-run with `--replay`.
SOLID_COLOR_PIXEL_RATIO = 0.97
SOLID_COLOR_QUANT_COLORS = 8


# ============================================================
# Stage 3 — score aggregation
# ============================================================
# Evaluation output directories to aggregate (each is a Stage 2 OUTPUT_BASE).
PARSE_OUTPUT_DIRS = [
    str(AGENT_OUTPUT_BASE),
]
GLOBAL_SUMMARY_OUT = RESULTS_DIR / "global_summary.json"
