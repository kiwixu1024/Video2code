import os
import sys
import json
import copy
import time
import re
import base64
import argparse
import subprocess
import requests as http_requests  # avoid conflict with openai
import openai
from openai import OpenAI

from ....main.agent_tools import helpers


def uniform_sample_frames(frames: list, target: int) -> list:
    """Uniformly pick `target` frames spread across the whole sequence, always
    keeping the first and last frame and preserving order. Returns the input
    unchanged when it already has <= target frames.

    Used to trim an over-long frame sequence without dropping its tail: a page
    that reacts slowly only reaches its finished state in the LATER frames, so a
    plain head-slice would throw away the very evidence of the action taking
    effect. Even spacing keeps both ends plus a representative middle.
    """
    n = len(frames)
    if target <= 0:
        return []
    if n <= target:
        return list(frames)
    if target == 1:
        return [frames[0]]
    # Evenly spaced indices across [0, n-1], inclusive of both ends.
    idxs = [round(i * (n - 1) / (target - 1)) for i in range(target)]
    out = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            out.append(frames[idx])
    return out


class ActionRecognizer:
    """Analyzes video frames using a large model to recognize interaction actions and map them to DOM elements."""

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _call_model(self, messages: list[dict]) -> str:
        """Unified model call interface with retry."""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=32000,
                    timeout=600,
                )
                content = response.choices[0].message.content
                if content:
                    return str(content)
                print(f"[WARN] Model returned empty content (attempt {attempt + 1}/3), retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"[WARN] Model call failed (attempt {attempt + 1}/3): {e}")
                time.sleep(2)
        return ""

    def analyze_full_sequence(
        self,
        frames: list[dict],
        domtree: str,
        page_screenshot_b64: str,
    ) -> tuple[list[list[dict]], str]:
        """
        Send all frames at once and let the model recognize the complete interaction sequence.
        Suitable for scenarios with few frames (<=10 frames).

        Returns: (action_seqs, model_raw_output)
        """
        prompt = f"""You are a webpage interaction replay assistant. I will give you a set of video frame screenshots arranged in chronological order, capturing a user interaction on a webpage.
I will also provide a DOM Tree and current screenshot of a replicated webpage.

## DOM Tree Key Fields
Each interactable element contains: id (identifier used in action instructions), tag, attrs, visible_text, address ("zone--[x,y,w,h]"), can_interact, input_value, options, etc.

## Your Task
1. Compare the frames to identify the interaction that occurred in the original webpage
2. The interaction is usually exactly one of: **click**, **enter** (typing), **select** (dropdown), or **scroll**
3. Locate the corresponding element in the DOM Tree by matching visible_text, address, tag, input_value, options, and surrounding context
4. Output the action instruction needed to replay the interaction on the replicated webpage

## Important Notes on Number of Actions
- In almost all cases, the interaction can and should be reproduced with **one single action**.
- Only output **two actions** in the special case where the video shows a dropdown/select-like interaction, but the replicated DOM Tree does not contain a usable `select` element or matching selectable option.
- In that special case, you may reproduce the interaction with two `click` actions, for example: first click to open the dropdown, then click the target option.
- Do **not** output multiple actions for ordinary clicks, typing, scrolling, or standard select elements.
- Prefer a single `select[element_id][option_text]` whenever a proper select/dropdown option exists in the DOM Tree.

## Important Notes on Missing Operations
- If the interaction shown in the video cannot be matched to any actionable element in either the current screenshot or the DOM Tree, output `\\boxed{{None}}`.
- Use `None` only when no reasonable click, enter, select, or scroll action can reproduce the observed interaction on the replicated webpage.

## Important Notes on Scroll
- If the interaction is a scroll, it is **very likely a scroll to the bottom of the page** — use `scroll[]` with no element id.
- Only use `scroll[element_id]` if the frames clearly show scrolling to a specific element rather than to the page bottom.

## Output Format
Wrap the final action instruction in LaTeX \\boxed{{}}.

Allowed formats:
- click[element_id]
- enter[element_id][input_content]
- select[element_id][option_text]
- scroll[element_id] — scroll to bring a specific element into view
- scroll[] — scroll all the way to the bottom of the page
- click[element_id_1]; click[element_id_2] — only for the special dropdown/select replacement case described above
- None — only when the interaction cannot be matched to the screenshot or DOM Tree

Briefly explain your reasoning first, then give the final answer.
The final answer must contain exactly one \\boxed{{}}. If two clicks are needed, put both click actions inside the same \\boxed{{}}. If no valid operation can be found, output exactly \\boxed{{None}}.

## DOM Tree
{domtree}

The following are the video frame sequence and replicated webpage screenshot:"""
        content = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            content.append({"type": "text", "text": f"[Frame {i + 1}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{frame['b64']}"},
            })
        content.append({"type": "text", "text": "[Replicated Webpage Current Screenshot]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{page_screenshot_b64}"},
        })

        messages = [{"role": "user", "content": content}]
        output = self._call_model(messages)

        if not output:
            return [], ""

        action_seqs = self._parse_all_boxed_actions(output)
        return action_seqs, output
    

    def verify_video_frames_sequence(
        self,
        expected_frames_b64: list[str],
        actual_frames_b64: list[str],
        sequence_desc: str = "",
    ) -> tuple[int, str, str]:
        """
        Verify whether the replicated webpage correctly executed a single operation by comparing
        frame sequences. Returns (visual_score, functional_score, raw_output).
        - visual_score: 0–100 page similarity score, or -1 if pages are incomparable
        - functional_score: "success" or "failed", whether the operation took effect
        """
        prompt = f"""You are evaluating whether a web interaction was correctly replicated by comparing **page visual similarity**.

The two groups of frames below each capture a complete interaction process including before, during, and after the action.
- Group 1 (Expected): Original webpage recording
- Group 2 (Actual): Replicated webpage after attempting the same interaction

{"The operation being performed is: " + sequence_desc if sequence_desc else ""}

## Step 1: Identify the Operation Type
Determine whether the operation is:
- **Click / Enter / Select**: An action that changes a specific UI element's state
- **Scroll**: An action whose purpose is to reveal page content

## Step 2: Visual Score (integer 0–100, or -1 if incomparable)

### For Click, Enter, or Select operations:
- Compare the **final post-action page state** between Group 1 and Group 2
- Ignore any differences in the initial page state before the action — only the result matters
- Score how visually similar the two pages look **after** the operation completes
- **Output -1 if the two post-action pages are so fundamentally different in layout or content that no meaningful visual comparison is possible** (e.g. completely different pages loaded, drastically different structure)

### For Scroll operations:
- Scrolling's sole purpose is to make content visible. Do NOT evaluate whether a scroll motion occurred.
- Compare the **total visible content** across all frames in each group
- If Group 1 required scrolling to reveal content, but Group 2 already shows all that content without scrolling, this is a **successful match** — score based on content similarity between what is visible in both
- Focus on whether the same page content is accessible/visible, not on the scrolling gesture itself

### Special Case — Pre-executed State:
If the action was already in the "done" state before it was attempted (e.g., a checkbox already checked, a tab already active, a value already set), the operation had no visible effect in Group 2. In this case:
- The functional score will be "failed" (no change occurred)
- But the **visual score should still reflect the similarity of the page states normally** — the pages can still look identical even without an observable change

Scoring criteria (for 0–100 cases):
- 90–100: Post-action pages look visually identical or nearly identical
- 70–89: Post-action pages are very similar with only minor visual differences
- 50–69: Post-action pages are similar but with noticeable differences
- 20–49: Post-action pages differ significantly
- 0–19: Post-action pages look completely different or an error state appeared
- **-1**: The two post-action pages are fundamentally incomparable (different pages, broken layout, etc.)

## Step 3: Functional Score (success / failed)
Determine whether the operation was actually executed in Group 2, and briefly explain the reason:
- `success`: The operation clearly took effect, OR the target state was already achieved before the action
- `failed`: The operation did not take effect or produced an unexpected error state

When making this judgment, explicitly mention the key visual evidence, such as:
- whether the expected UI state change appeared
- whether the target element changed as intended
- whether the page navigated, updated, expanded, selected, or scrolled as expected
- whether the action produced no visible effect or an error state


## Output Format
Briefly explain your reasoning first,ENSURE TO ADD JUSTIFICATION OF FUNCTIONAL SCORE!!!!! then provide both scores on separate lines:
- Visual score: \\score{{N}} where N is an integer from 0 to 100, or \\score{{-1}} if the pages are incomparable
- Functional score: \\passed{{success}} or \\passed{{failed}}"""

        content = [{"type": "text", "text": prompt}]


        # ── Cap total images at 49. When the actual sequence must be trimmed,
        #    uniformly sample across ALL actual frames (keeping first & last)
        #    instead of head-slicing, so a slowly-reacting page's later execution
        #    is not dropped. ────────────────────────────────────────────────
        max_images = 49
        total_images = len(expected_frames_b64) + len(actual_frames_b64)
        if total_images > max_images:
            # allowed_actual may be <= 0 (expected alone already exceeds budget);
            # uniform_sample_frames returns [] in that case.
            allowed_actual = max(max_images - len(expected_frames_b64), 0)
            actual_frames_b64 = uniform_sample_frames(actual_frames_b64, allowed_actual)
        # ────────────────────────────────────────────────────────────────
        # First group: expected state sequence
        content.append({"type": "text", "text": f"======== [Group 1: Expected State ({len(expected_frames_b64)} frames total)] ========"})
        for i, b64 in enumerate(expected_frames_b64):
            content.append({"type": "text", "text": f"Expected State - Frame {i+1}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}"}})

        # Second group: actual state sequence
        content.append({"type": "text", "text": f"======== [Group 2: Actual State ({len(actual_frames_b64)} frames total)] ========"})
        for i, b64 in enumerate(actual_frames_b64):
            content.append({"type": "text", "text": f"Actual State - Frame {i+1}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}"}})

        messages = [{"role": "user", "content": content}]

        output = self._call_model(messages)

        visual_score = self._parse_score(output)
        functional_score = self._parse_functional_score(output)
        return visual_score, functional_score, output

    def judge_homepage_similarity(
        self,
        original_frame_b64: str,
        rendered_page_b64: str,
    ) -> tuple[int, str]:
        """
        Compare the first frame of the original recording with the initial
        rendered webpage screenshot to evaluate homepage visual similarity.
        Returns (visual_score 0–100, raw_output).
        """
        prompt = """You are evaluating the visual similarity between two webpage screenshots.
- Image 1: The first frame from the original webpage recording (reference)
- Image 2: A screenshot of the replicated/rendered webpage's initial state

## Your Task
Compare these two webpage screenshots and score their visual similarity.

Focus on:
- Overall layout and structure
- Color scheme and visual style
- Content placement and hierarchy
- Key UI elements (headers, navigation, main content area, hero sections)

Scoring criteria:
- 90–100: The pages look visually identical or nearly identical
- 70–89: Very similar with only minor visual differences (slight color variation, minor spacing)
- 50–69: Similar but with noticeable differences
- 20–49: Differ significantly in layout or content
- 0–19: Look completely different

## Output Format
Briefly explain your reasoning, then provide the score on a separate line:
Visual similarity score: \\score{N} where N is an integer from 0 to 100"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "[Image 1: Original webpage first frame]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{original_frame_b64}"}},
            {"type": "text", "text": "[Image 2: Rendered webpage initial screenshot]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{rendered_page_b64}"}},
        ]

        messages = [{"role": "user", "content": content}]
        output = self._call_model(messages)
        visual_score = self._parse_score(output)
        return visual_score, output

    # ---------- Helper methods ----------

    def _format_history(self, history_actions: list[dict]) -> str:
        if not history_actions:
            return ""
        lines = []
        for i, act in enumerate(history_actions):
            action_type = act.get("action", "unknown")
            elem_id = act.get("id", "?")
            value = act.get("value") or act.get("input_val") or ""
            dom_info = act.get("dom_info", {})
            text = ""
            if dom_info and isinstance(dom_info, dict):
                text = (dom_info.get("visible_text", "") or
                        dom_info.get("attrs", {}).get("text", "") or
                        dom_info.get("attrs", {}).get("value", ""))
            line = f"  Step {i+1}: {action_type}[{elem_id}]"
            if value:
                line += f"[{value}]"
            if text:
                line += f"  (element text: {text})"
            lines.append(line)
        return "\n".join(lines)

    def _parse_score(self, output: str) -> int:
        """Parse score from \\score{N} in model output. Returns -1 to 100 (-1 = incomparable)."""
        match = re.search(r'\\score\{([^}]*)\}', output) if output else None
        try:
            raw = match.group(1).strip()
            score = int(re.search(r'-?\d+', raw).group())
            if score == -1:
                return -1
            return max(0, min(100, score))
        except (AttributeError, ValueError):
            return 0

    def _parse_functional_score(self, output: str) -> str:
        """Parse functional score ('success' or 'failed') from \\passed{} in model output."""
        if not output:
            return "failed"
        match = re.search(r'\\passed\{([^}]*)\}', output)
        if match:
            t = match.group(1).strip().lower()
            if t == "success":
                return "success"
            if t == "failed":
                return "failed"
        return "failed"

    def _parse_model_output(self, output: str) -> list[dict]:
        """Parse action instructions from model output."""
        boxed_raws = helpers.extract_boxed_actions(output)
        all_actions = []
        for bx in boxed_raws:
            if "no_action" in bx:
                continue
            parsed = helpers.parse_action_seq(bx)
            if parsed is not None:
                all_actions.extend(parsed)
        return all_actions

    def _parse_all_boxed_actions(self, output: str) -> list[list[dict]]:
        """Parse all boxed action sequences from model output."""
        boxed_raws = helpers.extract_boxed_actions(output)
        action_seqs = []
        for bx in boxed_raws:
            if "no_action" in bx:
                continue
            if bx.strip().lower() == "none":
                action_seqs.append([{"action": "none"}])
                continue
            parsed = helpers.parse_action_seq(bx)
            if parsed is not None:
                action_seqs.append(parsed)
        return action_seqs
