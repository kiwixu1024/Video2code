"""
Stage 2 — synthesis GUI-agent replay & verification.

扫描 htmls/ 下的所有 HTML 文件，与 frames/<ID>/ 按 ID 匹配，
对每个匹配的任务启动独立的 WebEnv 实例（独立端口），并行执行所有任务。
每个任务内部按 segment_* 子目录顺序执行，每个 segment 下含若干 operation_* 子目录。
每个 operation 只有一次执行机会（无候选重试）。
每个 segment 的全部 operation 执行完毕后调用 reset 恢复页面初始状态。
"""


import os
import sys
import json
import time
import base64
import shutil
import argparse
import concurrent.futures
import traceback
import threading

import openai
from openai import OpenAI

# Make the synthesis package root importable when run from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent_tools import helpers
from agent_tools.webenvclient import WebEnvClient
from agent_tools.actionRecongnize import ActionRecognizer

# ============================================================
# 1. 内置参数配置
# ============================================================

API_BASE   = config.API_BASE
MODEL_NAME = config.AGENT_MODEL

HTMLS_DIR      = str(config.AGENT_HTMLS_DIR)
FRAMES_DIR     = str(config.FRAMES_DIR)
PORT_BASE      = config.AGENT_PORT_BASE
PORT_POOL_SIZE = config.AGENT_PORT_POOL_SIZE
MAX_WORKERS    = config.AGENT_MAX_WORKERS
OUTPUT_BASE    = str(config.AGENT_OUTPUT_BASE)
MAX_RETRIES    = config.AGENT_MAX_RETRIES
FRAME_STEP     = config.AGENT_FRAME_STEP
MODE           = config.AGENT_MODE
RECORD_VIDEO   = config.AGENT_RECORD_VIDEO
WEBENV_PY      = config.WEBENV_PY

# 若图像中超过 SOLID_COLOR_PIXEL_RATIO 比例的像素与最多数量颜色（量化后）一致，
# 则认为该 init.png 是纯色页面。
SOLID_COLOR_PIXEL_RATIO   = config.SOLID_COLOR_PIXEL_RATIO
SOLID_COLOR_QUANT_COLORS  = config.SOLID_COLOR_QUANT_COLORS

# ============================================================
# 2. 动态端口分配池
# ============================================================

class PortPool:
    """线程安全的 WebEnv 端口池，支持动态申请与释放。"""

    def __init__(self, port_base: int, pool_size: int):
        self._all_ports = list(range(port_base, port_base + pool_size))
        self._available = list(self._all_ports)
        self._lock = threading.Lock()
        print(f"[PortPool] 初始化端口池: {self._all_ports[0]}~{self._all_ports[-1]}，共 {pool_size} 个端口")

    def acquire(self, timeout: float = 60.0) -> int | None:
        """申请一个可用端口，最多等待 timeout 秒；无可用端口时返回 None。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._available:
                    port = self._available.pop(0)
                    print(f"[PortPool] 分配端口 {port}，剩余 {len(self._available)} 个")
                    return port
            time.sleep(0.5)
        print("[PortPool] 等待超时，无可用端口")
        return None

    def release(self, port: int):
        """将端口归还到池中。"""
        with self._lock:
            if port not in self._available:
                self._available.append(port)
                print(f"[PortPool] 释放端口 {port}，当前可用 {len(self._available)} 个")

    @property
    def available_count(self) -> int:
        with self._lock:
            return len(self._available)


# ============================================================
# 3. 图片加载工具
# ============================================================

def load_frames(frames_dir: str, step: int = 1) -> list[dict]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted([
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not files:
        raise FileNotFoundError(f"在 {frames_dir} 中未找到图片文件")

    frames = []
    for i, fname in enumerate(files):
        if i % step != 0:
            continue
        fpath = os.path.join(frames_dir, fname)
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        frames.append({"path": fpath, "b64": b64, "index": i, "filename": fname})

    print(f"[INFO] 已加载 {len(frames)} 帧图片（共 {len(files)} 帧，step={step}）")
    return frames


# ============================================================
# 4. 核心重放引擎
# ============================================================

class ReplayEngine:
    def __init__(self, env: WebEnvClient, recognizer: ActionRecognizer,
                 output_dir: str, max_retries: int = 2, record_video: bool = False):
        self.env = env
        self.recognizer = recognizer
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.record_video = record_video
        self.step_records = []
        self.executed_actions = []
        self.batch_verification_result = None
        self.model_output = None
        self.request_domtree = None
        self.init_screenshot_b64 = None
        self.homepage_similarity_result = None

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    def _save_img(self, img_b64: str, filename: str) -> str:
        path = os.path.join(self.output_dir, "screenshots", filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        return path

    def _save_cdp_frames(self, frames_b64: list[str]) -> str:
        frames_dir = os.path.join(self.output_dir, "screenshots", "replay_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, b64_str in enumerate(frames_b64):
            path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_str))
        print(f"[VIDEO] 已保存 {len(frames_b64)} 帧到: {frames_dir}")
        return frames_dir

    def replay_batch(self, frames: list[dict]):
        print(f"\n{'='*60}")
        print(f" 批量分析模式，共 {len(frames)} 帧")
        print(f"{'='*60}\n")

        domtree, id2xpath = self.env.get_dom_info()
        page_screenshot = self.env.get_screenshot_b64()
        self.init_screenshot_b64 = page_screenshot
        self._save_img(page_screenshot, "init.png")

        self.request_domtree = str(domtree)
        action_seqs, model_output = self.recognizer.analyze_full_sequence(
            frames=frames,
            domtree=self.request_domtree,
            page_screenshot_b64=page_screenshot,
        )
        self.model_output = model_output

        print(f"  [INFO] 模型识别到 {len(action_seqs)} 组操作序列")

        if not action_seqs or action_seqs == []:
            print("  [INFO] 模型无输出操作，跳过当前 operation（不执行，不验证）")
            self.step_records.append({
                "seq_index": 0,
                "actions": [],
                "action_desc": "none",
                "status": "no_action",
                "model_output": model_output,
            })
            return

        # 模型输出 None 操作 → 直接跳过当前 operation，不执行不验证
        if action_seqs and all(
            len(seq) == 1 and seq[0].get("action") == "none"
            for seq in action_seqs
        ):
            print("  [INFO] 模型输出 None 操作，跳过当前 operation（不执行，不验证）")
            self.step_records.append({
                "seq_index": 0,
                "actions": [],
                "action_desc": "none",
                "status": "no_action",
                "model_output": model_output,
            })
            return

        if self.record_video:
            vid_resp = self.env.start_video_cdp()
            print(f"[VIDEO] 录制启动: {vid_resp}")

        for seq_idx, actions in enumerate(action_seqs):
            print(f"\n--- 执行操作序列 {seq_idx + 1}/{len(action_seqs)} ---")


            domtree, id2xpath = self.env.get_dom_info()
            for act in actions:
                act["dom_info"] = helpers.find_node_by_id_for_prompt(domtree, act.get("id"))

            time.sleep(2)
            exec_results = self.env.execute_action_sequence(actions, id2xpath)
            for j, res in enumerate(exec_results):
                act = res["action"]
                api_resp = res["api_response"]
                self._save_img(res["img_b64_after"], f"seq{seq_idx}_step{j}.png")
                result_flag = api_resp.get("result", False) if isinstance(api_resp, dict) else bool(api_resp)
                print(f"  Step {j+1}: {act['action']}[{act.get('id','')}] → "
                      f"{'OK' if result_flag else 'FAIL'}")

            self.executed_actions.extend(actions)
            self.step_records.append({
                "seq_index": seq_idx,
                "actions": [self._serialize_action(a) for a in actions],
                "action_desc": self._format_action_desc(actions),
                "status": "executed",
                "model_output": model_output if seq_idx == 0 else "(same batch)",
            })

        # time.sleep(2)
        actual_frames_b64 = []
        if self.record_video:
            vid_data = self.env.end_video_cdp()
            if vid_data.get("result"):
                actual_frames_b64 = vid_data.get("frames", [])
                self._save_cdp_frames(actual_frames_b64)
            else:
                print(f"[VIDEO] 录制结束异常: {vid_data.get('error', 'unknown')}")

        print(f"\n 批量重放完成，共执行 {len(self.executed_actions)} 个操作")

        if actual_frames_b64:
            print(f"\n{'='*60}")
            print(f" 开始进行全局动态序列验证...")
            print(f"{'='*60}\n")

            expected_frames_b64 = [f["b64"] for f in frames]
            if len(expected_frames_b64) > 30:
                expected_frames_b64 = expected_frames_b64[:26]
            seq_descs = [r.get("action_desc", "") for r in self.step_records if r.get("action_desc")]
            full_sequence_desc = " -> ".join(seq_descs)

            visual_score, functional_score, verify_output = self.recognizer.verify_video_frames_sequence(
                expected_frames_b64=expected_frames_b64,
                actual_frames_b64=actual_frames_b64,
                sequence_desc=full_sequence_desc,
            )

            print(f"  [VERIFY] 视觉得分: {visual_score}/100，功能状态: {functional_score}")
            self.batch_verification_result = {
                "visual_score": visual_score,
                "functional_score": functional_score,
                "is_match": visual_score >= 60 if visual_score != -1 else False,
                "reasoning_output": verify_output,
                "expected_frames_count": len(expected_frames_b64),
                "actual_frames_count": len(actual_frames_b64),
                "sequence_desc": full_sequence_desc,
            }
        else:
            print("\n[WARN] 录屏未开启或未捕获到帧，跳过全局动态序列验证。")

        time.sleep(2)
        self.env.scroll_to_top()
        print("  [INFO] 已滚动回页面顶部")

    def _format_action_desc(self, actions: list[dict]) -> str:
        parts = []
        for a in actions:
            desc = f"{a['action']}[{a.get('id', '')}]"
            val = a.get('value') or a.get('input_val') or ''
            if val:
                desc += f"[{val}]"
            parts.append(desc)
        return ", ".join(parts)

    def _serialize_action(self, action: dict) -> dict:
        a = dict(action)
        if "dom_info" in a and a["dom_info"] is not None:
            di = a["dom_info"]
            if isinstance(di, dict):
                a["dom_info"] = {
                    "id": di.get("id"),
                    "tag": di.get("tag"),
                    "visible_text": di.get("visible_text"),
                    "address": di.get("address"),
                    "can_interact": di.get("can_interact"),
                }
            else:
                a["dom_info"] = str(di)
        return a

    def save_report(self, extra_meta: dict = None) -> str:
        total_steps = len(self.step_records)
        success_steps = sum(1 for r in self.step_records if r.get("status") == "success")
        no_action_steps = sum(1 for r in self.step_records if r.get("status") == "no_action")
        failed_steps = sum(1 for r in self.step_records if r.get("status") == "failed")
        executed_steps = sum(1 for r in self.step_records if r.get("status") == "executed")

        report = {
            "summary": {
                "total_steps": total_steps,
                "success": success_steps,
                "no_action": no_action_steps,
                "failed": failed_steps,
                "executed_batch": executed_steps,
                "total_actions_executed": len(self.executed_actions),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "model_output": self.model_output,
            "request_domtree": self.request_domtree,
            "steps": self.step_records,
            "executed_actions": [self._serialize_action(a) for a in self.executed_actions],
            "global_verification": self.batch_verification_result,
            "homepage_similarity": self.homepage_similarity_result,
        }
        if extra_meta:
            report["meta"] = extra_meta

        report_path = os.path.join(self.output_dir, "replay_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] 报告已保存: {report_path}")
        print(f"  总步骤: {total_steps} | 成功: {success_steps} | "
              f"无操作: {no_action_steps} | 失败: {failed_steps} | "
              f"批量执行: {executed_steps}")
        return report_path


# ============================================================
# 5. 目录发现工具
# ============================================================

def discover_matched_tasks(htmls_dir: str, frames_dir: str) -> list[dict]:
    """
    扫描 htmls_dir 下所有 .html 文件，提取 ID（文件名去掉扩展名），
    与 frames_dir 下同名子目录匹配，返回匹配列表。
    """
    if not os.path.isdir(htmls_dir):
        raise FileNotFoundError(f"HTML 目录不存在: {htmls_dir}")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames 根目录不存在: {frames_dir}")

    html_ids = {
        os.path.splitext(f)[0]: os.path.join(htmls_dir, f)
        for f in os.listdir(htmls_dir)
        if f.lower().endswith(".html")
    }

    matched = []
    import re as _re
    def _task_sort_key(item):
        m = _re.match(r'^(\d+)', item[0])
        return (int(m.group(1)) if m else float('inf'), item[0])
    for task_id, html_path in sorted(html_ids.items(), key=_task_sort_key):
        frames_base = os.path.join(frames_dir, task_id)
        if os.path.isdir(frames_base):
            matched.append({
                "task_id": task_id,
                "html_path": html_path,
                "frames_base": frames_base,
            })
        else:
            print(f"[WARN] ID={task_id}: frames 目录不存在 ({frames_base})，跳过")

    return matched


def discover_segments_and_operations(frames_base: str) -> list[dict]:
    """
    返回 frames_base 下所有 segment_* 子目录（按数字顺序），
    每个 segment 包含其下按数字顺序排列的 operation_* 子目录列表。
    返回格式：
      [
        {
          "segment_name": "segment_0",
          "segment_dir":  "/path/to/segment_0",
          "operations": [
            {"op_name": "operation_0", "op_dir": "/path/to/segment_0/operation_0"},
            ...
          ]
        },
        ...
      ]
    """
    def _numeric_key(name: str, prefix: str) -> int:
        suffix = name[len(prefix):]
        return int(suffix) if suffix.isdigit() else float("inf")

    seg_names = sorted(
        [d for d in os.listdir(frames_base)
         if d.startswith("segment_") and os.path.isdir(os.path.join(frames_base, d))],
        key=lambda d: _numeric_key(d, "segment_"),
    )

    segments = []
    for seg_name in seg_names:
        seg_dir = os.path.join(frames_base, seg_name)
        op_names = sorted(
            [d for d in os.listdir(seg_dir)
             if d.startswith("operation_") and os.path.isdir(os.path.join(seg_dir, d))],
            key=lambda d: _numeric_key(d, "operation_"),
        )
        operations = [
            {"op_name": op_name, "op_dir": os.path.join(seg_dir, op_name)}
            for op_name in op_names
        ]
        segments.append({
            "segment_name": seg_name,
            "segment_dir": seg_dir,
            "operations": operations,
        })
    return segments


# ============================================================
# 6. --replay 模式：纯色 init.png 检测与任务清理
# ============================================================

def _is_solid_color_image(img_path: str,
                           ratio_threshold: float = SOLID_COLOR_PIXEL_RATIO,
                           quant_colors: int = SOLID_COLOR_QUANT_COLORS) -> bool:
    """
    判断一张图片是否为"近纯色"页面。

    策略：将图片量化到 quant_colors 种颜色，若出现次数最多的颜色占全图像素比例
    超过 ratio_threshold，则认为是纯色页面。

    依赖 Pillow（PIL）。若未安装则回退为 False（不误删）。
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        # Pillow 或 numpy 未安装时保守返回 False
        print("[WARN] Pillow/numpy 未安装，跳过纯色检测，请执行: pip install Pillow numpy")
        return False

    try:
        img = Image.open(img_path).convert("RGB")
        # 量化降低颜色数，使相近颜色合并
        quantized = img.quantize(colors=quant_colors, method=Image.Quantize.MEDIANCUT)
        arr = np.array(quantized)
        total_pixels = arr.size
        if total_pixels == 0:
            return False
        # 找出出现最多的颜色索引及其像素数
        counts = np.bincount(arr.flatten())
        dominant_ratio = counts.max() / total_pixels
        return dominant_ratio >= ratio_threshold
    except Exception as e:
        print(f"[WARN] 检测 {img_path} 时出错: {e}")
        return False


def find_solid_color_tasks(output_base: str,
                           ratio_threshold: float = SOLID_COLOR_PIXEL_RATIO,
                           quant_colors: int = SOLID_COLOR_QUANT_COLORS) -> list[str]:
    """
    扫描 output_base/task_* 目录，找出所有 segment/operation 的 init.png **全部**
    均为纯色的 task，返回这些 task 的 task_id 列表。

    判定规则：
      - 一个 task 目录下可能有多个 segment_*/operation_* 子目录。
      - 每个 operation 目录下有 screenshots/init.png。
      - 只有当该 task 下**所有**找到的 init.png 都是纯色时，才标记该 task。
      - 若某个 task 完全没有找到任何 init.png，则跳过（不标记）。
    """
    if not os.path.isdir(output_base):
        print(f"[WARN] 输出目录不存在: {output_base}")
        return []

    solid_task_ids: list[str] = []

    task_dirs = sorted([
        d for d in os.listdir(output_base)
        if d.startswith("task_") and os.path.isdir(os.path.join(output_base, d))
    ])

    for task_dir_name in task_dirs:
        task_id = task_dir_name[5:]  # 去掉 "task_" 前缀
        task_path = os.path.join(output_base, task_dir_name)

        # 收集该 task 下所有 segment/operation 的 init.png
        init_pngs: list[str] = []
        for seg_entry in sorted(os.listdir(task_path)):
            if not seg_entry.startswith("segment_"):
                continue
            seg_path = os.path.join(task_path, seg_entry)
            if not os.path.isdir(seg_path):
                continue
            for op_entry in sorted(os.listdir(seg_path)):
                if not op_entry.startswith("operation_"):
                    continue
                op_path = os.path.join(seg_path, op_entry)
                if not os.path.isdir(op_path):
                    continue
                candidate = os.path.join(op_path, "screenshots", "init.png")
                if os.path.isfile(candidate):
                    init_pngs.append(candidate)

        if not init_pngs:
            # 没有任何 init.png，可能是空任务或结构异常，跳过
            continue

        # 所有 init.png 都是纯色才标记
        all_solid = all(
            _is_solid_color_image(p, ratio_threshold, quant_colors)
            for p in init_pngs
        )

        if all_solid:
            print(f"[REPLAY] Task {task_id}: 全部 {len(init_pngs)} 个 init.png 均为纯色，标记重跑")
            solid_task_ids.append(task_id)
        else:
            solid_count = sum(
                1 for p in init_pngs
                if _is_solid_color_image(p, ratio_threshold, quant_colors)
            )
            if solid_count > 0:
                print(f"[REPLAY] Task {task_id}: {solid_count}/{len(init_pngs)} 个 init.png 为纯色，但非全部，保留")

    return solid_task_ids


def delete_task_output(output_base: str, task_id: str) -> bool:
    """删除 output_base/task_{task_id} 目录，返回是否成功。"""
    task_path = os.path.join(output_base, f"task_{task_id}")
    if not os.path.isdir(task_path):
        print(f"[WARN] 目录不存在，跳过删除: {task_path}")
        return False
    try:
        shutil.rmtree(task_path)
        print(f"[REPLAY] 已删除目录: {task_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 删除 {task_path} 失败: {e}")
        return False


def run_replay_mode(htmls_dir: str, frames_dir: str, output_base: str,
                    port_base: int, pool_size: int, max_workers: int,
                    frame_step: int, max_retries: int, record_video: bool,
                    ratio_threshold: float = SOLID_COLOR_PIXEL_RATIO,
                    quant_colors: int = SOLID_COLOR_QUANT_COLORS):
    """
    --replay 模式主流程：
      1. 扫描已完成任务，找出所有 init.png 均为纯色的 task。
      2. 打印列表并等待确认（可通过 --yes 跳过）。
      3. 删除这些 task 的输出目录。
      4. 重新用正常流程跑这批 task。
    """
    print(f"\n{'='*60}")
    print(f"  REPLAY 模式 — 纯色 init.png 任务重跑")
    print(f"  输出目录: {output_base}")
    print(f"  纯色判定阈值: {ratio_threshold*100:.1f}%，量化色数: {quant_colors}")
    print(f"{'='*60}\n")

    solid_ids = find_solid_color_tasks(output_base, ratio_threshold, quant_colors)

    if not solid_ids:
        print("[REPLAY] 未发现需要重跑的纯色任务，退出。")
        return

    print(f"\n[REPLAY] 共发现 {len(solid_ids)} 个需要重跑的任务: {solid_ids}")
    print("[REPLAY] 即将删除上述任务目录并重新执行...\n")

    # 删除旧输出
    deleted = [tid for tid in solid_ids if delete_task_output(output_base, tid)]
    if not deleted:
        print("[REPLAY] 没有成功删除任何目录，退出。")
        return

    print(f"\n[REPLAY] 已删除 {len(deleted)} 个旧任务目录，准备重新执行...\n")

    # 构建需要重跑的任务列表（从原始发现逻辑中过滤）
    try:
        all_tasks = discover_matched_tasks(htmls_dir, frames_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return

    replay_tasks = [t for t in all_tasks if t["task_id"] in set(deleted)]

    if not replay_tasks:
        print("[REPLAY] 在 HTML/frames 中未找到对应任务，无法重跑。")
        return

    print(f"[REPLAY] 将重新执行 {len(replay_tasks)} 个任务: "
          f"{[t['task_id'] for t in replay_tasks]}")

    # 复用正常执行逻辑
    _execute_tasks(
        tasks=replay_tasks,
        output_base=output_base,
        port_base=port_base,
        pool_size=pool_size,
        max_workers=max_workers,
        frame_step=frame_step,
        max_retries=max_retries,
        record_video=record_video,
        tag="[REPLAY]",
    )


# ============================================================
# 7. 单任务执行函数（供并行调用）
# ============================================================

def run_task(task: dict, port_pool: PortPool, output_base: str,
             frame_step: int, max_retries: int, record_video: bool) -> dict:
    """从端口池动态申请端口，在独立 WebEnv 实例上执行一个任务的所有 segment/operation。
    每个 operation 只有一次执行机会（无候选重试）。
    每个 segment 的全部 operation 执行完毕后调用 reset 恢复页面初始状态。
    """
    task_id     = task["task_id"]
    html_path   = task["html_path"]
    frames_base = task["frames_base"]
    prefix      = f"[Task {task_id}]"

    port = port_pool.acquire()
    if port is None:
        return {"task_id": task_id, "status": "error", "error": "端口池耗尽，无可用端口"}

    print(f"\n{'#'*60}")
    print(f"# {prefix} 开始 | port={port} | html={html_path}")
    print(f"{'#'*60}")

    segments = discover_segments_and_operations(frames_base)
    if not segments:
        print(f"{prefix} [WARN] 未找到任何 segment_* 子目录，跳过")
        port_pool.release(port)
        return {"task_id": task_id, "status": "skipped", "reason": "no segments"}

    print(f"{prefix} 发现 {len(segments)} 个 segment: "
          f"{[s['segment_name'] for s in segments]}")

    client = OpenAI(api_key=config.require_api_key(), base_url=API_BASE)
    env    = WebEnvClient(port=port, webenv_py=WEBENV_PY)

    try:
        env.start(html_path, is_url=False)
        recognizer = ActionRecognizer(client, MODEL_NAME)

        for seg_idx, segment in enumerate(segments):
            seg_name   = segment["segment_name"]
            operations = segment["operations"]
            print(f"\n{prefix} ── Segment {seg_idx + 1}/{len(segments)}: {seg_name} "
                  f"({len(operations)} operation(s))")

            for op_idx, operation in enumerate(operations):
                op_name = operation["op_name"]
                op_dir  = operation["op_dir"]
                print(f"\n{prefix}   Operation {op_idx + 1}/{len(operations)}: {op_name}")

                try:
                    frames = load_frames(op_dir, step=frame_step)
                except FileNotFoundError as e:
                    print(f"{prefix} [WARN] 跳过 {seg_name}/{op_name}: {e}")
                    continue

                if len(frames) < 2:
                    print(f"{prefix} [WARN] 跳过 {seg_name}/{op_name}: 帧数不足 ({len(frames)} 帧)")
                    continue

                output_dir = os.path.join(output_base, f"task_{task_id}", seg_name, op_name)
                engine = ReplayEngine(
                    env=env,
                    recognizer=recognizer,
                    output_dir=output_dir,
                    max_retries=max_retries,
                    record_video=record_video,
                )

                # 首页相似度比较：仅对第一个 segment 的第一个 operation 执行
                if seg_idx == 0 and op_idx == 0:
                    try:
                        engine.replay_batch(frames)
                    except Exception as e:
                        print(f"{prefix} [ERROR] {seg_name}/{op_name} 执行异常: {type(e).__name__}: {e}")
                        traceback.print_exc()

                    first_action_is_scroll = False
                    if engine.step_records:
                        first_actions = engine.step_records[0].get("actions", [])
                        if first_actions and first_actions[0].get("action") == "scroll":
                            first_action_is_scroll = True

                    if not first_action_is_scroll and frames and engine.init_screenshot_b64:
                        print(f"\n{prefix} 首个 operation 非 scroll，执行首页相似度比较...")
                        try:
                            hp_score, hp_output = recognizer.judge_homepage_similarity(
                                original_frame_b64=frames[0]["b64"],
                                rendered_page_b64=engine.init_screenshot_b64,
                            )
                            engine.homepage_similarity_result = {
                                "visual_score": hp_score,
                                "reasoning_output": hp_output,
                            }
                            print(f"{prefix} [HOMEPAGE] 首页视觉相似度: {hp_score}/100")
                        except Exception as e:
                            print(f"{prefix} [WARN] 首页相似度比较失败: {e}")
                else:
                    try:
                        engine.replay_batch(frames)
                    except Exception as e:
                        print(f"{prefix} [ERROR] {seg_name}/{op_name} 执行异常: {type(e).__name__}: {e}")
                        traceback.print_exc()

                try:
                    engine.save_report(extra_meta={
                        "task_id":      task_id,
                        "segment":      seg_name,
                        "operation":    op_name,
                        "html_path":    html_path,
                        "frames_dir":   op_dir,
                        "total_frames": len(frames),
                        "frame_step":   frame_step,
                        "port":         port,
                        "mode":         MODE,
                        "record_video": record_video,
                    })
                except Exception as e:
                    print(f"{prefix} [WARN] 保存报告失败: {e}")

            # 每个 segment 结束后 reset 页面回初始状态
            print(f"\n{prefix} {seg_name} 全部 operation 执行完毕，reset 页面...")
            try:
                reset_result = env.reset()
                print(f"{prefix} reset 结果: {reset_result}")
            except Exception as e:
                print(f"{prefix} [WARN] reset 失败: {e}")
            time.sleep(2)

        print(f"\n{prefix} 所有 segment 执行完毕")
        return {"task_id": task_id, "status": "done", "port": port}

    except Exception as e:
        print(f"{prefix} [ERROR] 致命错误: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"task_id": task_id, "status": "error", "error": str(e)}

    finally:
        env.shutdown()
        port_pool.release(port)


# ============================================================
# 8. 共用任务执行核心（正常模式 & replay 模式复用）
# ============================================================

def _execute_tasks(tasks: list[dict], output_base: str,
                   port_base: int, pool_size: int, max_workers: int,
                   frame_step: int, max_retries: int, record_video: bool,
                   tag: str = "[INFO]"):
    """并行执行给定任务列表，写入 summary.json。"""
    os.makedirs(output_base, exist_ok=True)

    print(f"{tag} 共执行 {len(tasks)} 个任务: {[t['task_id'] for t in tasks]}")
    print(f"{tag} 并行度: {max_workers}，动态端口池: "
          f"{port_base}~{port_base + pool_size - 1}（{pool_size} 个）")

    port_pool = PortPool(port_base=port_base, pool_size=pool_size)

    task_args = [
        (task, port_pool, output_base, frame_step, max_retries, record_video)
        for task in tasks
    ]

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for args in task_args:
                task_id = args[0]["task_id"]
                future = executor.submit(run_task, *args)
                futures[future] = task_id
                print(f"{tag} 已提交 Task {task_id}，缓冲等待 3 秒...")
                time.sleep(3)

            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"\n[DONE] Task {task_id} 完成: {result.get('status')}")
                except Exception as e:
                    print(f"\n[ERROR] Task {task_id} 发生未捕获异常: {e}")
                    results.append({"task_id": task_id, "status": "error", "error": str(e)})

    except KeyboardInterrupt:
        print(f"\n{tag} 用户手动中断")

    # 更新 summary.json（追加本批结果）
    summary_path = os.path.join(output_base, "summary.json")
    existing = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    prev_results = existing.get("results", [])
    merged_results = prev_results + results

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_tasks_ever": len(merged_results),
            "results": merged_results,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    done  = sum(1 for r in results if r.get("status") == "done")
    error = sum(1 for r in results if r.get("status") == "error")
    skip  = sum(1 for r in results if r.get("status") == "skipped")
    print(f"\n{tag} 本批执行完成 — 成功: {done} | 错误: {error} | 跳过: {skip}")
    print(f"{tag} 汇总报告已更新至: {summary_path}")


# ============================================================
# 9. 主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Replay GUI Agent — Parallel Batch Runner"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="扫描已完成任务，找出所有 init.png 均为纯色的 task，删除并重跑",
    )
    parser.add_argument(
        "--solid-ratio",
        type=float,
        default=SOLID_COLOR_PIXEL_RATIO,
        metavar="RATIO",
        help=f"纯色判定阈值（0~1），默认 {SOLID_COLOR_PIXEL_RATIO}",
    )
    parser.add_argument(
        "--solid-colors",
        type=int,
        default=SOLID_COLOR_QUANT_COLORS,
        metavar="N",
        help=f"量化色数，默认 {SOLID_COLOR_QUANT_COLORS}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    htmls_dir    = HTMLS_DIR
    frames_dir   = FRAMES_DIR
    port_base    = PORT_BASE
    pool_size    = PORT_POOL_SIZE
    output_base  = OUTPUT_BASE
    max_workers  = MAX_WORKERS
    frame_step   = FRAME_STEP
    max_retries  = MAX_RETRIES
    record_video = RECORD_VIDEO

    # ── replay 模式 ──────────────────────────────────────────
    if args.replay:
        run_replay_mode(
            htmls_dir=htmls_dir,
            frames_dir=frames_dir,
            output_base=output_base,
            port_base=port_base,
            pool_size=pool_size,
            max_workers=max_workers,
            frame_step=frame_step,
            max_retries=max_retries,
            record_video=record_video,
            ratio_threshold=args.solid_ratio,
            quant_colors=args.solid_colors,
        )
        return

    # ── 正常模式 ─────────────────────────────────────────────
    try:
        all_tasks = discover_matched_tasks(htmls_dir, frames_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if not all_tasks:
        print("[ERROR] 未找到任何匹配的任务", file=sys.stderr)
        sys.exit(1)

    # 断点续传：过滤已存在的任务
    os.makedirs(output_base, exist_ok=True)
    completed_tasks = {
        entry[5:]
        for entry in os.listdir(output_base)
        if entry.startswith("task_")
    }

    tasks = [t for t in all_tasks if t["task_id"] not in completed_tasks]
    skipped_count = len(all_tasks) - len(tasks)

    if skipped_count > 0:
        print(f"[INFO] 已跳过 {skipped_count} 个已完成的任务。")

    if not tasks:
        print("[INFO] 所有任务已完成，无需执行。")
        return

    _execute_tasks(
        tasks=tasks,
        output_base=output_base,
        port_base=port_base,
        pool_size=pool_size,
        max_workers=max_workers,
        frame_step=frame_step,
        max_retries=max_retries,
        record_video=record_video,
        tag="[INFO]",
    )

    # 补充 skipped_count 到 summary（仅正常模式）
    summary_path = os.path.join(output_base, "summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["skipped_by_checkpoint"] = skipped_count
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    main()
