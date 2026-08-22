"""
解析 agent_batch 输出目录下所有 task 的 segment/operation replay_report.json。

支持多目录：在 OUTPUT_DIRS 列表中填入所有待处理目录，
每个目录会生成独立的 summary_parsed.json，最终输出一个跨目录的全局汇总。
"""

import os
import re
import json
import glob
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

OUTPUT_DIRS = [str(d) for d in config.PARSE_OUTPUT_DIRS]
GLOBAL_SUMMARY_OUT = str(config.GLOBAL_SUMMARY_OUT)


def _functional(gv) -> str | None:
    if gv is None:
        return None
    fs = gv.get("functional_score")
    return fs if fs in ("success", "failed") else None


def _visual(gv) -> float | None:
    if gv is None:
        return None
    vs = gv.get("visual_score")
    if vs is None or vs == -1:
        return None
    return float(vs)


def _hs_visual(hs) -> float | None:
    if not isinstance(hs, dict):
        return None
    vs = hs.get("visual_score")
    if vs is None or vs == -1:
        return None
    return float(vs)


def _sort_key(x: str):
    m = re.match(r'^(\d+)', x)
    return (int(m.group(1)) if m else float('inf'), x)


def process_one_dir(output_dir: str) -> dict | None:
    """处理单个目录，返回该目录的汇总结果，写入 summary_parsed.json。"""
    summary_out_path = os.path.join(output_dir, "summary_parsed.json")
    pattern = os.path.join(output_dir, "**/replay_report.json")
    report_files = sorted(glob.glob(pattern, recursive=True))

    if not report_files:
        print(f"[WARN] 未找到任何 replay_report.json in {output_dir}，跳过")
        return None

    print(f"\n{'='*60}")
    print(f"处理目录: {output_dir}  ({len(report_files)} 个报告)")
    print(f"{'='*60}")

    tasks_raw: dict[str, list[dict]] = defaultdict(list)

    for fpath in report_files:
        parts = fpath.replace("\\", "/").split("/")
        try:
            task_dir = next(p for p in parts if p.startswith("task_"))
            seg_dir  = next(p for p in parts if p.startswith("segment_"))
            op_dir   = next(p for p in parts if p.startswith("operation_"))
        except StopIteration:
            print(f"[WARN] 无法解析路径，跳过: {fpath}")
            continue

        task_id = task_dir[len("task_"):]

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        gv      = data.get("global_verification")
        hs      = data.get("homepage_similarity")
        summary = data.get("summary", {})

        func = _functional(gv)
        if func == "success" and gv is not None and gv.get("visual_score") == -1:
            func = "failed"

        entry = {
            "segment":      seg_dir,
            "operation":    op_dir,
            "functional":   func,
            "visual":       _visual(gv),
            "hs_visual":    _hs_visual(hs),
            "is_no_action": summary.get("no_action", 0) > 0,
            "gv":           gv,
        }
        tasks_raw[task_id].append(entry)

    task_summaries: dict[str, dict] = {}
    all_task_visual:     list[float] = []
    all_task_functional: list[float] = []

    for task_id in sorted(tasks_raw, key=_sort_key):
        entries = tasks_raw[task_id]

        normal_visual_scores: list[float] = []
        for entry in entries:
            if entry["functional"] == "success" and entry["visual"] is not None:
                normal_visual_scores.append(entry["visual"])

        visual_scores: list[float] = []
        if normal_visual_scores:
            visual_scores.extend(normal_visual_scores)
            hs_vals = [e["hs_visual"] for e in entries if e["hs_visual"] is not None]
            if hs_vals:
                visual_scores.append(sum(hs_vals) / len(hs_vals))

        func_scores: list[float] = [
            1.0 if e["functional"] == "success" else 0.0
            for e in entries
        ]

        avg_vis  = round(sum(visual_scores) / len(visual_scores), 2) if visual_scores else None
        avg_func = round(sum(func_scores)   / len(func_scores),   4) if func_scores   else None
        success_n = int(sum(func_scores))
        total_n   = len(func_scores)

        task_summaries[task_id] = {
            "avg_visual_score":     avg_vis,
            "avg_functional_score": avg_func,
            "success_rate":         f"{avg_func * 100:.1f}%" if avg_func is not None else None,
            "success_count":        success_n,
            "total_operations":     total_n,
            "operations": {
                f"{e['segment']}/{e['operation']}": {
                    "functional":             e["functional"],
                    "visual":                 e["visual"],
                    "homepage_similarity_vs": e["hs_visual"],
                    "is_no_action":           e["is_no_action"],
                }
                for e in sorted(entries, key=lambda x: (x["segment"], x["operation"]))
            },
        }

        if avg_vis is not None:
            all_task_visual.append(avg_vis)
        if avg_func is not None:
            all_task_functional.append(avg_func)

        print(f"Task {task_id:>4}: visual={str(avg_vis):>6} | "
              f"functional={avg_func} ({success_n}/{total_n})")

    global_visual     = round(sum(all_task_visual)     / len(all_task_visual),     2) if all_task_visual     else None
    global_functional = round(sum(all_task_functional) / len(all_task_functional), 4) if all_task_functional else None

    dir_summary = {
        "overall": {
            "avg_visual_score":     global_visual,
            "avg_functional_score": global_functional,
            "success_rate":         f"{global_functional * 100:.1f}%" if global_functional is not None else None,
            "total_tasks":          len(task_summaries),
            "tasks_with_visual":    len(all_task_visual),
        },
        "tasks": task_summaries,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(dir_summary, f, ensure_ascii=False, indent=2)

    print(f"\n目录 visual 均值: {global_visual}")
    if global_functional is not None:
        print(f"目录 functional 均值: {global_functional}  ({global_functional * 100:.1f}% success)")
    print(f"任务数: {len(task_summaries)} | 有 visual 得分的任务数: {len(all_task_visual)}")
    print(f"结果已写入: {summary_out_path}")

    return {
        "directory":            output_dir,
        "avg_visual_score":     global_visual,
        "avg_functional_score": global_functional,
        "success_rate":         f"{global_functional * 100:.1f}%" if global_functional is not None else None,
        "total_tasks":          len(task_summaries),
        "tasks_with_visual":    len(all_task_visual),
    }


def main():
    dir_results = []

    for output_dir in OUTPUT_DIRS:
        result = process_one_dir(output_dir)
        if result is not None:
            dir_results.append(result)

    if not dir_results:
        print("\n[ERROR] 所有目录均未找到有效报告，退出。")
        return

    # 跨目录全局汇总
    all_visuals     = [r["avg_visual_score"]     for r in dir_results if r["avg_visual_score"]     is not None]
    all_functionals = [r["avg_functional_score"] for r in dir_results if r["avg_functional_score"] is not None]
    global_visual     = round(sum(all_visuals)     / len(all_visuals),     2) if all_visuals     else None
    global_functional = round(sum(all_functionals) / len(all_functionals), 4) if all_functionals else None

    global_summary = {
        "overall": {
            "avg_visual_score":     global_visual,
            "avg_functional_score": global_functional,
            "success_rate":         f"{global_functional * 100:.1f}%" if global_functional is not None else None,
            "total_directories":    len(dir_results),
        },
        "directories": dir_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(GLOBAL_SUMMARY_OUT)), exist_ok=True)
    with open(GLOBAL_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"跨目录全局汇总")
    print(f"{'='*60}")
    print(f"全局 visual score  均值: {global_visual}")
    if global_functional is not None:
        print(f"全局 functional score 均值: {global_functional}  ({global_functional * 100:.1f}% success)")
    print(f"处理目录数: {len(dir_results)}")
    print(f"全局汇总已写入: {GLOBAL_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
