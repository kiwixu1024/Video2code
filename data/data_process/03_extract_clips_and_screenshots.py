import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm  # pip install tqdm
from pipeline_config import data_path

# ===================== 配置 =====================

# 输入输出路径
INPUT_JSONL_PATH = data_path("raw_dataset_splitted.jsonl")
OUTPUT_JSONL_PATH = data_path("raw_dataset_splitted_clipped.jsonl")

# 剪辑视频片段输出根目录
CLIP_OUTPUT_PREFIX = data_path("clipped_videos")
# 截图输出根目录
SCREENSHOT_OUTPUT_PREFIX = data_path("operation_screenshots")

# --- 功能开关 ---
ENABLE_VIDEO_CLIPPING = True   # 是否执行视频切片 (Operation 级别)
ENABLE_SCREENSHOTS = True      # 是否执行关键帧截图

# --- 并发控制 ---
CONCURRENT_WORKERS = 16   # 同时处理的视频文件数 (控制总体并发)
CLIP_FFMPEG_WORKERS = 4   # 剪辑并发
SHOT_FFMPEG_WORKERS = 4   # 截图并发


CLIP_INSTANT_BEFORE_OFFSET = 0.5
CLIP_INSTANT_AFTER_OFFSET = 1.2

CLIP_DURATION_AFTER_OFFSET = 0.3

# 日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"), 
        logging.StreamHandler()                
    ]
)
logger = logging.getLogger("video_processor")

# ===================== 核心剪辑逻辑 (精确版) =====================
async def clip_local_video_segment(source_path: str, start: float, end: float, output_dir: str, seg_idx: int, op_idx: int) -> Optional[str]:
    """
    剪辑单个 Operation 片段
    """
    stem = Path(source_path).stem
    out_filename = f"{stem}_{start:.2f}_{end:.2f}.mp4"
    out_path = os.path.join(output_dir, out_filename)
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    duration = end - start
    if duration <= 0:
        return None

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),         
        "-t", str(duration),       
        "-i", source_path,
        "-c:v", "libx264",         
        "-preset", "veryfast",     
        "-crf", "23",              
        "-c:a", "aac",             
        "-b:a", "128k",
        "-avoid_negative_ts", "1", 
        "-loglevel", "error",      
        out_path
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        
        if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        else:
            err_msg = stderr.decode().strip() if stderr else "No stderr"
            if os.path.exists(out_path): os.remove(out_path)
            if "Invalid argument" not in err_msg:
                logger.warning(f"Clip failed {out_filename}: {err_msg}")
            return None

    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        logger.error(f"Error clipping {out_filename}: {e}")
        return None

async def extract_local_video_segments(video_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments: return []
    if not os.path.exists(video_path): return []
        
    video_stem = Path(video_path).stem
    job_dir = os.path.join(CLIP_OUTPUT_PREFIX, video_stem)
    os.makedirs(job_dir, exist_ok=True)
    
    semaphore = asyncio.Semaphore(CLIP_FFMPEG_WORKERS)
    
    async def worker(seg_idx: int, op_idx: int, op_data: Dict[str, Any]):
        async with semaphore:
            start = op_data.get("start")
            end = op_data.get("end")
            
            if start is None or end is None:
                return None
            
            abs_path = await clip_local_video_segment(video_path, float(start), float(end), job_dir, seg_idx, op_idx)
            
            if abs_path:
                parent_dir_name = os.path.basename(os.path.dirname(abs_path))
                file_name = os.path.basename(abs_path)
                rel_path = f"./clipped_videos/{parent_dir_name}/{file_name}"
                return {"seg_idx": seg_idx, "op_idx": op_idx, "path": rel_path}
            return None
    
    tasks = []
    for i, seg in enumerate(segments):
        ops = seg.get("operations", [])
        for j, op in enumerate(ops):
            tasks.append(worker(i, j, op))

    if not tasks: return []
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# ===================== 核心截图逻辑 (融合动态多帧逻辑) =====================
async def capture_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    timestamp = max(0.0, timestamp)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path, "-frames:v", "1", "-q:v", "2", "-loglevel", "error", output_path]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        
        file_valid = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if proc.returncode == 0 and file_valid:
            return True
        else:
            if os.path.exists(output_path) and not file_valid:
                os.remove(output_path)
            return False
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

async def process_video_operations(video_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量执行截图任务 (动态抽帧版)"""
    if not segments: return []
    if not os.path.exists(video_path): return []
    
    video_stem = Path(video_path).stem
    video_output_dir = os.path.join(SCREENSHOT_OUTPUT_PREFIX, video_stem)
    os.makedirs(video_output_dir, exist_ok=True)
    
    semaphore = asyncio.Semaphore(SHOT_FFMPEG_WORKERS)

    async def _capture_and_record(v_path: str, t_sec: float, p_abs: str, p_rel: str, f_type: str):
        async with semaphore:
            success = await capture_frame(v_path, t_sec, p_abs)
        if success:
            return {
                "time_seconds": t_sec,
                "frame_type": f_type,
                "image_path": p_rel
            }
        return None

    async def process_single_op(seg_idx: int, op_idx: int, op_data: Dict[str, Any]):
        # 兼容不同字段名
        start_time = op_data.get("start")
        action_time = op_data.get("global_start_time")
        end_time = op_data.get("end")
        
        # 新增解析字段
        global_end_time = op_data.get("global_end_time") 
        action_finish_time = op_data.get("action_finish_time")
        subtype = op_data.get("subtype")
        if_scroll = op_data.get("if_scroll", False)
        
        # 新增提取 action_detail 中的 action 类型
        action_detail = op_data.get("action_detail", {})
        action_type = action_detail.get("action") if action_detail else None

        if start_time is None: return None
        # 防御性赋值，防止 action_time 为空导致后续报错
        if action_time is None: action_time = start_time 
            
        timestamps_to_extract = []
        
        # --- 动态构建抽帧时间点 ---
        if subtype == "input_duration":
            
            # ================= [新增补丁逻辑] =================
            if action_type == "select":
                if if_scroll:
                    # 如果有scroll，从头 (action_time - offset) 一直按每秒2帧抽到 end_time 结束
                    current_time = max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET)
                    limit_time = end_time if end_time is not None else action_time
                    while round(current_time, 2) <= round(limit_time, 2):
                        timestamps_to_extract.append((current_time, "during_select_scroll"))
                        current_time += 0.5
                else:
                    # 如果没有scroll，在 global_start_time 和 global_end_time 之间按每秒2帧抽帧
                    current_time = action_time
                    limit_time = global_end_time if global_end_time is not None else end_time
                    if limit_time is None: limit_time = action_time
                    
                    # 依然保留一个before帧以记录初始状态
                    timestamps_to_extract.append((max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET), "before_select"))
                    
                    while round(current_time, 2) <= round(limit_time, 2):
                        timestamps_to_extract.append((current_time, "during_select"))
                        current_time += 0.5
            # =================================================
            
            else:
                # [原有逻辑]: 普通的 input_duration
                if if_scroll:
                    timestamps_to_extract.append((max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET), "before_input_duration"))
                    if end_time is not None and action_finish_time is not None:
                        current_time = action_finish_time + CLIP_DURATION_AFTER_OFFSET
                        while round(current_time, 2) <= round(end_time, 2):
                            timestamps_to_extract.append((current_time, "after_input_scroll_duration"))
                            current_time += 0.5
                else:
                    timestamps_to_extract.append((max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET), "before_input"))
                    if end_time is not None:
                        timestamps_to_extract.append((end_time, "after_input"))
                    
        elif subtype == "click":
            if if_scroll:
                timestamps_to_extract.append((max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET), "before_click_duration"))
                if end_time is not None:
                    current_time = action_time + CLIP_DURATION_AFTER_OFFSET
                    while round(current_time, 2) <= round(end_time, 2):
                        timestamps_to_extract.append((current_time, "after_click_scroll_duration"))
                        current_time += 0.5
            else:
                timestamps_to_extract.append((max(0.0, action_time - CLIP_INSTANT_BEFORE_OFFSET), "before_instant_click"))
                timestamps_to_extract.append((action_time + CLIP_INSTANT_AFTER_OFFSET, "after_instant_click"))
                
        elif subtype == "init_scroll":
            # 默认保底逻辑: 按中间抽帧处理
            current_time = start_time
            timestamps_to_extract.append((current_time, "before_init_scroll"))
            current_time += 0.5
            if end_time is not None:
                while round(current_time, 2) <= round(end_time, 2):
                    timestamps_to_extract.append((current_time, "during_init_scroll"))
                    current_time += 0.5
            else:
                timestamps_to_extract.append((start_time, "ERROR"))
        else:
            raise RuntimeError(f"Operation Type Not Allowed: {subtype}")
        
        # --- 并发执行该 Operation 内的所有截图任务 ---
        capture_tasks = []
        for t_sec, frame_type in timestamps_to_extract:
            frame_name = f"{video_stem}_seg{seg_idx}_op{op_idx}_t{t_sec:.2f}_{frame_type}.jpg"
            path_abs = os.path.join(video_output_dir, frame_name)
            path_rel = f"./operation_screenshots/{video_stem}/{frame_name}"
            
            capture_tasks.append(_capture_and_record(video_path, t_sec, path_abs, path_rel, frame_type))
            
        if not capture_tasks: return None
        
        results = await asyncio.gather(*capture_tasks)
        extracted_frames_info = [r for r in results if r is not None]

        if extracted_frames_info:
            return {
                "segment_index": seg_idx,
                "operation_index": op_idx,
                "extracted_frames": extracted_frames_info,
                "description": op_data.get("description", "")
            }
        return None

    tasks = []
    for i, seg in enumerate(segments):
        operations = seg.get("operations", [])
        for j, op in enumerate(operations):
            tasks.append(process_single_op(i, j, op))
    
    if not tasks: return []
    ops_results = await asyncio.gather(*tasks)
    return [r for r in ops_results if r is not None]

# ===================== JSONL 处理逻辑 =====================
async def process_jsonl_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        logger.error(f"JSONL file not found: {input_path}")
        return
    
    logger.info("Reading input file...")
    raw_lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    total_files = len(raw_lines)
    logger.info(f"Loaded {total_files} items. Starting processing...")

    video_semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)

    async def video_task(item: Dict) -> Optional[Dict]:
        async with video_semaphore:
            base_dir = os.path.dirname(input_path)
            rel_video_path = item.get("video_path")
            
            item['clip_video_list'] = []
            item['operation_screenshots'] = []
            
            if not rel_video_path: return None
            
            video_path = os.path.join(base_dir, rel_video_path)
            segments = item.get("segments", [])

            if not segments: return item

            parallel_tasks = []
            task_types = []

            if ENABLE_VIDEO_CLIPPING:
                parallel_tasks.append(extract_local_video_segments(video_path, segments))
                task_types.append("clip")

            if ENABLE_SCREENSHOTS:
                parallel_tasks.append(process_video_operations(video_path, segments))
                task_types.append("shot")

            if not parallel_tasks: return item

            results = await asyncio.gather(*parallel_tasks)

            for type_name, res in zip(task_types, results):
                if type_name == "clip":
                    clip_paths_flat = []
                    clip_map = {} 
                    for r in res:
                        clip_map[(r['seg_idx'], r['op_idx'])] = r['path']
                        clip_paths_flat.append(r['path'])
                    
                    for i, seg in enumerate(item['segments']):
                        ops = seg.get('operations', [])
                        for j, op in enumerate(ops):
                            if (i, j) in clip_map:
                                op['clip_path'] = clip_map[(i, j)]
                    
                    item['clip_video_list'] = clip_paths_flat

                elif type_name == "shot":
                    item['operation_screenshots'] = res

            return item

    tasks = [video_task(item) for item in raw_lines]
    valid_count = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=total_files, desc="Processing Videos", unit="file")
        
        for future in asyncio.as_completed(tasks):
            try:
                res = await future
                if res:
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                    f_out.flush() 
                    valid_count += 1
            except Exception as e:
                logger.error(f"Task failed with error: {e}")
            finally:
                pbar.update(1)
        
        pbar.close()

    logger.info("=" * 40)
    logger.info(f"处理任务结束!")
    logger.info(f"成功写出数据: {valid_count} 条")
    logger.info(f"输出文件路径: {output_path}")
    logger.info("=" * 40)

# ===================== 入口 =====================
if __name__ == "__main__":
    if ENABLE_VIDEO_CLIPPING:
        os.makedirs(CLIP_OUTPUT_PREFIX, exist_ok=True)
    if ENABLE_SCREENSHOTS:
        os.makedirs(SCREENSHOT_OUTPUT_PREFIX, exist_ok=True)
    
    try:
        asyncio.run(process_jsonl_file(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
