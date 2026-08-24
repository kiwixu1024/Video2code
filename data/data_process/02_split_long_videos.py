import os
import json
import asyncio
import logging
import copy
import math
import subprocess
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from pipeline_config import data_path

# ===================== 配置 =====================
INPUT_JSONL_PATH = data_path("raw_dataset.jsonl")
OUTPUT_JSONL_PATH = data_path("raw_dataset_splitted.jsonl")
SPLIT_VIDEO_OUTPUT_DIR = data_path("splitted_videos")

TARGET_OPS = 10       # 每段目标操作数
SPLIT_PADDING = 0.5   # 切割延后缓冲时间（秒）

# 缩减控制
TARGET_GAP = 1.0      # 特殊操作后保留的总空隙时长（秒）

# 并发设置
CONCURRENT_WORKERS = 12
FFMPEG_WORKERS = 4

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("video_processor")

# ===================== 辅助工具 =====================

def round_floats(obj):
    """递归遍历数据结构，将所有浮点数保留 3 位小数 (为了时间戳更精准)"""
    if isinstance(obj, float):
        return round(obj, 3)
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i) for i in obj]
    else:
        return obj

async def get_video_duration(video_path: str) -> float:
    """获取视频真实时长"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            return float(stdout.decode().strip())
    except Exception:
        pass
    return 0.0

# ===================== 核心视频切割逻辑 =====================
def calculate_split_strategy(segments: List[Dict]) -> List[List[Dict]]:
    """针对细粒度 Segment 的最优平均分配策略"""
    
    # 修改点：根据 if_scroll 动态计算 operation 数量权重
    seg_ops_counts = []
    for seg in segments:
        count = 0
        for op in seg.get('operations', []):
            if op.get('if_scroll', False):
                count += 4  # if_scroll 为真，算作 4 个
            else:
                count += 1  # 否则算作 1 个
        seg_ops_counts.append(count)
        
    total_ops = sum(seg_ops_counts)
    
    if total_ops <= TARGET_OPS:
        return [segments]

    num_parts = math.ceil(total_ops / TARGET_OPS)
    target_per_part = total_ops / num_parts
    
    parts = []
    current_part_segments = []
    current_ops = 0
    
    for i, seg in enumerate(segments):
        ops_in_this_seg = seg_ops_counts[i]
        
        if len(parts) == num_parts - 1:
            current_part_segments.append(seg)
            continue
            
        diff_now = abs(current_ops - target_per_part)
        diff_after = abs((current_ops + ops_in_this_seg) - target_per_part)
        
        if current_ops > 0 and diff_now < diff_after:
            parts.append(current_part_segments)
            current_part_segments = [seg]
            current_ops = ops_in_this_seg
        else:
            current_part_segments.append(seg)
            current_ops += ops_in_this_seg
            
    if current_part_segments:
        parts.append(current_part_segments)
        
    return parts

def get_part_split_time(part_segments: List[Dict]) -> float:
    """获取该部分的切割结束时间点"""
    if not part_segments:
        return 0.0
    last_seg = part_segments[-1]
    if 'end' in last_seg:
        return float(last_seg['end'])
    ops = last_seg.get('operations', [])
    if ops:
        return float(ops[-1].get('end', 0.0))
    return 0.0

async def cut_video_part(source_path: str, start_time: float, duration: Optional[float], output_path: str) -> bool:
    """执行基础 FFmpeg 切割"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True

    cmd = ["ffmpeg", "-y", "-ss", f"{start_time:.3f}"]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    
    cmd.extend([
        "-i", source_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-avoid_negative_ts", "1",
        "-loglevel", "quiet", 
        output_path
    ])

    TIMEOUT = 200 
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, 
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL 
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            try: proc.kill() 
            except: pass
            await proc.wait()
            if os.path.exists(output_path): os.remove(output_path)
            return False

        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return True
        else:
            if os.path.exists(output_path): os.remove(output_path)
            return False
    except Exception:
        return False

async def concat_videos(video1_path: str, video2_path: str, output_path: str) -> bool:
    """无损拼接两个视频"""
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        return False
        
    list_filename = f"concat_list_{os.path.basename(output_path)}.txt"
    list_path = os.path.join(os.path.dirname(output_path), list_filename)
    
    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(f"file '{os.path.abspath(video1_path).replace(chr(92), '/')}'\n")
        f.write(f"file '{os.path.abspath(video2_path).replace(chr(92), '/')}'\n")
        
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy",
        "-loglevel", "quiet", output_path
    ]
    
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), timeout=120)
        if os.path.exists(list_path): os.remove(list_path)
        return proc.returncode == 0 and os.path.exists(output_path)
    except Exception:
        if os.path.exists(list_path): os.remove(list_path)
        return False

# ===================== 空白缩减与时间戳修正逻辑 =====================

def is_special_segment(seg: Dict) -> bool:
    ops = seg.get("operations", [])
    if len(ops) != 1: return False
    op = ops[0]
    op_type = op.get("subtype", "")
    if "click" not in op_type: return False
    has_scroll = seg.get("if_scroll", False) or op.get("if_scroll", False)
    if has_scroll: return False
    return True

def get_seg_times(seg: Dict) -> Tuple[float, float]:
    ops = seg.get("operations", [])
    if not ops:
        return float(seg.get('start', 0.0)), float(seg.get('end', 0.0))
    start = float(ops[0].get('start', ops[0].get('global_time', seg.get('start', 0))))
    end = float(ops[-1].get('end', ops[-1].get('global_time', seg.get('end', start + 1))))
    return start, end

def calculate_cuts(segments: List[Dict]) -> List[Tuple[float, float]]:
    cuts = []
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        if is_special_segment(current_seg):
            _, current_end = get_seg_times(current_seg)
            next_start, _ = get_seg_times(next_seg)
            gap_duration = next_start - current_end
            
            if gap_duration > TARGET_GAP:
                cut_start = current_end + (TARGET_GAP / 2.0)
                cut_end = next_start - (TARGET_GAP / 2.0)
                cuts.append((cut_start, cut_end))
    return cuts

def adjust_time(t: float, cuts: List[Tuple[float, float]]) -> float:
    if t is None: return t
    shift = 0.0
    for c_start, c_end in cuts:
        if t >= c_end:
            shift += (c_end - c_start)
        elif c_start < t < c_end:
            shift += (t - c_start)
    return max(0.0, t - shift)

def adjust_timestamps(segments: List[Dict], time_offset: float) -> List[Dict]:
    """平移整体基础时间偏移"""
    new_segments = copy.deepcopy(segments)
    
    # 将所有的特殊 global 时间直接纳入常规平移名单
    time_keys = [
        'global_time', 'start', 'end', 
        'local_transition_time', 'local_keyframe_time',
        'global_start_time', 'global_end_time', 'global_scroll_end_time'
    ]
    
    for seg in new_segments:
        if 'start' in seg: seg['start'] = max(0.0, float(seg['start']) - time_offset)
        if 'end' in seg: seg['end'] = max(0.0, float(seg['end']) - time_offset)
        
        ops = seg.get('operations', [])
        for op in ops:
            for key in time_keys:
                if key in op and isinstance(op[key], (int, float)):
                    op[key] = max(0.0, op[key] - time_offset)
                    
    return new_segments

def adjust_item_timestamps_for_cuts(item: Dict, cuts: List[Tuple[float, float]]) -> Dict:
    """消除由于空隙被剪裁产生的时间偏差"""
    # 让 global_start_time 等直接走常规的 cuts 时间轴校准
    time_keys = [
        'start', 'end', 'global_time', 
        'local_transition_time', 'local_keyframe_time',
        'global_start_time', 'global_end_time', 'global_scroll_end_time'
    ]
    
    def walk_and_update(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k in time_keys and isinstance(v, (int, float)):
                    new_dict[k] = adjust_time(float(v), cuts)
                else:
                    new_dict[k] = walk_and_update(v)
            return new_dict
        elif isinstance(obj, list):
            return [walk_and_update(i) for i in obj]
        else:
            return obj
            
    # 因为所有特殊 global 时间戳都进入了 walk_and_update 进行统一处理，
    # 这里不需要再额外遍历还原，直接返回即可。
    return walk_and_update(item)

async def process_video_cuts(source_video: str, target_video: str, cuts: List[Tuple[float, float]], duration: float) -> bool:
    keep_intervals = []
    current_time = 0.0
    for c_start, c_end in cuts:
        if c_start > current_time:
            keep_intervals.append((current_time, c_start))
        current_time = c_end
    if current_time < duration:
        # 这里给一个极大值兜底，让 FFmpeg 切割到视频末尾
        keep_intervals.append((current_time, 999999.0))
        
    temp_dir = os.path.join(os.path.dirname(target_video), f"temp_cuts_{Path(source_video).stem}")
    os.makedirs(temp_dir, exist_ok=True)
    concat_list_path = os.path.join(temp_dir, "concat.txt")
    
    try:
        with open(concat_list_path, "w", encoding='utf-8') as f_concat:
            for idx, (k_start, k_end) in enumerate(keep_intervals):
                clip_dur = k_end - k_start
                if clip_dur <= 0.1: continue
                
                clip_name = f"clip_{idx:03d}.mp4"
                clip_path = os.path.join(temp_dir, clip_name)
                
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{k_start:.3f}"]
                if k_end < 999900.0:
                    cmd.extend(["-t", f"{clip_dur:.3f}"])
                cmd.extend([
                    "-i", source_video,
                    "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                    "-c:a", "aac", clip_path
                ])
                proc = await asyncio.create_subprocess_exec(*cmd)
                await proc.wait()
                if os.path.exists(clip_path):
                    f_concat.write(f"file '{clip_name}'\n")
                    
        merge_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", "concat.txt",
            "-c", "copy",
            os.path.abspath(target_video)
        ]
        proc_merge = await asyncio.create_subprocess_exec(*merge_cmd, cwd=temp_dir)
        await proc_merge.wait()
        return proc_merge.returncode == 0
    except Exception:
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

# ===================== 主干处理流程 =====================

async def process_single_item(item: Dict, ffmpeg_semaphore: asyncio.Semaphore) -> List[Dict]:
    segments = item.get("segments", [])
    if not segments: return [item]

    async with ffmpeg_semaphore:
        source_rel_path = item.get("video_path", "")
        base_dir = os.path.dirname(INPUT_JSONL_PATH)
        source_abs_path = os.path.abspath(source_rel_path)
        
        if not os.path.exists(source_abs_path):
            source_abs_path = os.path.join(base_dir, source_rel_path)
        
        if not os.path.exists(source_abs_path):
            return []

        real_duration = await get_video_duration(source_abs_path)
        if real_duration == 0.0: real_duration = 999999.0
        
        segment_parts = calculate_split_strategy(segments)
        video_stem = Path(source_rel_path).stem
        num_total_parts = len(segment_parts)

        # 1. 寻找并在必要时预先切割提取 init_scroll 片段
        init_scroll_seg = next((s for s in segments if s.get("video_name_segment") == "init_scroll"), None)
        init_start, init_duration = 0.0, 0.0
        init_scroll_abs_path = ""
        
        if init_scroll_seg:
            init_start = float(init_scroll_seg.get('start', 0.0))
            ops = init_scroll_seg.get('operations', [])
            if 'end' in init_scroll_seg: init_end = float(init_scroll_seg['end'])
            elif ops and 'end' in ops[-1]: init_end = float(ops[-1]['end'])
            else: init_end = init_start + 3.0
            
            init_duration = init_end - init_start
            init_scroll_abs_path = os.path.join(SPLIT_VIDEO_OUTPUT_DIR, f"{video_stem}_init_scroll.mp4")
            await cut_video_part(source_abs_path, init_start, init_duration, init_scroll_abs_path)

        tqdm.write(f"\n🎬 视频: {video_stem} | 计划切分为: {num_total_parts} 段")

        new_items = []
        current_start_time = 0.0

        for idx, part_segs in enumerate(segment_parts):
            part_num = idx + 1
            
            if idx < num_total_parts - 1:
                current_part_end_time = get_part_split_time(part_segs)
                ideal_split_time = current_part_end_time + SPLIT_PADDING
                
                next_part_start_time = 999999.0
                if segment_parts[idx + 1]:
                    first_seg = segment_parts[idx + 1][0]
                    if first_seg.get('operations'): next_part_start_time = float(first_seg['operations'][0].get('start', 999999.0))
                    elif 'start' in first_seg: next_part_start_time = float(first_seg['start'])

                split_time = max(min(ideal_split_time, next_part_start_time), current_part_end_time)
                split_time = min(split_time, real_duration)

                duration = split_time - current_start_time
                next_start_time = split_time
            else:
                duration = None
                next_start_time = None

            if current_start_time >= real_duration - 0.1 and duration is not None:
                continue
            
            # 目标输出路径
            new_video_filename = f"{video_stem}_part{part_num}.mp4"
            final_video_abs_path = os.path.join(SPLIT_VIDEO_OUTPUT_DIR, new_video_filename)
            new_video_rel_path = os.path.join("./splitted_videos", new_video_filename)

            # 临时 Base Split 路径
            base_split_video_abs = os.path.join(SPLIT_VIDEO_OUTPUT_DIR, f"base_split_{video_stem}_part{part_num}.mp4")

            has_init_scroll = any(s.get("video_name_segment") == "init_scroll" for s in part_segs)
            success = False
            base_segments = []

            # 2. 生成 Base Split 视频，并重置其对应的时间戳到 0
            if init_scroll_seg and not has_init_scroll:
                temp_cut = os.path.join(SPLIT_VIDEO_OUTPUT_DIR, f"temp_cut_{video_stem}_part{part_num}.mp4")
                if await cut_video_part(source_abs_path, current_start_time, duration, temp_cut):
                    success = await concat_videos(init_scroll_abs_path, temp_cut, base_split_video_abs)
                    if os.path.exists(temp_cut): os.remove(temp_cut)
                    
                    time_offset = current_start_time - init_duration
                    adjusted_part_segs = adjust_timestamps(part_segs, time_offset)
                    adjusted_init_seg = adjust_timestamps([init_scroll_seg], init_start)[0]
                    base_segments = [adjusted_init_seg] + adjusted_part_segs
            else:
                success = await cut_video_part(source_abs_path, current_start_time, duration, base_split_video_abs)
                base_segments = adjust_timestamps(part_segs, current_start_time)
            
            if not success:
                tqdm.write(f"   ❌ [Part {part_num}] 基础切割/拼接失败。")
                if next_start_time is not None: current_start_time = next_start_time
                continue

            # 3. 空隙缩减判定与执行
            cuts = calculate_cuts(base_segments)
            final_item = copy.deepcopy(item)
            final_item['id'] = f"{item['id']}_part{part_num}"
            final_item['video_path'] = new_video_rel_path
            final_item['segments'] = base_segments
            
            if cuts:
                tqdm.write(f"   ✂️ [Part {part_num}] 发现 {len(cuts)} 处可缩减空隙，执行压缩...")
                base_dur = await get_video_duration(base_split_video_abs)
                if base_dur == 0.0: base_dur = 999999.0
                
                reduce_success = await process_video_cuts(base_split_video_abs, final_video_abs_path, cuts, base_dur)
                if reduce_success:
                    final_item = adjust_item_timestamps_for_cuts(final_item, cuts)
                else:
                    tqdm.write(f"   ⚠️ [Part {part_num}] 缩减失败，保留原本段落。")
                    os.rename(base_split_video_abs, final_video_abs_path)
            else:
                # 不需要压缩，直接重命名并采用
                os.rename(base_split_video_abs, final_video_abs_path)
            
            # 清理无用文件
            if os.path.exists(base_split_video_abs): os.remove(base_split_video_abs)

            for i, seg in enumerate(final_item['segments']):
                seg['segment_index'] = i
            new_items.append(round_floats(final_item))

            if next_start_time is not None:
                current_start_time = next_start_time

        if init_scroll_abs_path and os.path.exists(init_scroll_abs_path):
            os.remove(init_scroll_abs_path)

        return new_items

async def process_jsonl(input_path: str, output_path: str):
    if not os.path.exists(input_path): return
    os.makedirs(SPLIT_VIDEO_OUTPUT_DIR, exist_ok=True)
    
    ffmpeg_semaphore = asyncio.Semaphore(FFMPEG_WORKERS)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    if not lines: return

    pbar = tqdm(total=len(lines), desc="Processing", unit="video")

    async def wrapped_task(line):
        try:
            data = json.loads(line)
            return await process_single_item(data, ffmpeg_semaphore)
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return []
        finally:
            pbar.update(1)

    results = await asyncio.gather(*(wrapped_task(line) for line in lines))
    pbar.close()
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item_list in results:
            for item in item_list:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
    print(f"\nProcessing complete! Generated {count} valid entries.")

if __name__ == "__main__":
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'ffprobe' not found.")
        exit(1)

    try:
        asyncio.run(process_jsonl(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH))
    except KeyboardInterrupt:
        print("\nStopped.")
