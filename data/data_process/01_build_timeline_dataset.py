import os
import json
import random
import glob
from pipeline_config import data_path

# ===================== 路径配置 (请根据实际情况修改) =====================
# 输入 JSON 文件夹路径
JSON_DIR = data_path("results")
# 输入 Video 文件夹路径
VIDEO_DIR = data_path("videos")
# 输出 JSONL 文件路径
OUTPUT_JSONL = data_path("raw_dataset.jsonl")

# ===================== 参数配置 =====================
RANDOM_TIME_MIN = 1.0  # 随机前置/后置时间的最小值
RANDOM_TIME_MAX = 1.5  # 随机前置/后置时间的最大值
TIME_UNIT = 0.1        # 时间归一化单位

# ===================== 工具函数 =====================

def normalize_time(t, unit=TIME_UNIT):
    """将时间归一化到指定单位（如0.1秒）"""
    if t is None:
        return 0.0
    return round(round(t / unit) * unit, 2)

def process_single_json(json_path, video_dir):
    """读取单个JSON并处理成目标格式"""
    file_name = os.path.basename(json_path)
    file_id = os.path.splitext(file_name)[0]  # 去除后缀作为 ID
    
    # 1. 检查视频文件是否存在
    video_filename = f"{file_id}.mp4"
    abs_video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(abs_video_path):
        print(f"[Warning] Video not found for ID: {file_id}, skipping...")
        return None

    target_video_path = f"./videos/{video_filename}"

    # 2. 读取 JSON 内容
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[Error] Failed to decode JSON: {json_path}")
            return None

    # 3. 处理 Timeline
    raw_segments = data.get("action_frames_timeline_filtered", [])
    processed_segments = []

    for segment in raw_segments:
        raw_ops = segment.get("operations", [])
        processed_ops = []
        
        # --- 修改点：适配不同操作类型的逻辑 ---
        for op in raw_ops:
            subtype = op.get("subtype")
            if_scroll = op.get("if_scroll", False)
            global_start_time = op.get("global_start_time")
            
            # 如果没有 global_start_time，无法定位起始点，跳过
            if global_start_time is None:
                continue

            # 预先生成随机偏移量 (后续根据情况决定是否采用)
            current_before = random.uniform(RANDOM_TIME_MIN, RANDOM_TIME_MAX)
            current_after = random.uniform(RANDOM_TIME_MIN, RANDOM_TIME_MAX)

            op_start_time = 0.0
            op_end_time = 0.0
            op_action_finish_time = 0.0

            # 逻辑分支 1 & 2: input_duration
            if subtype == "input_duration":
                # 修改点：让 start 也往前 offset 一点，并确保不为负数
                op_start_time = max(0.0, global_start_time - current_before)
                
                if not if_scroll:
                    # input_duration 且 if_scroll 为 false
                    op_end_time = op.get("global_end_time")
                else:
                    # input_duration 且 if_scroll 为 true
                    op_end_time = op.get("global_scroll_end_time")
                    op_action_finish_time = op.get("global_end_time")
                
                # 因为没有使用后置偏移来计算 end，重置 current_after 为 0 保持数据清晰
                current_after = 0.0
                
            # 逻辑分支 3 & 4: click
            elif subtype == "click":
                op_start_time = max(0.0, global_start_time - current_before)
                if not if_scroll:
                    # click 且 if_scroll 为 false
                    op_end_time = global_start_time + current_after
                else:
                    # click 且 if_scroll 为 true
                    op_end_time = op.get("global_scroll_end_time")
                    # 后置时间未被用来计算 end，重置 offset_after 为 0
                    current_after = 0.0 
            elif subtype == "init_scroll":
                # 修改点：让 start 也往前 offset 一点，并确保不为负数
                op_start_time = max(0.0, global_start_time)
                
                if not if_scroll:
                    # input_duration 且 if_scroll 为 false
                    op_end_time = op.get("global_end_time")
                else:
                    # input_duration 且 if_scroll 为 true
                    op_end_time = op.get("global_scroll_end_time")
                    op_action_finish_time = op.get("global_end_time")
                
                # 因为没有使用后置偏移来计算 end，重置 current_after 为 0 保持数据清晰
                current_after = 0.0

            # 容错处理：如果算出的 end_time 为空（比如缺少对应的键），则跳过
            if op_end_time is None:
                continue
            
            # 复制原有 op 数据，避免修改原始数据
            new_op = op.copy()
            
            # 注入剪辑时间戳到 operation 内部

            new_op.update({
                "start": normalize_time(op_start_time),
                "action_finish_time": normalize_time(op_action_finish_time) if op_action_finish_time else None,
                "end": normalize_time(op_end_time),
                "clip_offset": {
                    "before": round(current_before, 2),
                    "after": round(current_after, 2)
                }
            })
            
            processed_ops.append(new_op)

        if not processed_ops:
            continue

        segment_item = {
            "segment_index": segment.get("segment_index"),
            "video_name_segment": segment.get("video_name"), 
            "start": segment.get("segment_start_time"),
            "end": segment.get("segment_end_time"),
            "operations": processed_ops
        }
        processed_segments.append(segment_item)

    if not processed_segments:
        return None

    return {
        "id": file_id,
        "video_path": target_video_path,
        "segments": processed_segments
    }

    
# ===================== 主程序 =====================
def main():
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 json 文件
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files in {JSON_DIR}")

    count = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for i, json_file in enumerate(json_files):
            result = process_single_json(json_file, VIDEO_DIR)
            
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(json_files)} files...")

    print(f"Done! Successfully generated {count} records in {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
