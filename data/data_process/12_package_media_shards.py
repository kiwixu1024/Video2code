import os
import json
import tarfile
import shutil
from pipeline_config import data_path

# ====== 全局变量配置 ======
VIDEO_BASE_DIR = data_path()
CLIP_VIDEO_BASE_DIR = data_path()

JSONL_PATH = data_path("data.jsonl")
OUTPUT_DIR = data_path("unistore", "raw")

# 每个分片的行数上限
LINE_LIMIT = 1000

def process_sharded_data(jsonl_path, output_dir, limit=1000):
    """
    读取 JSONL，每 limit 行生成一个新的 tar 包和对应的 jsonl 分片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    
    current_tar = None
    current_jsonl_f = None
    part_idx = 0
    line_in_part = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # --- 检查是否需要开启新的分片 ---
                if line_in_part % limit == 0:
                    # 关闭旧文件
                    if current_tar:
                        current_tar.close()
                    if current_jsonl_f:
                        current_jsonl_f.close()
                    
                    # 开启新文件
                    part_idx += 1
                    line_in_part = 0
                    
                    new_jsonl_path = os.path.join(output_dir, f"{base_name}_part{part_idx}.jsonl")
                    new_tar_path = os.path.join(output_dir, f"{base_name}_part{part_idx}.tar")
                    
                    print(f"\n[NEW PART] 正在创建分片 {part_idx}...")
                    current_jsonl_f = open(new_jsonl_path, "w", encoding="utf-8")
                    current_tar = tarfile.open(new_tar_path, "w")

                # --- 处理当前行数据 ---
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 第 {line_num} 行 JSON 解析失败: {e}")
                    continue

                # 1. 写入新的分片 JSONL
                current_jsonl_f.write(line + "\n")
                line_in_part += 1

                # 2. 打包主视频
                video_rel_path = data.get("video_path")
                if video_rel_path:
                    video_abs_path = os.path.join(VIDEO_BASE_DIR, video_rel_path)
                    if os.path.exists(video_abs_path):
                        arcname = os.path.basename(video_abs_path)
                        current_tar.add(video_abs_path, arcname=arcname)
                    else:
                        print(f"[WARN] 主视频不存在: {video_abs_path}")

                # 3. 打包 clip 视频列表
                clip_list = data.get("clip_video_list", [])
                if isinstance(clip_list, list):
                    for clip_rel_path in clip_list:
                        clip_abs_path = os.path.join(CLIP_VIDEO_BASE_DIR, clip_rel_path)
                        if os.path.exists(clip_abs_path):
                            arcname = os.path.basename(clip_abs_path)
                            current_tar.add(clip_abs_path, arcname=arcname)
                        else:
                            print(f"[WARN] clip 视频不存在: {clip_abs_path}")
                
                if line_num % 100 == 0:
                    print(f"[PROGRESS] 已处理 {line_num} 行...")

    finally:
        # 确保最后的文件被正常关闭
        if current_tar:
            current_tar.close()
        if current_jsonl_f:
            current_jsonl_f.close()

    print(f"\n[OK] 所有分片处理完成，共生成 {part_idx} 个分片。")
    print(f"[OK] 输出目录: {output_dir}")

if __name__ == "__main__":
    process_sharded_data(JSONL_PATH, OUTPUT_DIR, limit=LINE_LIMIT)
