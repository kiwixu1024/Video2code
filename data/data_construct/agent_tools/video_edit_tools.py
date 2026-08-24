import os
import json
import base64
import subprocess
import shutil
import tempfile
import cv2
import numpy as np
import argparse

# ==========================================
# 1. Configuration & Constants
# ==========================================

# --- HSV Thresholds (OpenCV Range: H=0-180, S=0-255, V=0-255) ---

# 1. 红色 (点击/关键帧 - Point Event)
LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])

# 2. 橙色 (浏览状态 - 延长上一个操作的结束时间) [NEW]
# #fbbc05 色相(H)在OpenCV中约等于 20
LOWER_ORANGE = np.array([11, 70, 50])
UPPER_ORANGE = np.array([34, 255, 255])

# 3. 绿色 (持续输入 - Duration Event)
# H: 35-85 覆盖黄绿到深绿
LOWER_GREEN = np.array([35, 70, 50])
UPPER_GREEN = np.array([85, 255, 255])

# 4. 紫色 (Blue->Red 过渡色，也视为红色点击的一部分)
LOWER_PURPLE = np.array([125, 50, 50]) 
UPPER_PURPLE = np.array([160, 255, 255])

# 5. 蓝色 (静默/空闲状态)
LOWER_BLUE = np.array([100, 70, 50])
UPPER_BLUE = np.array([125, 255, 255])

# --- Detection Config ---
DETECTION_BAR_HEIGHT = 12   # 底部检测条高度
RED_RATIO_THRESH = 0.6      # 红色判定阈值
ORANGE_RATIO_THRESH = 0.6   # 橙色判定阈值 [NEW]
GREEN_RATIO_THRESH = 0.6    # 绿色判定阈值
BLUE_RATIO_THRESH = 0.6     # 蓝色判定阈值

# --- Timing Configuration ---
DELAY_SECONDS = 1.0         # 红色点击后的 "关键帧" 延迟时间

# --- Output Config ---
FINAL_CROP_PIXELS = 12      # 最终输出视频裁剪掉底部多少像素 (0表示不裁)

# ==========================================
# 2. HSV Color Classification Logic
# ==========================================

def classify_bar_color_hsv(frame):
    """
    使用 HSV 空间识别底部条颜色。
    Returns: status_string ("red", "orange", "green", "blue", "other")
    """
    h_img, w_img, _ = frame.shape
    
    if h_img < DETECTION_BAR_HEIGHT:
        return "other", 0, 0
    
    strip = frame[h_img - DETECTION_BAR_HEIGHT : h_img, :, :]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    
    # 1. 生成掩码
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_r1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_r2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_p  = cv2.inRange(hsv, LOWER_PURPLE, UPPER_PURPLE)
    mask_o  = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE) # 橙色掩码
    mask_g  = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)   # 绿色掩码
    
    # 2. 合并 红色+紫色
    mask_red_event = cv2.bitwise_or(mask_r1, mask_r2)
    mask_red_event = cv2.bitwise_or(mask_red_event, mask_p)
    
    # 3. 计算占比
    total_pixels = strip.shape[0] * strip.shape[1]
    if total_pixels == 0: return "other", 0, 0

    ratio_blue  = np.count_nonzero(mask_blue) / total_pixels
    ratio_red   = np.count_nonzero(mask_red_event) / total_pixels
    ratio_orange= np.count_nonzero(mask_o) / total_pixels 
    ratio_green = np.count_nonzero(mask_g) / total_pixels 
    
    # 4. 判定优先级: 红 > 绿 > 橙 > 蓝
    if ratio_red >= RED_RATIO_THRESH:
        return "red", ratio_red, ratio_blue
    elif ratio_green >= GREEN_RATIO_THRESH:
        return "green", ratio_green, ratio_blue
    elif ratio_orange >= ORANGE_RATIO_THRESH:
        return "orange", ratio_orange, ratio_blue
    elif ratio_blue >= BLUE_RATIO_THRESH:
        return "blue", ratio_red, ratio_blue
    else:
        return "other", ratio_red, ratio_blue

def detect_transitions_in_video(video_path, only_green=False):
    """
    扫描视频并检测视觉事件。
    如果 only_green=True，则仅检测绿色状态，并忽略其他一切颜色逻辑。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for detection: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    detected_events = []
    last_status = "unknown"
    
    # 状态机
    green_active = False
    green_start_time = 0.0
    
    orange_active = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame_idx / fps
        
        status, _, _ = classify_bar_color_hsv(frame)

        # --- A. 如果处于仅检测绿色模式 ---
        if only_green:
            if status == "green":
                if not green_active:
                    green_active = True
                    green_start_time = current_time
            else:
                if green_active:
                    green_active = False
                    green_end_time = current_time
                    # 过滤 < 0.1s 的噪点
                    if green_end_time - green_start_time > 0.1:
                        detected_events.append({
                            "type": "duration",
                            "start": green_start_time,
                            "end": green_end_time,
                            "scroll_end": green_end_time,
                            "if_scroll": True
                        })
            continue # 跳过下方所有的红/橙逻辑

        # --- B. 正常逻辑 (Rising Edge 红色) ---
        if status == "red" and last_status != "red":
            # 红色打断绿色
            if green_active:
                detected_events.append({
                    "type": "duration",
                    "start": green_start_time,
                    "end": current_time
                })
                green_active = False
            
            # 给红条初始也分配一个 end time，后续橙色条出现时会修改它
            detected_events.append({"type": "point", "time": current_time, "end": current_time})
        
        # --- C. 正常逻辑 (State Machine 绿色) ---
        if status == "green":
            if not green_active:
                green_active = True
                green_start_time = current_time
        else:
            if green_active:
                green_active = False
                green_end_time = current_time
                # 过滤 < 0.1s 的噪点
                if green_end_time - green_start_time > 0.1:
                    detected_events.append({
                        "type": "duration",
                        "start": green_start_time,
                        "end": green_end_time
                    })

        # --- D. 正常逻辑 (State Machine 橙色) ---
        if status == "orange":
            if not orange_active:
                orange_active = True
        else:
            if orange_active:
                orange_active = False
                if len(detected_events) > 0:
                    detected_events[-1]["scroll_end"] = current_time
                    detected_events[-1]["if_scroll"] = True

        last_status = status

    # 收尾
    if green_active:
        detected_events.append({
            "type": "duration",
            "start": green_start_time,
            "end": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        })
    elif not only_green and orange_active and len(detected_events) > 0:
        # 如果视频结束时橙条还在，就把上一事件延长到视频结束
        detected_events[-1]["scroll_end"] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        detected_events[-1]["if_scroll"] = True


    cap.release()
    return detected_events

# ==========================================
# 3. Helpers
# ==========================================

def _get_video_duration(path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path
        ]
        out = subprocess.check_output(cmd)
        data = json.loads(out)
        return float(data["format"]["duration"])
    except:
        return 0.0
    

def _make_video_op_label(vid_data, op_labels, detected_events, segment_start, segment_end):
    segment_ops = []
    for op_idx, label in enumerate(op_labels):
        if op_idx < len(detected_events):
            event = detected_events[op_idx]
            if_scroll = False
            if event.get("if_scroll", False) or label == "Init Scroll Operation":
                if_scroll = True
            if label == "Init Scroll Operation":
                action_detail = None
            else:
                action_detail = vid_data.get("actions_detail")[op_idx]
            if event["type"] == "point":
                trans_time = event["time"]
                end_t = event.get("end", trans_time)
                keyframe_time = trans_time + DELAY_SECONDS
                raw_op = {
                    "type": label, "subtype": "click", "is_duration": if_scroll, "if_scroll":if_scroll, 
                    "global_start_time": round(segment_start + keyframe_time, 3),
                    "global_end_time": round(segment_start + end_t, 3),
                    "description": f"{label}",
                    "action_detail": action_detail
                }

                if if_scroll:
                    raw_op["global_scroll_end_time"] = round(event["scroll_end"] + segment_start, 3)

                segment_ops.append(raw_op)
            
            elif event["type"] == "duration":
                start_t, end_t = event["start"], event["end"]
                raw_op = {
                    "type": label, "subtype": "input_duration", 
                    "is_duration": True, "is_green_duration": True, "if_scroll":if_scroll, 
                    "global_start_time": round(segment_start + start_t, 3),
                    "global_end_time": round(segment_start + end_t, 3),
                    "duration_seconds": round(end_t - start_t, 3),
                    "description": f"{label}",
                    "action_detail": action_detail
                }
                if if_scroll:
                    raw_op["global_scroll_end_time"] = round(event["scroll_end"] + segment_start, 3)

                if label == "Init Scroll Operation":
                    raw_op["subtype"] = "init_scroll"
                
                segment_ops.append(raw_op)

    return segment_ops

# ==========================================
# 4. Merge Logic: Strict Version
# ==========================================

def merge_video(resp_or_payload):
    if isinstance(resp_or_payload, dict):
        data = resp_or_payload
    else:
        resp = resp_or_payload
        resp.raise_for_status()
        data = resp.json()

    videos = data.get("videos", [])
    if not videos:
        raise RuntimeError("empty video list")

    videos = [v for v in videos if v.get("pass_flag", True)]
    if not videos:
        raise RuntimeError("empty video list after pass_flag filter")

    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH")

    tmpdir = tempfile.mkdtemp(prefix="redbar_merge_strict_")
    
    valid_segments_paths = []
    current_global_time = 0.0
    detailed_timeline = []

    try:
        for i, vid_data in enumerate(videos):
            b64 = vid_data.get("video_b64", "")
            if not b64: continue
            
            ct = vid_data.get("content_type", "video/mp4")
            ext = ".mp4" if "mp4" in ct else ".webm"
            raw_path = os.path.join(tmpdir, f"raw_{i}{ext}")
            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(b64))

            old_m = vid_data.get("old_moments") or []
            new_m = vid_data.get("new_moments") or []
            if isinstance(old_m, str): old_m = json.loads(old_m)
            if isinstance(new_m, str): new_m = json.loads(new_m)
            
            # --- [NEW] 判断是否为初始滚动视频 ---
            is_init = vid_data.get("is_init_scroll", False) or vid_data.get("name") == "init_scroll"
            
            op_labels = []
            if is_init:
                # 为初始滚动视频占位一个操作 label，确保后续 timeline 记录不越界
                op_labels.append("Init Scroll Operation")
            else:
                for _ in old_m: op_labels.append("Old Operation")
                for _ in new_m: op_labels.append("New Operation")
                
            total_expected = len(op_labels)

            print(f"--- Processing Video {i} ({vid_data.get('name')}) ---")

            # --- [NEW] 针对性检测与异常中断 ---
            if is_init:
                if not vid_data.get("pass_flag", False):
                    raise RuntimeError("Init scroll video Failed! Aborting all processes.")
                detected_events = detect_transitions_in_video(raw_path, only_green=True)
                if not detected_events:
                    # 如果没检测到绿色，直接阻断，停止后续一切流程
                    raise RuntimeError("Init scroll video missing green bar! Aborting all processes.")
                
                # 若检测到绿色，取最初的一帧 start 和最后一帧 end 当作整个滚动操作的时间
                # (这样做可以防止哪怕录制过程中有一帧颜色闪烁导致的段落断裂)
                overall_start = detected_events[0]["start"]
                overall_end = detected_events[-1]["end"]
                detected_events = [{
                    "type": "duration", 
                    "start": overall_start, 
                    "end": overall_end, 
                    "scroll_end": overall_end,  # 补上这个键
                    "if_scroll": True           # 保持数据结构一致性
                }]
                num_detected = 1
            else:
                detected_events = detect_transitions_in_video(raw_path)
                num_detected = len(detected_events)

            if num_detected != total_expected:
                print(f"[REJECT] Video {i}: Detected {num_detected} != Expected {total_expected}")
                continue
            
            print(f"[ACCEPT] Video {i} accepted.")

            norm_path = os.path.join(tmpdir, f"norm_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", raw_path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                norm_path
            ], check=True)

            dur = _get_video_duration(norm_path)
            segment_start = current_global_time
            segment_end = current_global_time + dur

            segment_ops = _make_video_op_label(vid_data, op_labels, detected_events, segment_start, segment_end)

            detailed_timeline.append({
                "segment_index": i,
                "segment_start_time": round(segment_start, 3),
                "segment_end_time": round(segment_end, 3),
                "video_name": vid_data.get("name"),
                "operations": segment_ops,
            })

            current_global_time = segment_end
            valid_segments_paths.append(norm_path)

        return _finalize_video_output(tmpdir, valid_segments_paths, detailed_timeline)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise e

# ==========================================
# 5. Merge Logic: Longest/Dedup Version
# ==========================================

def merge_video_longest(resp_or_payload):
    if isinstance(resp_or_payload, dict):
        data = resp_or_payload
    else:
        resp = resp_or_payload
        resp.raise_for_status()
        data = resp.json()

    videos = data.get("videos", [])
    if not videos: raise RuntimeError("empty video list")

    videos = [v for v in videos if v.get("pass_flag", True)]
    if not videos: raise RuntimeError("empty video list after filter")

    indices_to_drop = set()
    n_cand = len(videos)
    for i in range(n_cand):
        name_i = videos[i].get("name", "")
        for j in range(n_cand):
            if i == j: continue
            name_j = videos[j].get("name", "")
            if name_j.startswith(name_i) and len(name_j) > len(name_i):
                indices_to_drop.add(i)
                break
    videos = [v for idx, v in enumerate(videos) if idx not in indices_to_drop]
    print(f"[DEDUP] Keeping {len(videos)} unique videos.")

    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None: raise RuntimeError(f"{tool} not found")

    tmpdir = tempfile.mkdtemp(prefix="redbar_merge_longest_")
    
    valid_segments_paths = []
    current_global_time = 0.0
    detailed_timeline = []

    try:
        for i, vid_data in enumerate(videos):
            b64 = vid_data.get("video_b64", "")
            if not b64: continue
            
            ct = vid_data.get("content_type", "video/mp4")
            ext = ".mp4" if "mp4" in ct else ".webm"
            raw_path = os.path.join(tmpdir, f"raw_{i}{ext}")
            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(b64))

            old_m = vid_data.get("old_moments") or []
            new_m = vid_data.get("new_moments") or []
            if isinstance(old_m, str): old_m = json.loads(old_m)
            if isinstance(new_m, str): new_m = json.loads(new_m)
            
            # --- [NEW] 判断是否为初始滚动视频 ---
            is_init = vid_data.get("is_init_scroll", False) or vid_data.get("name") == "init_scroll"
            
            op_labels = []
            if is_init:
                # 为初始滚动视频占位一个操作 label，确保后续 timeline 记录不越界
                op_labels.append("Init Scroll Operation")
            else:
                for _ in old_m: op_labels.append("Old Operation")
                for _ in new_m: op_labels.append("New Operation")
                
            total_expected = len(op_labels)

            print(f"--- Processing Video {i} ({vid_data.get('name')}) ---")

            # --- [NEW] 针对性检测与异常中断 ---
            if is_init:
                detected_events = detect_transitions_in_video(raw_path, only_green=True)
                if not detected_events:
                    # 如果没检测到绿色，直接阻断，停止后续一切流程
                    raise RuntimeError("Init scroll video missing green bar! Aborting all processes.")
                
                # 若检测到绿色，取最初的一帧 start 和最后一帧 end 当作整个滚动操作的时间
                # (这样做可以防止哪怕录制过程中有一帧颜色闪烁导致的段落断裂)
                overall_start = detected_events[0]["start"]
                overall_end = detected_events[-1]["end"]
                detected_events = [{
                    "type": "duration", 
                    "start": overall_start, 
                    "end": overall_end, 
                    "scroll_end": overall_end,  # 补上这个键
                    "if_scroll": True           # 保持数据结构一致性
                }]
                num_detected = 1
            else:
                detected_events = detect_transitions_in_video(raw_path)
                num_detected = len(detected_events)

            if num_detected != total_expected:
                print(f"[REJECT] Video {i}: Detected {num_detected} != Expected {total_expected}")
                continue
            
            norm_path = os.path.join(tmpdir, f"norm_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", raw_path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                norm_path
            ], check=True)

            dur = _get_video_duration(norm_path)
            segment_start = current_global_time
            segment_end = current_global_time + dur

            segment_ops = _make_video_op_label(vid_data, op_labels, detected_events, segment_start, segment_end)

            detailed_timeline.append({
                "segment_index": i,
                "segment_start_time": round(segment_start, 3),
                "segment_end_time": round(segment_end, 3),
                "video_name": vid_data.get("name"),
                "operations": segment_ops,
            })

            current_global_time = segment_end
            valid_segments_paths.append(norm_path)

        return _finalize_video_output(tmpdir, valid_segments_paths, detailed_timeline, True)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise e

# ==========================================
# 6. Shared Output Helper
# ==========================================

def _finalize_video_output(tmpdir, valid_segments_paths, detailed_timeline, cut = False):
    if not valid_segments_paths:
        raise RuntimeError("No valid segments remained after validation.")

    concat_list_path = os.path.join(tmpdir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in valid_segments_paths:
            f.write(f"file '{p}'\n")

    concat_mp4 = os.path.join(tmpdir, "concat_raw.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        concat_mp4
    ], check=True)

    final_mp4 = os.path.join(tmpdir, "final_merged.mp4")
    ffmpeg_cmd = ["ffmpeg", "-y", "-v", "error", "-i", concat_mp4]
    
    if cut and FINAL_CROP_PIXELS > 0:
        ffmpeg_cmd += ["-vf", f"crop=in_w:in_h-{FINAL_CROP_PIXELS}:0:0"]

    ffmpeg_cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-c:a", "copy",
        final_mp4
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    with open(final_mp4, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("ascii")
    
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return video_b64, "video/mp4", detailed_timeline


def test_merge_video(resp_or_payload):
    # 1. 解析 Payload
    if isinstance(resp_or_payload, dict):
        data = resp_or_payload
    else:
        resp = resp_or_payload
        resp.raise_for_status()
        data = resp.json()

    videos = data.get("videos", [])
    if not videos:
        raise RuntimeError("empty video list")

    # 检查依赖
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH")

    tmpdir = tempfile.mkdtemp(prefix="simple_merge_")
    valid_segments_paths = []
    current_global_time = 0.0
    detailed_timeline = []

    try:
        # 2. 无脑循环所有视频
        for i, vid_data in enumerate(videos):
            b64 = vid_data.get("video_b64", "")
            if not b64: 
                continue
            
            # 解码写入临时文件
            ct = vid_data.get("content_type", "video/mp4")
            ext = ".mp4" if "mp4" in ct else ".webm"
            raw_path = os.path.join(tmpdir, f"raw_{i}{ext}")
            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(b64))

            print(f"--- Stitching Video {i} ({vid_data.get('name', 'unnamed')}) ---")

            # 3. 强制归一化 (保留这一步极其重要，否则不同分辨率/编码的视频直接拼会报错或花屏)
            norm_path = os.path.join(tmpdir, f"norm_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", raw_path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                norm_path
            ], check=True)

            # 4. 记录基础时间线信息（为了适配你原有的 _finalize_video_output）
            dur = _get_video_duration(norm_path)
            segment_start = current_global_time
            segment_end = current_global_time + dur

            detailed_timeline.append({
                "segment_index": i,
                "segment_start_time": round(segment_start, 3),
                "segment_end_time": round(segment_end, 3),
                "video_name": vid_data.get("name", f"video_{i}"),
                "operations": [], # 去掉复杂的操作记录，直接给空列表
            })

            current_global_time = segment_end
            valid_segments_paths.append(norm_path)

        # 5. 直接交给最后的拼接函数处理
        return _finalize_video_output(tmpdir, valid_segments_paths, detailed_timeline)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise e
    


# ==========================================
# 测试脚本 (改进版：精细打印 + 切换点高亮)
# ==========================================

def test_video_status_detailed(video_path, only_green=False, print_interval=0.5):
    """
    测试函数：精细打印视频状态，并高亮标识颜色状态的切换点。
    
    :param video_path: 视频路径
    :param only_green: 是否只检测绿色 (对应 init_scroll 逻辑)
    :param print_interval: 常规状态打印的时间间隔（秒），默认 0.5 秒
    """
    print(f"========== 开始测试视频: {video_path} ==========")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: 
        fps = 30.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}s")
    print(f"打印设置: 每 {print_interval} 秒打印一次常规状态，并实时抓取切换点。")
    print("-" * 50)

    last_status = None
    last_print_time = -print_interval  # 确保第 0 秒能打印出来

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame_idx / fps

        # 获取当前帧的颜色状态
        status, primary_ratio, blue_ratio = classify_bar_color_hsv(frame)

        # --------------------------------------------------
        # 核心逻辑 1：精准捕捉并高亮状态切换点 (逐帧检测)
        # --------------------------------------------------
        if last_status is not None and status != last_status:
            print(f"\n{'='*12} 🔄 [状态切换] 🔄 {'='*12}")
            print(f" ⏰ 时间点: {current_time:.3f}s (帧: {int(current_frame_idx)})")
            print(f" 🚥 变  化: [{last_status.upper()}]  --->  [{status.upper()}]")
            print(f" 📊 占  比: 判定色={primary_ratio:.2f}, 蓝色={blue_ratio:.2f}")
            print(f"{'='*42}\n")

        # --------------------------------------------------
        # 核心逻辑 2：精细化的常规心跳打印
        # --------------------------------------------------
        if current_time - last_print_time >= print_interval:
            print(f"  [时间 {current_time:06.2f}s | 帧 {int(current_frame_idx):04d}] 状态: {status:<6} | 判定色占比: {primary_ratio:.2f} | 蓝色占比: {blue_ratio:.2f}")
            last_print_time = current_time

        last_status = status

    cap.release()

    # 运行实际的事件检测逻辑，验证提取结果是否与上方日志吻合
    print("-" * 50)
    print("🚀 运行提取逻辑 (detect_transitions_in_video):")
    detected_events = detect_transitions_in_video(video_path, only_green=only_green)
    
    print(f"\n=> 🎯 最终检测出 {len(detected_events)} 个事件 (Events)")
    for i, event in enumerate(detected_events):
        print(f"   Event {i+1}: {event}")
        
    print("================ 测试结束 ================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect timeline marker colors in a video")
    parser.add_argument("video", help="Path to the video to inspect")
    parser.add_argument("--only-green", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.video):
        # print_interval=0.2 表示每 0.2 秒打印一次常规状态，你可以根据需要调小或调大
        test_video_status_detailed(args.video, only_green=args.only_green, print_interval=0.5)
    else:
        parser.error(f"video does not exist: {args.video}")
