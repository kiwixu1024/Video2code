import os
import uuid
import json
import pickle
from copy import deepcopy
import hashlib
import traceback
import math
import cv2
import re
import random
from tqdm import tqdm
from wids import wids  # pip install wids
from faker import Faker    # pip install faker
from pipeline_config import data_path

# ===== 全局路径配置 =====
ORIG_DIR = data_path("unistore", "raw")
DEST_DIR = data_path("unistore", "final")
VIDEO_ROOT_DIR = data_path()

# 全局统计剔除数量
TOTAL_FILTERED = 0

# 初始化 Faker
fake = Faker()

# ===== 1. URL 生成工具函数 (仅作兜底，优先用数据里的) =====
def generate_mixed_video_url():
    """随机生成长短不一、风格迥异的真实感视频 URL"""
    short_domains = ['v.io', 'v.cc', 's.tv', 'm.net', 'cdn.li']
    long_domains = ['media-cache-cluster.internal.net', 'secure-delivery.prod.cloud', 'content-gateway.service.io']
    storage_domains = ['storage.googleapis.com', 's3.aws.com', 'oss-cn-beijing.aliyuncs.com', 'video-cdn.net']
    video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']
    style_roll = random.random()

    if style_roll < 0.4:
        domain = random.choice(storage_domains)
        ext = random.choice(video_extensions)
        if random.random() > 0.5:
            folder = random.choice(['uploads', 'public', 'assets', 'videos'])
            date_path = fake.date_this_year().strftime("%Y/%m")
            filename = f"{fake.word()}_{random.randint(1,99)}"
            return f"https://{domain}/{folder}/{date_path}/{filename}{ext}"
        else:
            user_hash = fake.hexify(text='^^^^^^^^')
            file_hash = hashlib.md5(fake.uuid4().encode()).hexdigest()[:16]
            return f"https://{domain}/u/{user_hash}/{file_hash}{ext}"
    elif style_roll < 0.6:
        domain = random.choice(short_domains)
        short_id = fake.bothify(text='??###?') if random.random() > 0.5 else fake.pystr(min_chars=6, max_chars=10)
        return f"https://{domain}/{short_id}"
    elif style_roll < 0.8:
        domain = random.choice(short_domains)
        res_id = random.randint(100000, 9999999)
        return f"https://{domain}/api/v1/res/{res_id}"
    else:
        domain = random.choice(long_domains)
        paths = [fake.hexify(text='^^^^'), hashlib.md5(fake.word().encode()).hexdigest()[:8]]
        resource_id = hashlib.sha1(fake.uuid4().encode()).hexdigest()[:24]
        token = fake.hexify(text='^^^^^^^^')
        return f"https://{domain}/{'/'.join(paths)}/{resource_id}?auth={token}"

# ===== 2. Tools 定义常量 =====
TOOLS_DEFINITION = [
    {
        "name": "zai_extract_multi_video_segments",
        "description": "Use this tool to get specific clips from a video using timestamps. It extracts media between start and end times and returns their direct URLs.",
        "parameters": {
            "$defs": {
                "VideoSegment": {
                    "properties": {
                        "start": {
                            "description": "Start time of the segment in seconds.",
                            "title": "Start",
                            "type": "number"
                        },
                        "end": {
                            "description": "End time of the segment in seconds.",
                            "title": "End",
                            "type": "number"
                        }
                    },
                    "required": ["start", "end"],
                    "title": "VideoSegment",
                    "type": "object"
                }
            },
            "properties": {
                "video_url": {
                    "description": "The full HTTP(S) URL of the source video to be processed.",
                    "title": "Video Url",
                    "type": "string"
                },
                "segments": {
                    "description": "A list of time ranges to extract from the video. Each element MUST be a dictionary with 'start' and 'end' keys. Unit: Seconds (float).",
                    "examples": [
                        [
                            {"start": 1.5, "end": 10.0},
                            {"start": 20.0, "end": 25.5}
                        ]
                    ],
                    "items": {
                        "$ref": "#/$defs/VideoSegment"
                    },
                    "title": "Segments",
                    "type": "array"
                }
            },
            "required": ["video_url", "segments"],
            "title": "zai_extract_multi_video_segmentsArguments",
            "type": "object"
        }
    }
]

def find_files_with_extension(directory, extension):
    out_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    if len(out_files) == 0:
        print(f"[WARN] No files with extension {extension} found in {directory}")
        return []
    return out_files

def generate_uuid():
    return str(uuid.uuid4())

def build_tar_index_unistore(tar_path):
    try:
        index_info = {}
        ds = wids.IndexedTarSamples(path=str(tar_path), use_mmap=True)
        mmap_reader = ds.reader
        indices = mmap_reader.by_index
        
        for indice, mm_item in zip(indices, mmap_reader):
            if mm_item[0].endswith("mp4"):
                hash_md5 = hashlib.md5(mm_item[1]).hexdigest()
                assert mm_item[0] not in index_info, f"media name {mm_item[0]} is not unique"
                index_info[mm_item[0]] = (indice[1], hash_md5)
        
        with open(tar_path.replace(".tar", ".index"), "wb") as f:
            pickle.dump(index_info, f)
    except:
        print("Failed:", tar_path)
        print(traceback.format_exc())

def get_video_info(video_rel_path):
    if not video_rel_path:
        return None
    full_path = os.path.join(VIDEO_ROOT_DIR, video_rel_path)
    if not os.path.exists(full_path):
        if os.path.exists(video_rel_path):
            full_path = video_rel_path
        else:
            return None

    try:
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            return None
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = 0.0
        if fps > 0:
            duration = frame_count / fps
        cap.release()
        return [int(height), int(width), int(duration)]
    except Exception as e:
        print(f"[ERR] 读取视频失败 {full_path}: {e}")
        return None

def extract_tool_payload(text):
    if not text:
        return None, None

    pattern_code = r'```json\s*(.*?)\s*```'
    match = re.search(pattern_code, text, re.DOTALL)
    if match:
        json_payload = match.group(1).strip()
        cleaned_text = text.replace(match.group(0), "").strip()
        return json_payload, cleaned_text

    pattern_tool = r'<tool>\s*(.*?)\s*</tool>'
    match = re.search(pattern_tool, text, re.DOTALL)
    if match:
        json_payload = match.group(1).strip()
        cleaned_text = text.replace(match.group(0), "").strip()
        return json_payload, cleaned_text

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            json.loads(candidate)
            cleaned_text = (text[:start_idx] + text[end_idx+1:]).strip()
            return candidate, cleaned_text
        except json.JSONDecodeError:
            pass

    return None, None

# 你给的函数直接加到这里
def build_system_prompt(tools):
    system_prompt = '''# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>{tool_text}
</tools>
For each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}
<arg_key>{{arg-key-1}}</arg_key>
<arg_value>{{arg-value-1}}</arg_value>
<arg_key>{{arg-key-2}}</arg_key>
<arg_value>{{arg-value-2}}</arg_value>
...
</tool_call>'''
    tool_text = ""
    for tool in tools:
        if tool:
            tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
    system_prompt = system_prompt.format(tool_text=tool_text)
    messages = [
        {
            "role": "system",
            "text": system_prompt,
            "if_train": False
        }
    ]
    return messages


def convert_video_to_image_format(video_item, tar_uuid):
    main_uuid = generate_uuid()
    conversations = []
    reference_list = video_item.get("reference", [])
    video_tags = []
    clip_tags = []
    media_map = {}
    media_size = {}
    idx = 0

    # 1. 主视频
    main_video_path = video_item.get("video_path")
    if main_video_path:
        tag = f"video{idx}"
        video_tags.append(tag)
        media_map[tag] = [tar_uuid, os.path.basename(main_video_path)]
        media_size[tag] = get_video_info(main_video_path)
        idx += 1

    # 2. Clip 视频
    clip_urls = []
    raw_clips = video_item.get("clip_video_list", [])

    for raw_clip in raw_clips:
        tag = f"video{idx}"
        clip_path = ""
        clip_url = ""
        if isinstance(raw_clip, dict):
            clip_url = raw_clip.get("url")
            clip_path = raw_clip.get("path") or raw_clip.get("video_path") or raw_clip.get("file_path", "")
        else:
            clip_path = raw_clip
        if not clip_url:
            clip_url = generate_mixed_video_url()

        clip_urls.append(clip_url)
        video_tags.append(tag)
        clip_tags.append(tag)

        if clip_path:
            media_map[tag] = [tar_uuid, os.path.basename(clip_path)]
            media_size[tag] = get_video_info(clip_path)
        else:
            media_map[tag] = [tar_uuid, "unknown.mp4"]
            media_size[tag] = [0, 0, 0]

        idx += 1

    media_type = {k: "video" for k in media_map.keys()}

    # 3. Observation 占位符
    obs_parts = []
    for tag, url in zip(clip_tags, clip_urls):
        obs_parts.append(f"<|ZP_MM_PLH={tag}|>")
        obs_parts.append(f"<url>{url}</url>")
    observation_placeholder = "".join(obs_parts)

    # **加：在 conversations 最前面插入 system prompt**
    # 原有 TOOLS_DEFINITION 是全局的，这里调用 build_system_prompt
    conversations.extend(build_system_prompt(TOOLS_DEFINITION))

    first_assistant_processed = False
    for i, ref in enumerate(reference_list):
        t = ref.get("type")
        content_text = ref.get("text", "")

        if t == "Q":
            if i == 0 and len(video_tags) > 0 and "video0" in video_tags:
                content_text = f"<|ZP_MM_PLH=video0|>{content_text}"
            conversations.append({
                "role": "user",
                "text": content_text,
                "if_train": False
            })

        elif t == "A":
            if not first_assistant_processed:
                json_payload, cleaned_text = extract_tool_payload(content_text)
                if json_payload is None:
                    return None
                try:
                    parsed_obj = json.loads(json_payload)
                    compact_json = json.dumps(parsed_obj, separators=(',', ':'), ensure_ascii=False)
                except:
                    compact_json = re.sub(r'\n\s*', '', json_payload)
                tool_call_struct = {
                    "name": "zai_extract_multi_video_segments",
                    "arguments": compact_json
                }

                conversations.append({
                    "role": "assistant",
                    "text": cleaned_text,
                    "tool_calls": [tool_call_struct],
                    "if_train": True
                })
                first_assistant_processed = True
            else:
                conversations.append({
                    "role": "assistant",
                    "text": content_text,
                    "if_train": True
                })

        elif t == "O":
            current_output = observation_placeholder
            if content_text:
                current_output = f"{current_output}\n{content_text}"
            conversations.append({
                "role": "observation",
                "content": [{"output": current_output}],
                "if_train": False
            })

    if not first_assistant_processed:
        return None

    video_residual = deepcopy(video_item)
    video_residual.pop("reference", None)
    image_item = {
        "uuid": main_uuid,
        "media_map": media_map,
        "media_size": media_size,
        "media_type": media_type,
        "conversations": conversations,  # 已经包含 system prompt
        "metadata": {
            "type": "VideoQA",
            "source_path": video_item.get("video_path", ""),
            "orig_video_dict": video_residual
        }
    }
    return image_item

def convert_file(input_file, output_file, tar_uuid):
    print(f"Converting {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
    local_filtered_count = 0
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                video_item = json.loads(line)
                image_item = convert_video_to_image_format(video_item, tar_uuid)
                
                if image_item is None:
                    print("未找到image_number")
                    local_filtered_count += 1
                    continue
                
                f_out.write(json.dumps(image_item, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[ERR] 处理第 {line_num} 行时出错: {e}")

    global TOTAL_FILTERED
    TOTAL_FILTERED += local_filtered_count

def build_jsonl_directory(tar_dict, orig_dir, root_dir):
    all_jsonls = find_files_with_extension(orig_dir, ".jsonl")
    filename_to_uuid = {k: v['uuid'] for k, v in tar_dict.items()}
    
    new_dir = os.path.join(root_dir, "MetaFiles")
    os.makedirs(new_dir, exist_ok=True)
    
    for jsonl_path in all_jsonls:
        jsonl_name = os.path.basename(jsonl_path)
        file_name_stem = os.path.splitext(jsonl_name)[0] 
        corresponding_tar_name = file_name_stem + ".tar"
        
        tar_uuid = filename_to_uuid.get(corresponding_tar_name)
        
        if tar_uuid is None:
            print(f"[WARN] No corresponding tar UUID found for {jsonl_name}")
            continue
            
        new_jsonl_path = os.path.join(new_dir, jsonl_name)
        convert_file(jsonl_path, new_jsonl_path, tar_uuid)

def convert_dataset(orig_dir, dest_dir):
    print("Step 1: Processing Tar Files & Building Indices...")
    all_tars = find_files_with_extension(orig_dir, ".tar")
    tar_files_dir = os.path.join(dest_dir, "TarFiles")
    os.makedirs(tar_files_dir, exist_ok=True)
    
    index_dir = os.path.join(tar_files_dir, ".index")
    os.makedirs(index_dir, exist_ok=True)
    
    final_index = {}
    detailed_map = {} 
    
    for tar_path in tqdm(all_tars, desc="Processing Tars"):
        tar_name = os.path.basename(tar_path)
        u = str(uuid.uuid4())
        dest_tar_path = os.path.join(tar_files_dir, tar_name)
        
        if not os.path.exists(dest_tar_path):
             os.system(f"cp {tar_path} {dest_tar_path}")
        
        build_tar_index_unistore(dest_tar_path)
        
        final_index[u] = tar_name
        detailed_map[tar_name] = {"uuid": u, "tar_path": tar_path}

    print("Writing global index files...")
    with open(os.path.join(index_dir, "index.json"), "w") as f:
        json.dump(final_index, f, indent=4)
    with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
        pickle.dump(final_index, f)

    print("Step 2: Processing JSONL Files...")
    build_jsonl_directory(detailed_map, orig_dir, dest_dir)
    
    print("\n" + "="*40)
    print(f"处理结束。总共剔除数据: {TOTAL_FILTERED} 条")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists(VIDEO_ROOT_DIR):
        pass 
        
    os.makedirs(DEST_DIR, exist_ok=True)
    convert_dataset(ORIG_DIR, DEST_DIR)
    print(f"[OK] 转换完成，输出目录: {DEST_DIR}")
