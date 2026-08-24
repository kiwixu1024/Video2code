import os
import json
import base64
from multiprocessing import Process
import openai
from openai import OpenAI
import math
from collections import defaultdict
from pipeline_config import data_path


openai.api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY is required")
client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)


# ========== 配置 ==========
# 请修改为包含 operation_screenshots 的新 jsonl 路径
INPUT_FILE = data_path("raw_dataset_splitted_clipped_url.jsonl")
IMAGE_DIR = data_path()
NEED_DIR = data_path("code_results") + os.sep
OUTPUT_PREFIX = NEED_DIR + "1/generate_worker" 
INDEX_PREFIX = NEED_DIR + "1/data_index_worker" 
JSON_OUTPUT_DIR = NEED_DIR.replace('_res/', '_output_jsons/')

NUM_WORKERS = int(os.environ.get("VIDEO2CODE_NUM_WORKERS", "4"))
CHUNK_SIZE = None 

# MODEL_NAME = "claude-3-7-sonnet-20250219-thinking"
# MODEL_NAME = "gpt-4o" # 请替换为你实际使用的支持视觉的模型
# MODEL_NAME = "claude-sonnet-4-5-20250929-thinking"
MODEL_NAME = os.environ.get("VIDEO2CODE_MODEL", "gpt-4.1")

REPEAT_TRY = False 
max_repeat_times = 1 
LOOP_UNTIL_RATIO = 0 
LOOP_MAX_TIMES = 2 
TOTAL = -1 
CONTINUE_FROM_LOOP = 1

# ===========================

def relative_to_absolute_path(relative_img_path):
    """将相对路径转换为绝对路径"""
    if not relative_img_path:
        return None
    # 拼接成完整路径
    image_dir = IMAGE_DIR
    absolute_img_path = os.path.join(image_dir, relative_img_path)
    return os.path.normpath(absolute_img_path)

def describe_action(action_detail):
    """
    根据 action_detail 解析操作类型、内容和 DOM 信息，
    返回对应的自然语言描述。
    """
    if not action_detail:
        return "无鼠标操作，滚动浏览页面"
    
    action = action_detail.get("action", "")
    # 防止 dom_info 为 None
    dom_info = action_detail.get("dom_info") or {}
    
    # 防止 attrs 为 None
    attrs = dom_info.get("attrs") or {}
    
    # 优先获取 attrs 中的 text，如果没有则获取 visible_text
    attrs_text = attrs.get("text", "")
    visible_text = dom_info.get("visible_text", "")
    dom_text = attrs_text if attrs_text else visible_text
    
    # 如果两种 text 都没有内容，则使用纯文本描述
    if not dom_text:
        if action == "click":
            dom_text_plain = "一个按钮"
        elif action == "input":
            dom_text_plain = "一个输入框"
        elif action == "select":
            dom_text_plain = "一个选择框"
        else:
            dom_text_plain = "一个元素"
        dom_text = dom_text_plain

    if action == "click":
        return f"点击了“{dom_text}”"
    elif action == "input":
        input_val = action_detail.get("input_val", "")
        return f"在“{dom_text}”里输入了“{input_val}”"
    elif action == "select":
        select_val = action_detail.get("value", "")
        return f"在“{dom_text}”选择框选择了“{select_val}”"
    else:
        return f"在“{dom_text}”执行了 {action} 操作"

def get_response(inputs, idx, worker_id, output_file, index_file, error_idxs):
    try:
        data_item = inputs[idx]
        
        # ================== 核心修改逻辑：构建图片列表和操作清单 ==================
        image_paths = []
        ops_data = data_item.get('operation_screenshots', [])
        
        if not ops_data:
            print(f"[Worker {worker_id}] Index {idx}: No screenshots found.")
            return

        # 1. 按 segment 分组
        seg_map = defaultdict(list)
        for op in ops_data:
            seg_map[op.get('segment_index', 0)].append(op)
        
        sorted_seg_idxs = sorted(seg_map.keys())
        global_img_idx = 1 # 全局图片索引从1开始
        
        # 2. 生成描述文本并收集图片路径
        generated_instruction = "【自动生成的图片与操作时间线对应关系】\n"
        
        for seg_idx in sorted_seg_idxs:
            ops = seg_map[seg_idx]
            generated_instruction += f"### 片段 Segment {seg_idx + 1}:\n"
            
            for op in ops:
                op_idx = op.get('operation_index', 0)
                frames = op.get('extracted_frames', [])
                if not frames:
                    continue
                
                start_img = global_img_idx
                end_img = global_img_idx + len(frames) - 1
                desc = describe_action(op.get("action_detail"))
                
                generated_instruction += f"  - 操作 {op_idx + 1}: {desc}\n"
                generated_instruction += f"    * 涵盖图片范围：第 {start_img} 张图 到 第 {end_img} 张图\n"
                
                # 遍历每张截屏，翻译类型并提取路径
                for frame in frames:
                    img_path_raw = frame.get('image_path')
                    abs_path = relative_to_absolute_path(img_path_raw)  # 假设这是外部定义的函数
                    image_paths.append(abs_path)
                    
                    f_type = frame.get('frame_type', '')
                    t_sec = frame.get('time_seconds', '未知')
                    
                    # ================= [修改点：匹配截屏状态描述，加入 select 补丁] =================
                    if f_type in ["before_click_duration", "before_instant_click", "before_input", "before_init_scroll", "before_select"]:
                        frame_desc = "操作前的初始画面"
                    elif f_type == "after_click_scroll_duration":
                        frame_desc = "页面正在滚动展示操作后产生的新内容"
                    elif f_type == "during_init_scroll":
                        frame_desc = "页面正在进行纯滚动浏览"
                    elif f_type == "after_instant_click":
                        frame_desc = "操作发生后的即时变化状态"
                    elif f_type == "after_input":
                        frame_desc = "长时间操作完成后的最终状态"
                    elif f_type == "during_select":
                        frame_desc = "正在进行选择操作过程中的变化画面"
                    elif f_type == "during_select_scroll":
                        frame_desc = "正在进行选择操作并伴随页面滚动的变化画面"
                    else:
                        frame_desc = "操作过程中的过渡画面"
                    # =================================================================================
                        
                    generated_instruction += f"      - 第 {global_img_idx} 张图: {frame_desc}\n"
                    global_img_idx += 1

            generated_instruction += "\n"

        # ================== Prompt 构建 ==================

        prompt_instruction = """
你非常擅长用 React 和 Tailwind 构建互动网页，能精确根据用户提供的多张网页截图还原完整HTML互动网页。

【界面要求】
1. 按照用户提供的首张网页截图，构建初始页面，一定要和给你的首张截图内容完全一致。
2. 不要遗漏任何细节，包括背景色、字体、字号、间距、边框、图标、文本等，都要和截图严格匹配。
3. 截图里的每一句文字都要原样呈现。
4. 图片内容请使用 https://picsum.photos/ 库中的真实图片，url类似https://picsum.photos/id/.../.../...。 **每张图片需显式列出url**，不要使用可复用的图片组件。每个网页组件的图片url一定要固定，不要用随机数每次重新生成。

【任务核心要求】
请拿出“显微镜”般的观察力，实现对网页交互的1:1像素级复刻：
1. **观察页面整体变化**：仔细观察下方提供的时间线截图序列。重点对比【操作前的初始画面】与【最终状态/即时变化状态】，不要遗漏页面上的任何细节（如右上角出现的弹窗提示、新增的列表项等）。对于标有【纯滚动浏览】的截图序列，请将其视为获取页面全局布局和隐藏内容的结构化拼图，确保整页的完整性。
2. **精准定位交互源头**：在【操作前的初始画面】截图中寻找到鼠标位置，推断出是哪个组件被点击或输入了，并确保在复刻网页中这些组件存在且具备与原本一样的能力。让它的功能和视觉反馈与截图中**一模一样**。
3. 所有给你的互动操作一定都要在生成的html中完美复刻，也就是同时具有完整的功能，且完成后页面与对应截图一致。

请使用以下库： 
<script src="https://cdn.jsdelivr.net/npm/react@18.0.0/umd/react.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18.0.0/umd/react-dom.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone/babel.js"></script>
用以下脚本引入 Tailwind： <script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

代码输出格式：
只输出完整的 <html></html> 标签内的代码
确保代码是能够直接渲染的完整HTML网页代码，不要用省略号省略
代码前后不要加 markdown “” 或 “html”
"""
        final_prompt_text = prompt_instruction + "\n" + generated_instruction
        
        # 构造 API 请求内容
        content_payload = [{"type": "text", "text": final_prompt_text}]

        # 添加图片
        for img_p in image_paths:
            if img_p and os.path.exists(img_p):
                with open(img_p, 'rb') as image_file:
                    imagebytes = base64.b64encode(image_file.read()).decode('utf-8')
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + imagebytes, # 假设是jpg，如果是png请改为image/png
                    },
                })
            else:
                print(f"[Worker {worker_id}] Image not found: {img_p}")

        if MODEL_NAME == "o4-mini-2025-04-16":
             response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content_payload}],
                max_completion_tokens=32000,
            )
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content_payload}],
                max_tokens=32000, # 根据模型调整
                stream=True,
                timeout=600
            )

        collected_messages = []
        for chunk in response:
            # 核心修复：先检查 choices 是否存在且不为空
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                # 检查 delta 是否有 content 属性，并且不为 None
                if hasattr(delta, 'content') and delta.content:
                    collected_messages.append(delta.content)
            
        output = "".join(collected_messages)

    except openai.BadRequestError as e:
        error_response = e.response.json()
        error_type = error_response.get('error', {}).get('code', 'Unknown')
        print(f"[Worker {worker_id}] BadRequestError: {error_type}")
        if error_type == 'content_policy_violation':
            output = 'Error: Content Policy Violation'
        else:
            error_idxs.append(idx)
            return
    except Exception as ex:
        error_idxs.append(idx)
        print(f"[Worker {worker_id}] Exception at {idx}: {ex}")
        return

    # 写入输出
    result = inputs[idx]
    result['mllm_generate'] = str(output)
    # 可选：把生成的prompt记录下来debug
    # result['generated_prompt_debug'] = final_prompt_text 

    with open(output_file, 'a', encoding='utf-8') as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + '\n')

    with open(index_file, 'w') as fidx:
        fidx.write(str(idx + 1))

    print(f"[Worker {worker_id}] Processed index: {idx}")

def worker_fn(worker_id, inputs, start_idx, end_idx):
    index_file = f"{INDEX_PREFIX}_{worker_id}.txt"
    output_file = f"{OUTPUT_PREFIX}_{worker_id}.jsonl"

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            try:
                start_idx = int(f.read().strip())
            except:
                pass

    error_idxs = []
    
    if start_idx >= end_idx:
        return

    for idx in range(start_idx, end_idx):
        get_response(inputs, idx, worker_id, output_file, index_file, error_idxs)

    if REPEAT_TRY: 
        repeat_times = 0
        while repeat_times < max_repeat_times and len(error_idxs) > 0:
            new_error_idxs = []
            for idx in error_idxs:
                get_response(inputs, idx, worker_id, output_file, index_file, new_error_idxs)
            error_idxs = new_error_idxs
            repeat_times += 1


def run_multiprocess():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_inputs = [json.loads(line) for line in f if line.strip()]

    total = len(all_inputs)
    processes = []
    
    real_chunk_size = CHUNK_SIZE if CHUNK_SIZE else math.ceil(total / NUM_WORKERS)

    for i in range(NUM_WORKERS):
        start = i * real_chunk_size
        if start >= total:
            break
        end = min(start + real_chunk_size, total)
        p = Process(target=worker_fn, args=(i, all_inputs, start, end))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def merge_res_to_json(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jsonl") and "generate_worker" in filename:
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():
                                json_obj = json.loads(line)
                                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                except:
                    pass

def read_json_lines(file):
    if not os.path.exists(file): return 0
    with open(file, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())

def update_params(loop_times):
    global INPUT_FILE, OUTPUT_PREFIX, INDEX_PREFIX, CHUNK_SIZE
    INPUT_FILE = JSON_OUTPUT_DIR + str(loop_times - 1) + '/' + 'remained.jsonl'
    OUTPUT_PREFIX = NEED_DIR + str(loop_times) + '/' + 'generate_worker'
    INDEX_PREFIX = NEED_DIR + str(loop_times) + '/' + 'data_index_worker'
    CHUNK_SIZE = math.ceil(read_json_lines(INPUT_FILE) / NUM_WORKERS)

def filter_exist(loop_times):
    file_a = INPUT_FILE
    file_b = JSON_OUTPUT_DIR + str(loop_times) + '/' + 'merged.jsonl'
    output_file = JSON_OUTPUT_DIR + str(loop_times) + '/' + 'remained.jsonl'
    KEY_FIELD = "id"

    remove_keys = set()
    if os.path.exists(file_b):
        with open(file_b, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        remove_keys.add(obj.get(KEY_FIELD))
                    except: pass

    with open(file_a, 'r', encoding='utf-8') as fa, open(output_file, 'w', encoding='utf-8') as out:
        for line in fa:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj.get(KEY_FIELD) not in remove_keys:
                        out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                except: pass
    
def merge_jsonl_files(file_list, output_file):
    merged_data = []
    for file in file_list:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
                merged_data.extend(data)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"合并完成！总共 {len(merged_data)} 条记录")

if __name__ == "__main__":
    total_lines = read_json_lines(INPUT_FILE)
    if total_lines == 0:
        print(f"错误：输入文件 {INPUT_FILE} 不存在或为空")
        exit()
        
    CHUNK_SIZE = math.ceil(total_lines / NUM_WORKERS)
    TOTAL = total_lines
    print(f"当前数据量为{total_lines}，每个进程处理{CHUNK_SIZE}条数据")
    
    os.makedirs(NEED_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    
    if CONTINUE_FROM_LOOP > 1:
        loop_times = CONTINUE_FROM_LOOP
        update_params(loop_times)
    else:
        loop_times = 1
        
    while loop_times <= LOOP_MAX_TIMES:
        print(f"=== 第{loop_times}次循环调用模型{MODEL_NAME} ===")
        cur_json_lines = read_json_lines(INPUT_FILE)
        cur_remain_ratio = cur_json_lines / TOTAL if TOTAL > 0 else 0
        
        print(f"当前剩余数据量{cur_json_lines}，剩余比例{cur_remain_ratio:.4f}")
        
        if cur_remain_ratio < LOOP_UNTIL_RATIO and cur_json_lines < TOTAL:
            print("剩余比例满足停止条件")
            break
        
        os.makedirs(NEED_DIR + str(loop_times) + '/', exist_ok=True)
        os.makedirs(JSON_OUTPUT_DIR + str(loop_times) + '/', exist_ok=True)
        
        run_multiprocess()
        merge_res_to_json(NEED_DIR + str(loop_times) + '/', JSON_OUTPUT_DIR + str(loop_times) + '/merged.jsonl')
        filter_exist(loop_times)
        
        loop_times += 1
        update_params(loop_times)
            
    file_list = [JSON_OUTPUT_DIR + str(i) + '/' + 'merged.jsonl' for i in range(1, loop_times)]
    output_file = JSON_OUTPUT_DIR + 'final_merged.jsonl'
    merge_jsonl_files(file_list, output_file)
