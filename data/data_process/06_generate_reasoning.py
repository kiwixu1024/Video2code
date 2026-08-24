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
NEED_DIR = data_path("think_results") + os.sep
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
LOOP_MAX_TIMES = 3
TOTAL = -1 
CONTINUE_FROM_LOOP = 1

# ===========================
def relative_to_absolute_path(relative_path):
    """将相对路径转换为基于 Base 目录的绝对路径"""
    # 注意：确保 INPUT_FILE 在全局上下文中已定义
    base_dir = IMAGE_DIR
    absolute_path = os.path.join(base_dir, relative_path)
    absolute_path = os.path.normpath(absolute_path)
    return absolute_path

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
    
def build_operation_info_from_data(prompt_data):
    """
    按全局顺序抽取所有 operation，结合 operation_screenshots 中的图片数据
    建立全局图文对应关系，并返回最终 Prompt 描述和图片路径列表。
    取消了 Segment 分组，仅展示操作性质，同时保留每张截图是操作前/操作后的状态描述。
    """
    segments = prompt_data.get("segments", [])
    operation_screenshots = prompt_data.get("operation_screenshots", [])
    
    if not segments:
        return "The video contains no valid operations.", []

    # 1. 建立截屏数据的索引字典: key 为 (segment_index, operation_index)
    screenshot_map = {}
    for screenshot_data in operation_screenshots:
        s_idx = screenshot_data.get("segment_index")
        o_idx = screenshot_data.get("operation_index")
        if s_idx is not None and o_idx is not None:
            screenshot_map[(s_idx, o_idx)] = screenshot_data.get("extracted_frames", [])

    lines = []
    lines.append("【视频与连续操作说明】")
    lines.append("这组交互视频中每个视频包含一个操作节点。请严格按照以下提供的操作性质和节点进行观察：\n")

    global_img_idx = 1
    global_op_idx = 1
    raw_image_paths = []

    for seg_idx, seg in enumerate(segments):
        operations = seg.get("operations", [])
        if not operations:
            continue
        
        for op_idx, op in enumerate(operations):   
            # 提取详细动作信息，兼容特殊的 select 补丁
            action_detail = op.get("action_detail", {})
            action_type = action_detail.get("action", "") if action_detail else None
            
            # 判别当前操作的物理性质
            if_scroll = op.get("if_scroll", False)
            subtype = op.get("subtype", "")
            
            if subtype == "init_scroll":
                op_nature = "【滚动浏览】无实际鼠标操作，而是滚动展示页面"
            elif subtype == "click" and if_scroll:
                op_nature = "【瞬时操作后滚动】一个鼠标操作，该操作导致页面变化超出可视区域，用户用滚动展示页面的全部变化"
            elif subtype == "input_duration":
                # ================= [新增对 select 补丁的判别] =================
                if action_type == "select":
                    if not if_scroll:
                        op_nature = "【连续选择】一个持续的下拉或选项块选择操作"
                    else:
                        op_nature = "【连续选择后滚动】一个持续的选项选择操作，操作过程伴随页面滚动"
                # ==============================================================
                else:
                    if not if_scroll:
                        op_nature = "【持续交互】一个连续鼠标操作"
                    else:
                        op_nature = "【持续交互后滚动】一个连续鼠标操作，操作导致页面变化超出可视区域，用户用滚动展示页面的全部变化"
            else:
                op_nature = "【瞬时反馈】一个鼠标瞬时操作，页面不滚动，产生即时视觉反馈"

            # 假设 describe_action 是你外部定义好的函数
            op_detail = describe_action(action_detail)
            lines.append(f"### 操作 {global_op_idx} \n  -操作内容 {op_detail} - 性质：{op_nature}")

            # 2. 从映射表中提取当前操作对应的截屏帧
            frames_data = screenshot_map.get((seg_idx, op_idx), [])
            
            if frames_data:
                start_img = global_img_idx
                end_img = global_img_idx + len(frames_data) - 1
                lines.append(f"  - 涵盖图片范围：第 {start_img} 张图 到 第 {end_img} 张图")
                
                for frame_info in frames_data:
                    f_type = frame_info.get("frame_type", "")
                    img_path = frame_info.get("image_path", "")
                    
                    # 收集图片路径
                    if img_path:
                        raw_image_paths.append(img_path)
                    
                    # 恢复对图片帧类型的自然语言描述 (加入补丁产生的各种 select 帧)
                    if f_type in ["before_click_duration", "before_instant_click", "before_input", "before_init_scroll", "before_select"]:
                        desc = "操作前的初始画面"
                    elif f_type == "after_click_scroll_duration":
                        desc = "页面正在滚动展示操作后产生的新内容"
                    elif f_type == "during_init_scroll":
                        desc = "页面正在进行纯滚动浏览"
                    elif f_type == "after_instant_click":
                        desc = "操作发生后的即时变化状态"
                    elif f_type == "after_input":
                        desc = "长时间操作完成后的最终状态"
                    # ================= [新增对 select 抽取帧的描述] =================
                    elif f_type == "during_select":
                        desc = "正在进行选择操作过程中的变化画面"
                    elif f_type == "during_select_scroll":
                        desc = "正在进行选择操作并伴随页面滚动的变化画面"
                    # ==============================================================
                    else:
                        desc = "操作过程中的过渡画面"
                        
                    lines.append(f"    * 第 {global_img_idx} 张图: {desc}")
                    global_img_idx += 1
            
            lines.append("") # 增加空行提升可读性
            global_op_idx += 1

    return "\n".join(lines), raw_image_paths

def get_response(inputs, idx, worker_id, output_file, index_file, error_idxs):
    try:
        prompt_data = inputs[idx]
        
        # 1. 基础角色设定与任务目标
        base_instruction = """你是一位精通前端开发（React + Tailwind）的资深工程师。
【任务目标】
你需要根据自己在上一轮观察长网页交互视频中捕捉到的**交互操作发生瞬间关键短视频**，构造**视觉化的视频观察过程**，同时**假装自己要生成复刻这个交互网页html代码**。

【核心任务】
你的任务是将这组**操作列表**，在脑海中“反向渲染”成**视觉化的视频观察过程**。
\n"""

        # 2. 拼接时间线与关键帧描述，同时获取对应的图片路径列表
        operation_info, raw_image_paths = build_operation_info_from_data(prompt_data)

        # 3. 输出结构规范
        rules_desc = """
【输出结构规范】
请严格按照以下要求输出，不要包含任何其他无关的寒暄：

### 输出要求：交互逻辑观察流 <think></think>
**这里是重点！** 请在 <think></think> 标签内，完成观察每个短视频的思考过程，假装自己正在研究网页中的交互时间点并试图复刻它。
你需要严格遵守以下规则：
1. **先观察网页整体布局**：在观看第一个交互短视频之前，先检查第一个视频的网页布局，这就是网页的初始布局，并细致地描述它。
2. **对于每一个操作进行观察**：
   - *观察鼠标位置*：对于每一个交互短视频，请观察**鼠标的位置变化**，因为一定是鼠标的位置与什么交互元素进行了操作导致页面发生了变化。
   - *确认点击对象*：通过观察点击后的反馈，在思考中明确确认鼠标点击的具体是哪个元素。
   - *构建交互叙事*：结合提示中的“操作后滚动”、“持续交互”等性质，描述操作引起的页面状态变化（例如：加载转圈、页面跳转、表格新增一行数据、滚动展示等）。
2.5 请注意！！！！有一种情况要特殊对待，就是在操作性质中标明这个操作是一个【滚动浏览】操作，此时无需描述上述的细节，只需要将滚动浏览中的各个截图拼凑起来，描述整个页面的构造即可。
3. **总结并假装自己要写代码**：在结尾时总结一下自己已经观察到了这个网页需要具备什么功能，并且声明自己开始编写代码。实际上不用编写。

**请注意：在此阶段不要提及任何代码术语（如 div, state, onclick 等），只描述视觉和逻辑。**

**请一定注意：你不是在看图片，而是在看视频！！！！不要在<think></think>中输出任何自己正在“假装看视频“或者读“图1”这种说法，而是完全伪造，只说自己正在看这几段自己剪切的视频。
请严格遵守以下禁令！！！1.不要提及任何视频时间，只说自己视频内容 2.不要提及用户发给你的参考操作信息，而是伪装成自己观察视频得到的信息。也就是不要在结果输出中直接输出（操作 1）等等用户发给你的参考内容。

"""

        # 最终合并完整的 Prompt
        final_prompt_text = base_instruction + operation_info + "\n" + rules_desc
    
        # 构造 API 请求内容
        content_payload = [{"type": "text", "text": final_prompt_text}]

        # ==========================================
        # 修复点：将相对路径转为绝对路径并验证文件存在
        # ==========================================
        for img_p_raw in raw_image_paths:
            # 假设你的 relative_to_absolute_path 函数在外部已定义
            abs_img_path = relative_to_absolute_path(img_p_raw) 
            if abs_img_path and os.path.exists(abs_img_path):
                with open(abs_img_path, 'rb') as image_file:
                    imagebytes = base64.b64encode(image_file.read()).decode('utf-8')
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + imagebytes,
                    },
                })
            else:
                print(f"[Worker {worker_id}] Image not found or path invalid: {abs_img_path}")

        # 发送请求
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
                timeout=600
            )
            
        output = response.choices[0].message.content

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
