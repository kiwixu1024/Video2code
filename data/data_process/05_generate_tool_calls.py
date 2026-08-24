import os
import json
import base64
from multiprocessing import Process
import math
import time
from collections import defaultdict
import openai
from openai import OpenAI
from pipeline_config import data_path

# ========== 配置 (结合 Code 2 的新配置) ==========
# 输入数据文件 (使用 Code 2 的新路径)
INPUT_FILE = data_path("raw_dataset_splitted_clipped_url.jsonl")
# 基础图片目录 (使用 Code 2 的路径)
IMAGE_DIR = data_path()
# 中间结果输出目录 (使用 Code 2 的路径)
NEED_DIR = data_path("long_video_understanding_results") + os.sep
OUTPUT_PREFIX = NEED_DIR + "1/generate_worker" # 第一次循环的输出前缀
INDEX_PREFIX = NEED_DIR + "1/data_index_worker" # 第一次循环的索引前缀
JSON_OUTPUT_DIR = NEED_DIR.replace('_res/', '_output_jsons/') # 输出json的目录

NUM_WORKERS = int(os.environ.get("VIDEO2CODE_NUM_WORKERS", "4"))
CHUNK_SIZE = None  # 每个进程处理多少条

# 设置 GLM API 的 client
openai.api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY is required")
client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

# 模型选择
# MODEL_NAME = "gpt-4o" # 如果支持视觉，可以用这个作为 placeholder
MODEL_NAME = os.environ.get("VIDEO2CODE_MODEL", "gpt-4.1")

REPEAT_TRY = False # 单次循环是否重复请求未完成的数据
max_repeat_times = 1 # 单次循环内的重复次数

LOOP_UNTIL_RATIO = 0 # 停止比例
LOOP_MAX_TIMES = 3 # 最大循环次数
TOTAL = -1 # 数据总量，自动计算
CONTINUE_FROM_LOOP = 1 # 从第几次循环开始

# ============================================

def relative_to_absolute_path(relative_path):
    """将相对路径转换为基于 IMAGE_DIR 的绝对路径"""
    # 拼接成完整路径
    absolute_path = os.path.join(IMAGE_DIR, relative_path)
    # 规范化路径
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

def build_operation_info_with_screenshots_for_prompt(prompt_data):
    """
    结合时间线、截图索引和操作描述，构建 Prompt 中的【用户时间线与操作说明】片段。
    它不仅提供时间，还明确指出每一时刻对应的截图范围，并收集图片路径。
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
    lines.append("【用户时间线与操作参考说明】")
    lines.append("这是一个长视频。我截取了其中关键时刻的画面作为短视频帧（图片序列）发给你作为参考。")
    lines.append("下面的列表描述了视频中的用户操作及其发生的**大概时间点**，并且明确指出了这个操作发生瞬间在**你收到的图片参考（短视频帧）中**的具体范围。\n")

    global_img_idx = 1
    global_op_idx = 1
    raw_image_paths = []

    for seg_idx, seg in enumerate(segments):
        operations = seg.get("operations", [])
        if not operations:
            continue
        
        for op_idx, op in enumerate(operations):   
            # 判别当前操作的时间范围
            start = op.get("start")
            end = op.get("end")
            if start is None or end is None:
                continue

            # 判别操作性质 (用于在 <think> 中更准确地描述)
            if_scroll = op.get("if_scroll", False)
            subtype = op.get("subtype", "")
            
            if subtype == "init_scroll":
                op_nature = "【滚动浏览】无实际鼠标操作，而是滚动展示页面"
            elif subtype == "click" and if_scroll:
                op_nature = "【瞬时操作后滚动】一个鼠标操作，该操作导致页面变化超出可视区域，用户用滚动展示页面的全部变化"
            elif subtype == "input_duration":
                if not if_scroll:
                    op_nature = "【持续交互】一个鼠标连续操作"
                else:
                    op_nature = "【持续交互后滚动】一个鼠标连续操作，操作导致页面变化超出可视区域，用户用滚动展示页面的全部变化"
            else:
                op_nature = "【瞬时反馈】一个鼠标瞬时操作，页面不滚动，产生即时视觉反馈"

            op_detail = describe_action(op.get("action_detail"))

            # 2. 从映射表中提取当前操作对应的截屏帧
            frames_data = screenshot_map.get((seg_idx, op_idx), [])
            
            if frames_data:
                start_img = global_img_idx
                end_img = global_img_idx + len(frames_data) - 1
                
                # 构建带有图片范围和时间线的操作说明行
                description_line = (
                    f"### 参考操作 {global_op_idx}\n"
                    f"  - 时间: 从 {start:.1f} 秒 到 {end:.1f} 秒\n"
                    f"  - 操作内容: {op_detail} (性质: {op_nature})\n"
                    f"  - 参考截图范围: 第 {start_img} 张图 到 第 {end_img} 张图"
                )
                lines.append(description_line)
                
                for frame_info in frames_data:
                    img_path = frame_info.get("image_path", "")
                    # 收集图片路径
                    if img_path:
                        raw_image_paths.append(img_path)
                    global_img_idx += 1
            
            lines.append("") # 增加空行提升可读性
            global_op_idx += 1

    return "\n".join(lines), raw_image_paths


def get_response(inputs, idx, worker_id, output_file, index_file, error_idxs):
    try:
        prompt_data = inputs[idx]
        
        # 1. 基础角色设定与任务目标 (结合 Code 1 和 Code 2 的角色和任务)
        fake_url = prompt_data.get('fake_video_url', 'https://placeholder.url') # placeholder

        base_instruction = f"""你是一位精通用户界面（UI）交互分析的视频理解专家，同时也具备深厚的前端工程背景。

【任务背景】
我将为你提供一个**长视频文件的URL**，以及一组**我为你准备的短视频截图（图片序列）**。
这些截图是长视频中用户进行关键交互操作（点击、输入、选择、滚动等）发生瞬间的关键帧，涵盖了操作前的初始状态、操作中、以及操作后的页面变化。用户同时也会给你每一组操作大概在长视频中进行了什么操作。
本视频URL: {fake_url}

【核心任务】
你需要将我发给你的操作列表，在脑海中与这些短视频截图映射，然后**伪造一个你正在看完整长视频的思考过程**。在这个过程中，你要通过视觉画面内容（即你参考的这些截图），**描述每一个交互发生在什么时间段，具体做了什么，页面有何反馈。**
最终，你需要调用工具对这些片段进行批量剪切。

【输出结构规范】
请严格按照以下要求输出，不要包含任何寒暄：

### 输出要求一：长视频观察与伪造思考过程 <think></think>
**这里是重点！** 请在 <think></think> 标签内，完成上述的“伪装独立发现和观察”过程，假装自己正在研究网页中的交互时间点，并细致地描述它。
你需要严格遵守以下规则：
1. **假装独立发现**：**不要在思考过程中提及“参考操作列表”、“用户给定的列表”或“第几张图片”等字眼。** 你的语气必须像是你正在一帧一帧看完整视频，然后**独立发现了这些操作**。
   * *错误示范*：“根据参考操作 1，我看到...”
   * *正确示范*：“在浏览视频开头时，我注意到在 4.5秒 左右，鼠标光标移动到了‘登录’按钮，随后页面跳转...”
2. **详细描述操作与视觉反馈**：对于你提到的每一个时间片段（你可以基于下面【用户时间线与操作参考说明】中的时间稍微修改，让它看起来更像独立发现），你必须精准描述该时段内画面上发生了什么。包括：
   * 鼠标点击了什么具体的按钮/链接？
   * 输入框里输完了什么字？
   * 页面对此有何响应（弹窗、变色、跳转）？
   * **请注意！使用你的这些短视频截图中的视觉信息（比如哪张是 before、哪张是 after，或者操作过程）来丰富你的描述。** 比如：“在 4.5秒 秒，我看到一个初始的页面布局（对应的其实是 before 截图），随后鼠标移动并点击，在 5.0秒 秒页面加载完成（对应的其实是 after 截图）”。
2.5 特殊对待！对于标明为【滚动浏览】的操作，无需描述具体操作细节，只需要描述滚动展示了整个页面的构造即可。

### 输出要求二：工具调用声明与工具调用 <tool></tool>
在完成思考后，用自然语言简要声明你已识别出关键交互片段，并将调用工具进行剪切提取。
* *范例*：经过分析，我独立识别出了视频中的关键交互操作。我将调用 `zai_extract_multi_video_segments` 工具来精确剪切并提取以下关键交互片段：第3.5秒到第5.5秒，第11.2秒到第13.2秒。

### 输出要求三：工具调用 <tool></tool>
最后在 `<tool></tool>` 标签中，输出严格符合以下格式的 JSON 数据。
* **注意**：工具接受两个主要参数：`video_url` (我传给你的URL) 和 `segments` (包含 start/end 的对象列表)。请一定注意！！！！！`video_url` 部分一定要写我传给你的完全一致的 URL！！！！！
* JSON 格式范例：
  {{ "video_url": "{fake_url}", "segments": [{{ "start": 3.5, "end": 5.5 }}, {{ "start": 11.2, "end": 13.2 }}] }}
"""

        # 2. 拼接时间线与关键帧描述，同时获取对应的图片路径列表
        operation_info, raw_image_paths = build_operation_info_with_screenshots_for_prompt(prompt_data)

        # 最终合并完整的 Prompt
        final_prompt_text = base_instruction + operation_info + "\n" + "\n請嚴格遵守以上輸出結構規範，不要輸出任何無關信息。"
    
        # 构造 API 请求内容
        content_payload = [{"type": "text", "text": final_prompt_text}]

        # ==========================================
        # 修复点：将相对路径转为绝对路径并验证文件存在
        # ==========================================
        for img_p_raw in raw_image_paths:
            # 假设你的 relative_to_absolute_path 函数在外部已定义
            abs_img_path = relative_to_absolute_path(img_p_raw) 
            if abs_img_path and os.path.exists(abs_img_path):
                try:
                    with open(abs_img_path, 'rb') as image_file:
                        imagebytes = base64.b64encode(image_file.read()).decode('utf-8')
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + imagebytes,
                        },
                    })
                except Exception as ex:
                    print(f"[Worker {worker_id}] Error encoding image {abs_img_path}: {ex}")
            else:
                print(f"[Worker {worker_id}] Image not found or path invalid: {abs_img_path}")

        # 发送请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content_payload}],
        )
            
        output = response.choices[0].message.content

    except Exception as ex:
        error_idxs.append(idx)
        print(f"[Worker {worker_id}] Exception at {idx}: {ex}")
        return

    # =========================================================================
    # 结果校验逻辑 (保留 Code 1 的质量检查逻辑)
    # =========================================================================
    
    # 1. 计算喂给模型的切好 segments (operations) 数量
    total_segments_count = 0
    input_segments = prompt_data.get("segments", [])
    if input_segments:
        for seg in input_segments:
            ops = seg.get("operations", [])
            total_segments_count += len(ops)
    
    # 2. 计算输出纯结果的字符串数
    content_str = str(output)
    content_len = len(content_str)
    
    should_discard = False
    
    # 规则1: 数量为1 且 长度小于500 -> 放弃
    if total_segments_count == 1 and content_len < 0: #500
        should_discard = True
        
    # 规则2: 数量>=2 且 长度小于1000 -> 放弃
    elif total_segments_count >= 2 and content_len < 0: #1000
        should_discard = True

    if should_discard:
        print(f"[Worker {worker_id}] ⚠️ Quality Check Failed (SKIP): ID={prompt_data.get('id')}, Segs={total_segments_count}, Len={content_len}. Waiting for next loop.")
        # 关键：更新 index 文件，让 worker 认为处理完了继续往下走
        with open(index_file, 'w') as fidx:
            fidx.write(str(idx + 1))
        return
    # =========================================================================

    # 写入输出
    result = inputs[idx]
    # Code 1 使用 'function_call_generate'，Code 2 使用 'mllm_generate'。这里沿用 'function_call_generate'
    result['function_call_generate'] = str(output)

    with open(output_file, 'a', encoding='utf-8') as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + '\n')

    with open(index_file, 'w') as fidx:
        fidx.write(str(idx + 1))

    print(f"[Worker {worker_id}] Processed index: {idx}")

def worker_fn(worker_id, inputs, start_idx, end_idx):
    index_file = f"{INDEX_PREFIX}_{worker_id}.txt"
    output_file = f"{OUTPUT_PREFIX}_{worker_id}.jsonl"

    # 读取当前进度 (断点重连)
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            try:
                start_idx = int(f.read().strip())
            except ValueError:
                pass # 如果文件为空或损坏，保持原 start_idx

    error_idxs = []

    # 确保当前进度不超过分配给该 worker 的最大范围
    if start_idx >= end_idx:
        return

    for idx in range(start_idx, end_idx):
        get_response(inputs, idx, worker_id, output_file, index_file, error_idxs)

    if REPEAT_TRY: # 对之前没成功的数据重复发送请求
        repeat_times = 0
        while repeat_times < max_repeat_times and len(error_idxs) > 0:
            print(f"[Worker {worker_id}] Retrying {len(error_idxs)} failed items...")
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
    
    # 重新计算实际的 chunk size，防止最后一个 worker 拿不到数据
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
    # 只合当前worker输出，防止重复累加
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(folder_path)): # 加个排序，保证顺序相对固定
            if filename.endswith(".jsonl") and "generate_worker" in filename: 
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():
                                json_obj = json.loads(line)
                                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error merging file {filename}: {e}")

def read_json_lines(file):
    if not os.path.exists(file):
        return 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)

def update_params(loop_times):
    global INPUT_FILE, OUTPUT_PREFIX, INDEX_PREFIX, CHUNK_SIZE
    INPUT_FILE = JSON_OUTPUT_DIR + str(loop_times - 1) + '/' + 'remained.jsonl'
    OUTPUT_PREFIX = NEED_DIR + str(loop_times) + '/' + 'generate_worker'
    INDEX_PREFIX = NEED_DIR + str(loop_times) + '/' + 'data_index_worker'
    CHUNK_SIZE = math.ceil(read_json_lines(INPUT_FILE) / NUM_WORKERS)

def filter_exist(loop_times):
    # 输入文件
    file_a = INPUT_FILE  # 待处理的主文件
    file_b = JSON_OUTPUT_DIR + str(loop_times) + '/' + 'merged.jsonl'  # 本轮已处理完的数据

    # 输出文件
    output_file = JSON_OUTPUT_DIR + str(loop_times) + '/' + 'remained.jsonl'

    KEY_FIELD = "id" # 根据 id 去重

    # 读取 file_b 中要去除的字段集合
    remove_keys = set()
    if os.path.exists(file_b):
        with open(file_b, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if KEY_FIELD in obj:
                            remove_keys.add(obj.get(KEY_FIELD))
                    except:
                        pass

    # 处理 file_a
    with open(file_a, 'r', encoding='utf-8') as fa, open(output_file, 'w', encoding='utf-8') as out:
        for line in fa:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj.get(KEY_FIELD) not in remove_keys:
                        out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                except:
                    pass
    
def merge_jsonl_files(file_list, output_file):
    merged_data = []
    # 读取每个文件的内容
    for file in file_list:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
                merged_data.extend(data)
            print(f"文件 {file} 包含 {len(data)} 条记录")

    # 写入合并后的数据到新文件
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
        print(f"从第{CONTINUE_FROM_LOOP}次循环开始")
        loop_times = CONTINUE_FROM_LOOP
        update_params(loop_times)
    else:
        loop_times = 1
        
    while loop_times <= LOOP_MAX_TIMES:
        print(f"=== 第{loop_times}次循环调用模型 ===")
        
        cur_json_lines = read_json_lines(INPUT_FILE)
        if TOTAL > 0:
            cur_remain_ratio = cur_json_lines / TOTAL
        else:
            cur_remain_ratio = 0
            
        print(f"当前剩余数据量{cur_json_lines}，剩余比例{cur_remain_ratio:.4f}")
        
        if cur_remain_ratio < LOOP_UNTIL_RATIO and cur_json_lines < TOTAL: # 只有当真正减少了且低于比例才停，防止一开始就停
            print(f"剩余数据比例少于{LOOP_UNTIL_RATIO}，停止循环")
            break
        
        # 创建本轮目录
        os.makedirs(NEED_DIR + str(loop_times) + '/', exist_ok=True)
        os.makedirs(JSON_OUTPUT_DIR + str(loop_times) + '/', exist_ok=True)
        
        # 运行多进程
        run_multiprocess()
        
        # 合并结果
        merge_res_to_json(NEED_DIR + str(loop_times) + '/', JSON_OUTPUT_DIR + str(loop_times) + '/merged.jsonl')
        
        # 过滤剩余
        filter_exist(loop_times)
        
        loop_times += 1
        update_params(loop_times)
            
    # 最后合并所有轮次的结果
    file_list = [JSON_OUTPUT_DIR + str(i) + '/' + 'merged.jsonl' for i in range(1, loop_times)]
    output_file = JSON_OUTPUT_DIR + 'final_merged.jsonl'
    merge_jsonl_files(file_list, output_file)
    print(f"运行结束，最终结果保存在: {output_file}")
