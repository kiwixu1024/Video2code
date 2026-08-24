import json
import random
import hashlib
from faker import Faker
from pipeline_config import data_path

# 初始化 Faker
fake = Faker()

# ==========================================
# 1. URL 生成逻辑
# ==========================================
def generate_mixed_video_url():
    """随机生成长短不一、风格迥异的真实感视频 URL"""
    
    short_domains = ['v.io', 'v.cc', 's.tv', 'm.net', 'cdn.li']
    long_domains = ['media-cache-cluster.internal.net', 'secure-delivery.prod.cloud', 'content-gateway.service.io']
    storage_domains = ['storage.googleapis.com', 's3.aws.com', 'oss-cn-beijing.aliyuncs.com', 'video-cdn.net']
    video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']

    style_roll = random.random()

    # --- 风格 A: 常规文件路径 --- 
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

    # --- 风格 B: 简洁短链 ---
    elif style_roll < 0.6:
        domain = random.choice(short_domains)
        short_id = fake.bothify(text='??###?') if random.random() > 0.5 else fake.pystr(min_chars=6, max_chars=10)
        return f"https://{domain}/{short_id}"

    # --- 风格 C: API 资源 ---
    elif style_roll < 0.8:
        domain = random.choice(short_domains)
        res_id = random.randint(100000, 9999999)
        return f"https://{domain}/api/v1/res/{res_id}"

    # --- 风格 D: 复杂长链 ---
    else:
        domain = random.choice(long_domains)
        paths = [fake.hexify(text='^^^^'), hashlib.md5(fake.word().encode()).hexdigest()[:8]]
        resource_id = hashlib.sha1(fake.uuid4().encode()).hexdigest()[:24]
        token = fake.hexify(text='^^^^^^^^')
        return f"https://{domain}/{'/'.join(paths)}/{resource_id}?auth={token}"

# ==========================================
# 2. Prompt 库配置
# ==========================================
PROMPT_LIBRARY = [
    "请根据我发给你的视频生成交互网页",
    "请分析视频中的操作并生成对应的网页界面",
    "请从视频推断用户交互逻辑并构建页面",
    "分析视频并输出网页交互代码",
    "请将视频内容转成前端网页",
    "根据视频互动设计对应的UI界面",
    "解析视频动作并生成网页脚本",
    "请基于视频交互生成应用前端",
    "从视频中获取信息并生成响应式网页",
    "根据用户视频生成具体的网页结构",
    "Please generate an interactive webpage based on the video",
    "Analyze the video interactions and generate a web interface",
    "Infer user logic from the video and construct the HTML",
    "Turn the video actions into a functional website",
    "Generate the frontend from the recorded video sequence",
    "Design a UI according to the video interactions",
    "Parse the video actions to output JS+HTML code",
    "Build a responsive page based on video behaviour",
    "Create an interactive application from the video",
    "Produce the page layout matching the video demonstrations"
]

URL_TEMPLATES = [
    "<url>{url}</url>"
]

# ==========================================
# 3. 核心处理逻辑
# ==========================================

def process_function_call_text(text):
    """
    检查并修复 function_call_generate 文本的 <think> 格式
    Returns: (new_text, status)
    status: 'fixed', 'failed', 'skipped' (already good)
    """
    if not text:
        return text, "failed" # 空内容算作失败

    stripped_text = text.strip()

    # 1. 检查是否已经是 <think> 开头且包含 </think>
    if stripped_text.startswith("<think>") and "</think>" in stripped_text:
        return text, "skipped"

    # 2. 如果不符合，寻找 <tool> 标签的起始位置
    # 我们假设 <tool> 是工具调用的开始
    tool_start_idx = text.find("<tool>")
    
    if tool_start_idx != -1:
        # 找到了 <tool>，把前面的内容包起来
        thought_content = text[:tool_start_idx]
        tool_content = text[tool_start_idx:]
        
        # 构造新文本: <think>原始内容</think><tool>...</tool>
        # 注意：这里不做额外的 strip，保留原始空格，只是添加标签
        new_text = f"<think>{thought_content}</think>{tool_content}"
        return new_text, "fixed"
    else:
        # 既没有 <think> 也没有 <tool>，无法自动修复
        return text, "failed"

def build_reference(data):
    """
    构造 reference 列表 [Q, A, O, A]
    """
    # --- Q 部分 ---
    fake_url = data.get("fake_video_url", "")
    url_prefix = ""
    if fake_url:
        template = random.choice(URL_TEMPLATES)
        url_prefix = f"<url>{fake_url}</url>"
    
    instruction = random.choice(PROMPT_LIBRARY)
    final_prompt = f"{url_prefix}{instruction}"

    # --- O 部分 (clip_video_list) ---
    clips = data.get("clip_video_list", [])
    if clips and isinstance(clips, list):
        for clip in clips:
            if isinstance(clip, dict):
                clip['url'] = generate_mixed_video_url()

    # --- 构造结构 ---
    reference_list = [
        {
            "type": "Q",
            "text": final_prompt,
            "content": data.get("video_path", None)
        },
        {
            "type": "A",
            # 注意：这里的 data["function_call_generate"] 应该已经在外部被 process_function_call_text 处理过了
            "text": data.get("function_call_generate", ""),
            "content": None
        },
        {
            "type": "O",
            "text": "",
            "content": clips 
        },
        {
            "type": "A",
            "text": data.get("mllm_generate", ""),
            "content": None
        }
    ]
    return reference_list

def process_jsonl(input_path, output_path):
    """
    读取 JSONL 文件，修复 <think> 标签，构造 reference 并清理旧字段
    """
    total_count = 0
    fixed_count = 0
    failed_count = 0
    skipped_count = 0 # 原本就是好的

    print(f"开始处理: {input_path} -> {output_path}")
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 格式错误: {e}")
                continue

            # =========================================
            # [NEW] 检查并处理 function_call_generate
            # =========================================
            raw_func_text = data.get("function_call_generate", "")
            new_func_text, status = process_function_call_text(raw_func_text)

            if status == "fixed":
                fixed_count += 1
                data["function_call_generate"] = new_func_text
            elif status == "failed":
                failed_count += 1
                continue
                # 失败的话保持原样，或者你可以选择清空，这里保持原样
            else: # skipped
                skipped_count += 1

            # =========================================
            # 构造 Reference
            # =========================================
            data["reference"] = build_reference(data)

            # 删除不需要的中间字段
            data.pop("mllm_generate", None)
            data.pop("function_call_generate", None)

            # 写入文件
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            total_count += 1
            
    print("-" * 40)
    print(f"处理完成！总数据量: {total_count}")
    print(f"无需修改 (<think> 格式正确): {skipped_count}")
    print(f"成功修复 (添加 <think> 包裹): {fixed_count}")
    print(f"修复失败 (未找到 <tool> 标签): {failed_count}")
    print("-" * 40)

if __name__ == "__main__":
    # 配置你的路径
    input_jsonl = data_path("merged_results_clean.jsonl")
    output_jsonl = data_path("data.jsonl")

    process_jsonl(input_jsonl, output_jsonl)
    print(f"[OK] 转换完成 -> {output_jsonl}")
