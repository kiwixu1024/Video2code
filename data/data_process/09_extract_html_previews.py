import json
import os
import re
from pipeline_config import data_path

def extract_html_smart(text: str) -> str:
    """
    智能提取HTML内容：
    1. 先尝试提取 <answer>...</answer> 内部内容
    2. 在结果中寻找 <html>...</html>
    3. 如果都没有，尝试寻找 ```html ... ```
    """
    if not text:
        return ""

    # 1. 尝试提取 <answer> 标签内容
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        content = answer_match.group(1)
    else:
        content = text

    # 2. 尝试提取标准的 <html>...</html> (包含标签本身)
    html_tag_match = re.search(r'<html.*?>.*?</html>', content, re.DOTALL | re.IGNORECASE)
    if html_tag_match:
        return html_tag_match.group(0).strip()

    # 3. 如果没找到 html 标签，尝试提取 markdown 代码块
    code_block_match = re.search(r'```html(.*?)```', content, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # 4. 如果还是没找到，尝试通用代码块
    generic_block_match = re.search(r'```(.*?)```', content, re.DOTALL | re.IGNORECASE)
    if generic_block_match:
        return generic_block_match.group(1).strip()

    return content.strip()

def save_jsonl_htmls(jsonl_path, output_dir):
    if not os.path.exists(jsonl_path):
        print(f"错误：输入文件不存在 {jsonl_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Line {i+1}] JSON 解析失败，跳过")
                continue

            # 获取原始生成内容
            mllm_generate_str = obj.get("mllm_generate", "")
            
            # 智能提取 HTML (直接作为最终结果，移除反转义处理)
            final_html = extract_html_smart(mllm_generate_str)

            # 简单验证提取结果是否有效
            if "<html" not in final_html.lower():
                print(f"[Line {i+1}] ID: {obj.get('id')} - 未提取到有效 HTML，跳过")
                continue

            file_id = obj.get("id", f"unknown_{i}")
            html_out_path = os.path.join(output_dir, f"{file_id}.html")
            
            with open(html_out_path, "w", encoding="utf-8") as outf:
                outf.write(final_html)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"已处理: {success_count} 条...")

    print(f"\n处理完成！")
    print(f"总数据量: {total_count}")
    print(f"成功提取并保存: {success_count}")
    print(f"文件保存在: {output_dir}")

if __name__ == "__main__":
    # 路径配置
    INPUT_FILE = data_path("merged_results.jsonl")
    OUTPUT_DIR = data_path("generated_htmls")

    save_jsonl_htmls(
        jsonl_path=INPUT_FILE, 
        output_dir=OUTPUT_DIR
    )
