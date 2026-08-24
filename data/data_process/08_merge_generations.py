import json
import re
import os
import random
from collections import Counter
from pipeline_config import data_path

def read_jsonl_file(filepath):
    """读取一个 JSONL 文件，返回 {id: 数据} 字典"""
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"[ERROR] 文件未找到: {filepath}")
        return data_dict
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[ERROR] JSON decode error in {filepath}: {line[:100]}")
                continue
            _id = obj.get('id')
            if _id is not None:
                data_dict[_id] = obj
    return data_dict

def extract_between_tags(text, tag):
    """提取第一个 <tag ...>...</tag>，返回包括标签本身。"""
    pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else None

def merge_mllm_generate(file1_dict, file2_dict):
    """按 id 合并，统计具体过滤原因并返回抽样列表"""
    merged_data = {}
    no_think_data_list = []
    
    # 用于统计原因
    stats = Counter()
    
    for _id, obj1 in file1_dict.items():
        obj2 = file2_dict.get(_id)
        if not obj2:
            stats["第二文件中没有此 id"] += 1
            print(f"[FILTER] id={_id} 被过滤: 第二文件中没有此 id")
            continue
            
        mg1 = obj1.get('mllm_generate', '')
        mg2 = obj2.get('mllm_generate', '')
        think_part = extract_between_tags(mg1, 'think')
        html_part  = extract_between_tags(mg2, 'html')
        fake_url = obj1.get("fake_video_url")
        
        if think_part and html_part:
            # 修改点：参考给定的格式，使用 Markdown 的 ```html 包裹
            new_mllm_generate = f"{think_part}\n```html\n{html_part.strip()}\n```"
            
            merged_data[_id] = {
                "id": _id,
                "video_path": obj1.get("video_path"),
                "segments": obj1.get("segments"),
                "clip_video_list": obj1.get("clip_video_list"),
                "mllm_generate": new_mllm_generate,
                "fake_video_url": fake_url
            }
        else:
            if not think_part and not html_part:
                reason = "同时缺失 think 和 html 标签"
                no_think_data_list.append(obj1)
            elif not think_part:
                reason = "没找到 think 标签"
                no_think_data_list.append(obj1)
            else: # not html_part
                reason = "没找到 html 标签"
            
            stats[reason] += 1
            print(f"[FILTER] id={_id} 被过滤: {reason}")
            
    return merged_data, stats, no_think_data_list

def attach_function_call(merged_dict, long_video_dict):
    """从 long_video_understanding_results 中合并 function_call_generate"""
    for _id, data in merged_dict.items():
        lv_obj = long_video_dict.get(_id)
        if lv_obj:
            data["function_call_generate"] = lv_obj.get("function_call_generate")
        else:
            print(f"[WARN] id={_id} 在long_video_understanding_results中未找到 function_call_generate")

def write_jsonl(data_collection, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    iterable = data_collection.values() if isinstance(data_collection, dict) else data_collection
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in iterable:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    base_dir = data_path()
    output_file = os.path.join(base_dir, 'merged_results.jsonl')
    no_think_sample_file = os.path.join(base_dir, 'no_think_samples_50.jsonl')
    
    # think_file = os.path.join(base_dir, 'think_results_combined_final/final_merged.jsonl')
    think_file = os.path.join(base_dir, 'think_results/final_merged.jsonl')
    code_file = os.path.join(base_dir, 'code_results/final_merged.jsonl')
    lv_file = os.path.join(base_dir, 'long_video_understanding_results/final_merged.jsonl')

    think_data = read_jsonl_file(think_file)
    code_data = read_jsonl_file(code_file)
    lv_data = read_jsonl_file(lv_file)

    # 1. 合并并获取统计信息
    merged_dict, filter_stats, no_think_list = merge_mllm_generate(think_data, code_data)

    # 2. 抽样保存
    if no_think_list:
        sample_num = min(len(no_think_list), 50)
        sampled_data = random.sample(no_think_list, sample_num)
        write_jsonl(sampled_data, no_think_sample_file)
    
    # 3. 补充 long video 字段
    attach_function_call(merged_dict, lv_data)

    # 4. 写入总结果
    write_jsonl(merged_dict, output_file)

    # 5. 输出详细统计结果
    print("\n" + "="*30)
    print("      数据合并统计报告")
    print("="*30)
    print(f"成功合并条目: {len(merged_dict)}")
    print(f"总过滤条目:   {sum(filter_stats.values())}")
    print("-" * 30)
    print("具体过滤原因分布:")
    for reason, count in filter_stats.items():
        print(f" - {reason}: {count} 条")
    print("-" * 30)
    if no_think_list:
        print(f"[采样] 缺失think的数据已抽取 {min(len(no_think_list), 50)} 条保存至: {os.path.basename(no_think_sample_file)}")
    print("="*30)
