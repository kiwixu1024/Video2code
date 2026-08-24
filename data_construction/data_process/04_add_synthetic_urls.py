import json
import random
import hashlib
from faker import Faker
from pipeline_config import data_path

fake = Faker()

def generate_mixed_video_url():
    """随机生成长短不一、风格迥异的真实感视频 URL"""
    
    # 基础域名配置
    short_domains = ['v.io', 'v.cc', 's.tv', 'm.net', 'cdn.li']
    long_domains = ['media-cache-cluster.internal.net', 'secure-delivery.prod.cloud', 'content-gateway.service.io']
    storage_domains = ['storage.googleapis.com', 's3.aws.com', 'oss-cn-beijing.aliyuncs.com', 'video-cdn.net']
    
    # 常见的视频后缀
    video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']

    # 决定生成风格
    style_roll = random.random()

    # --- 风格 A: 常规文件路径 (Standard File Path) --- 
    # 概率: 0.0 ~ 0.4 (40%)
    if style_roll < 0.4:
        domain = random.choice(storage_domains)
        ext = random.choice(video_extensions)
        
        # 随机决定是 "语义化文件名" 还是 "Hash文件名"
        if random.random() > 0.5:
            # 模拟：https://storage.googleapis.com/uploads/2024/01/demo_video.mp4
            folder = random.choice(['uploads', 'public', 'assets', 'videos'])
            date_path = fake.date_this_year().strftime("%Y/%m")
            filename = f"{fake.word()}_{random.randint(1,99)}"
            return f"https://{domain}/{folder}/{date_path}/{filename}{ext}"
        else:
            # 模拟：https://video-cdn.net/u/5f4dcc3b/source.mp4
            user_hash = fake.hexify(text='^^^^^^^^')
            file_hash = hashlib.md5(fake.uuid4().encode()).hexdigest()[:16]
            return f"https://{domain}/u/{user_hash}/{file_hash}{ext}"

    # --- 风格 B: 简洁短链 (Short & Direct) ---
    # 概率: 0.4 ~ 0.6 (20%)
    elif style_roll < 0.6:
        # 模拟：https://v.io/x92B7L
        domain = random.choice(short_domains)
        short_id = fake.bothify(text='??###?') if random.random() > 0.5 else fake.pystr(min_chars=6, max_chars=10)
        return f"https://{domain}/{short_id}"

    # --- 风格 C: API 资源 (Restful style) ---
    # 概率: 0.6 ~ 0.8 (20%)
    elif style_roll < 0.8:
        # 模拟：https://api.cdn.li/v1/video/8827101
        domain = random.choice(short_domains)
        res_id = random.randint(100000, 9999999)
        return f"https://{domain}/api/v1/res/{res_id}"

    # --- 风格 D: 复杂长链 (Messy & Encrypted) ---
    # 概率: 0.8 ~ 1.0 (20%)
    else:
        # 模拟：带token的长链接
        domain = random.choice(long_domains)
        paths = [fake.hexify(text='^^^^'), hashlib.md5(fake.word().encode()).hexdigest()[:8]]
        resource_id = hashlib.sha1(fake.uuid4().encode()).hexdigest()[:24]
        token = fake.hexify(text='^^^^^^^^')
        return f"https://{domain}/{'/'.join(paths)}/{resource_id}?auth={token}"

def process_jsonl(input_file, output_file):
    """读取 jsonl 并注入混合风格的 URL"""
    print(f"开始处理: {input_file} -> {output_file}")
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 注入随机生成的视频链接
            data['fake_video_url'] = generate_mixed_video_url()
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1

    print(f"处理完成！共处理 {count} 行数据。")


if __name__ == "__main__":
    # 你的文件路径
    input_path = data_path("raw_dataset_splitted_clipped.jsonl")
    output_path = data_path("raw_dataset_splitted_clipped_url.jsonl")
    
    process_jsonl(input_path, output_path)
