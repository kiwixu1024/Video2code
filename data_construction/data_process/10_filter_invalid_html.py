
import json
import os
import asyncio
import io
import numpy as np
from PIL import Image, ImageStat
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm
from pipeline_config import data_path

# ================= 配置区域 =================
INPUT_JSONL = data_path("merged_results.jsonl")
OUTPUT_JSONL = data_path("merged_results_clean.jsonl")
HTML_FOLDER = data_path("generated_htmls")
BLANK_IMG_FOLDER = data_path("blank_evidence")
ID_FIELD = "id"                        # JSONL 中的 ID 字段名
IS_BLANK_THRESHOLD = 5.0               # 空白判定阈值 (标准差)
CONCURRENCY = 24                        # [新增] 并发数量 (建议设置为 CPU 核心数 x 2 或 8-16)
TIMEOUT = 20000                         # 页面加载超时时间 (毫秒)，防止卡死
# ===========================================

import io
from PIL import Image, ImageStat

def is_image_blank_or_solid(image_bytes, blank_threshold=10.0, solid_color_threshold=10.0):
    """
    判断图片是否空白（无内容）或纯色页面
    :param blank_threshold: 空白阈值（灰度标准差，越小画面亮度越均匀）
    :param solid_color_threshold: 纯色阈值（RGB各通道标准差，越小画面颜色越纯粹）
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # 1. 空白判定（通常针对全白、全黑或灰度均匀的文档页）
        grayscale_image = image.convert('L')
        gray_stat = ImageStat.Stat(grayscale_image)
        gray_stddev = gray_stat.stddev[0]
        is_blank = gray_stddev < blank_threshold

        # 2. 纯色判定（针对任意颜色的纯色背景）
        # 改用 RGB 三个通道的标准差，极大提高对噪点、JPEG压缩伪影的容错率
        rgb_image = image.convert('RGB')
        rgb_stat = ImageStat.Stat(rgb_image)
        
        # rgb_stat.stddev 会返回包含 R, G, B 三个通道标准差的列表
        r_stddev, g_stddev, b_stddev = rgb_stat.stddev
        
        # 如果 RGB 三个通道的波动范围都在阈值内，则判定为纯色
        max_channel_stddev = max(r_stddev, g_stddev, b_stddev)
        is_solid = max_channel_stddev < solid_color_threshold

        return (is_blank or is_solid), image
        
    except Exception:
        # 遇到图片损坏等异常时，按原逻辑静默返回 True
        return True, None

async def process_single_item(sem, context, item, pbar):
    """
    处理单条数据的异步工作函数
    """
    record_id = item.get(ID_FIELD)
    result = {
        "item": item,
        "keep": False,
        "status": "unknown"
    }

    if not record_id:
        pbar.update(1)
        result["status"] = "missing_id"
        return result

    # 构造绝对路径
    html_path = os.path.join(HTML_FOLDER, f"{record_id}.html")
    absolute_path = os.path.abspath(html_path)

    if not os.path.exists(absolute_path):
        pbar.update(1)
        result["status"] = "file_not_found"
        return result

    # 限制并发数，获取信号量
    async with sem:
        page = None
        try:
            page = await context.new_page()
            
            # 加载页面，设置超时防止卡死
            # wait_until='domcontentloaded' 比 networkidle 快很多，适合本地文件
            await page.goto(f"file://{absolute_path}", wait_until='domcontentloaded', timeout=TIMEOUT)
            
            # 截图
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # 在单独的线程中进行图片计算，避免阻塞 asyncio 事件循环
            loop = asyncio.get_event_loop()
            is_blank, image_obj = await loop.run_in_executor(None, is_image_blank_or_solid, screenshot_bytes)

            if is_blank:
                # 如果判定为空白，保存截图证据
                evidence_path = os.path.join(BLANK_IMG_FOLDER, f"{record_id}_blank.png")
                if image_obj:
                    # 在线程池中保存图片
                    await loop.run_in_executor(None, image_obj.save, evidence_path)
                
                result["status"] = "blank_detected"
                result["keep"] = False
            else:
                result["status"] = "valid"
                result["keep"] = True

        except Exception as e:
            # print(f"Error processing {record_id}: {e}")
            result["status"] = "render_error"
            result["keep"] = False
        finally:
            if page:
                await page.close()
            pbar.update(1)
            
    return result

async def main():
    # 0. 准备文件夹
    if not os.path.exists(BLANK_IMG_FOLDER):
        os.makedirs(BLANK_IMG_FOLDER)
        print(f"创建文件夹: {BLANK_IMG_FOLDER}")

    # 1. 读取数据
    print(f"正在读取 {INPUT_JSONL} ...")
    lines = []
    if os.path.exists(INPUT_JSONL):
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
    else:
        print("未找到输入文件。")
        return

    total_tasks = len(lines)
    print(f"总任务数: {total_tasks} | 并发数: {CONCURRENCY}")

    # 2. 启动 Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 创建上下文，统一视口大小
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        
        # 信号量用于限制同时打开的页面数量
        sem = asyncio.Semaphore(CONCURRENCY)
        
        tasks = []
        # 初始化进度条
        with tqdm(total=total_tasks, desc="并发渲染检测中", unit="pg") as pbar:
            # 创建所有任务
            for item in lines:
                task = asyncio.create_task(process_single_item(sem, context, item, pbar))
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks)

        await browser.close()

    # 3. 统计与保存
    valid_items = []
    stats = {
        "valid": 0,
        "blank_detected": 0,
        "file_not_found": 0,
        "render_error": 0,
        "missing_id": 0
    }

    for res in results:
        status = res["status"]
        stats[status] = stats.get(status, 0) + 1
        
        if res["keep"]:
            valid_items.append(res["item"])

    # 4. 写入新文件
    print(f"正在写入新文件 {OUTPUT_JSONL} ...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 5. 最终报告
    removed_count = total_tasks - len(valid_items)
    print("\n" + "="*30)
    print(" 处理完成报告")
    print("="*30)
    print(f"原始行数 : {total_tasks}")
    print(f"保留行数 : {len(valid_items)}")
    print(f"删除行数 : {removed_count}")
    print("-" * 20)
    print(f" [删除详情]")
    print(f"   - 空白页面 (已截图保存): {stats['blank_detected']}")
    print(f"   - HTML文件缺失: {stats['file_not_found']}")
    print(f"   - 渲染/超时错误: {stats['render_error']}")
    print(f"   - ID缺失: {stats['missing_id']}")
    print("="*30)
    print(f"空白截图证据已保存在: ./{BLANK_IMG_FOLDER}/")

if __name__ == "__main__":
    asyncio.run(main())
