#修改了get_dom tree的元素数据，方便构建RLtask
#修改了dom_info保存逻辑
#HTML 最多探索1层

import os
import argparse
import json
import requests
import base64
import time
import subprocess
import psutil
import hashlib
import sys
import re
import copy
import ast
import shutil
import signal
from tqdm import tqdm
from multiprocessing import Process, Queue
import openai
from openai import OpenAI
import pathlib
from pathlib import Path
import cv2
import tempfile

import time

from data.data_construct.agent_tools import video_edit_tools as video_tools
from data.data_construct.agent_tools import config as LConfig
from data.data_construct.agent_tools import helpers as helpers
import queue

TASK_TIMEOUT_SECS = 1800  # 单任务最长 30 分钟

def _task_timeout_handler(signum, frame):
    raise TimeoutError(f"Task exceeded maximum allowed time ({TASK_TIMEOUT_SECS}s)")

# ------------------------------
# 1. 加载配置：支持命令行与配置文件
# ------------------------------

# ------------------------------
# 2. 初始化配置与全局变量
# ------------------------------
config = LConfig.load_config()
config = LConfig.validate_config(config)  # <<< 新增：严格校验（若失败会抛出并退出）

INIT_PATH = config["webenv_py"]           # 修正：使用校验阶段写入的键名
IMAGE_DIR = config["image_dir"]
INPUT_DIR = config.get("input_dir")
BUG_DIR   = config.get("bug_dir")
INPUT_TYPE = (config.get("input_type") or "html").lower()
URL_FILE  = config.get("url_file")

openai.api_base = config["api_base"]
openai.api_key  = config["api_key"]
MODEL_NAME = config["model_name"]

client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

NUM_PORT  = config["num_port"]
PORT_BASE = config["port_base"]
TIME_SHIFT = 0.2

HTML_MAX_DEPTH = 2
URL_MAX_DEPTH = 2


# ---- 安全命名工具 ----
MAX_FILENAME_BYTES = 255          # 常见文件系统单个文件名上限
MAX_FULLPATH_BYTES = 4096         # 进程/OS 对全路径一般极限（保守）



import hashlib
import re

def get_safe_url_name(url: str) -> str:
    """
    生成绝对唯一且安全的文件名：保留前30个字符保证可读性，拼接8位MD5哈希保证唯一性。
    """
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
    safe_prefix = re.sub(r'\W+', '_', url)[:30]
    return f"{safe_prefix}_{url_hash}"


def get_response(message):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=message,
        max_completion_tokens=10000,
        timeout=600
    )
    return response


def cp_bad_html(src_file):
    # 仅 HTML 模式使用
    try:
        basename = os.path.basename(src_file)
        copypath = os.path.join(BUG_DIR, basename)
        shutil.copy(src_file, copypath)
    except Exception:
        pass


def _utf8_len(s: str) -> int:
    return len(s.encode('utf-8'))

def build_seq_names_or_fallback(state_seq_name: str,
                                step_labels: list[str],
                                out_dir: str,
                                layer_idx: int) -> tuple[list[str], bool]:
    """
    给定 state 的已有前缀（state_seq_name）、本轮步骤标签（step_labels）与输出目录（out_dir），
    先尝试“逐步拼接”的原始命名；若任何一个文件名或全路径会超长，则整体回退为：
        {state_seq_name}_action-seq-layer{layer_idx}-{k}
    返回：(names, used_fallback)
        - names: 每一步的最终 seq_name 列表（不含扩展名）
        - used_fallback: 是否触发了统一回退
    """
    # 1) 先尝试原始增量命名
    orig_names = []
    for i in range(len(step_labels)):
        parts = []
        if state_seq_name:
            parts.append(state_seq_name)
        parts += step_labels[:i+1]
        seq_name = "_".join(p for p in parts if p)
        filename = f"{seq_name}.png"
        fullpath = os.path.join(out_dir, filename)

        if _utf8_len(os.path.basename(filename)) > MAX_FILENAME_BYTES or _utf8_len(fullpath) > MAX_FULLPATH_BYTES:
            # 任何一步超限则整段回退
            break
        orig_names.append(seq_name)

    if len(orig_names) == len(step_labels):
        return orig_names, False

    # 2) 统一回退命名方案
    fb_names = []
    for k in range(1, len(step_labels) + 1):
        fb = f"action-seq-layer{layer_idx}-{k}"
        seq_name = "_".join([p for p in [state_seq_name, fb] if p])
        fb_names.append(seq_name)

    return fb_names, True



def _save_video_safe(vid_list, output_dir, file_base_name):
    """
    合并并保存 filtered 视频。

    发布流程只保留 filtered 版本，并直接使用 ``<id>.mp4`` 命名，使输出可由
    ``01_build_timeline_dataset.py`` 直接读取。返回值仍保持两元素结构，以兼容
    ``_dump_partial_result_common`` 对 filtered timeline 的读取方式。
    """
    if not vid_list:
        return

    try:
        print(f"[INFO] 正在尝试保存视频 (片段数: {len(vid_list)})...")
        payload = {
            "result": True,
            "count": len(vid_list),
            "videos": vid_list
        }
        video_b64, mime, detailed_timeline = video_tools.merge_video_longest(payload)

        out_path = os.path.join(output_dir, "videos", f"{file_base_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        print(f"[INFO] Filtered 视频保存成功: {os.path.abspath(out_path)} ({mime})")
        return [None, detailed_timeline]
    except Exception as e:
        print(f"[WARN] 保存视频失败: {e}")
        


# ------------------------------
# 5. 请求模型
# ------------------------------



def get_compare_response(img_b64_list, interact_name, compare_messages):
    prompt_base = (
        "你会收到多张网页图片作为互动历史的验证过程，"
        "多张网页图片呈正序排列，最后一张图片代表互动完成，第一张图片是起始图，每进行一次操作会截一次图，直到最后一张完成操作。比如如果只有两张图，就是只进行了一个操作，操作前截图是第一张，操作后截图是第二张。如果有四张图，就是进行了三次操作，操作前截图是第一张，第一次操作后是第二张，第二次操作是第三章，以此类推。"
        "请完成以下两个内容：\n"
        "1. 判断本次互动序列后网页是否产生了符合互动组件的变化，比如点击编辑会弹出编辑窗口，在编辑窗口输入一系列内容后点击“保存”会讲刚才输入的内容完整地保存到页面上，点击“取消”或者“关闭”后不会保存。我会给你这次点击的互动组件序列名字叫什么，有哪些内容。请认真观察网页，理解网页进行这一轮操作之后应该会出现哪些变化，重点判断在最后一次操作之后图片是否发生了这一系列动作应该有的变化。没有发生的变化比如在点击一个按钮后只有按钮本身高亮了，网页本身无变化。或者点击保存后退出了弹窗，但网页中没有真的保存。给出一个判断，回复我“无”表示两张图片大体上无变化，“有”表示发生了变化。请注意。将你给出的答案提取出来放在latex的\\boxed{}中，并且给出你判断的理由"
        "2. 请你仔细比较本次序列生成的最后一张图片，判断网页与本次的起始图相比是否出现了新的重要部分。如果新部分出现，需要继续检测（即页面与所有历史图片都不极度相似），请用\\terminate{}包裹“继续”；"
        "请注意！如果没有新变化（如页面整体无明显新部分/出现内容与之前某张历史图极度相似/页面中出现的新部分极其微小或互动内容过少不值得继续/页面中出现的新内容与原图中内容高度重复），例如在页面中删除了一个栏目，没有任何新的互动组件出现。请用\\terminate{}包裹“完成”。请你务必只用terminate{...}包裹继续或完成。\n"
        "请详细说明你的理由（boxed和terminate都要输出，理由附在后面）。\n"
        "互动动作/组件名: " + str(interact_name)
    )
    image_content = [{"type": "text", "text": prompt_base}]
    for idx, img in enumerate(img_b64_list):
        image_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpg;base64," + img},
        })
    try:
        response = get_response([{"role": "user", "content": image_content}])
        output = str(response.choices[0].message)
        return output, compare_messages
    except Exception as ex:
        print(f"Exception: {ex}")
        return "", compare_messages

def get_op_actions_by_model_no_history(img_b64, domtree, messages, messages_to_get_history_clicked, domtree_history):
    """
    用模型生成可操作动作序列，并在prompt里包含历史已选动作的dom元素信息,
    domtree_history: 每条历史消息唯一id映射的domtree
    """
    history_actions = []
    for idx, msg in enumerate(messages_to_get_history_clicked):
        if msg.get('role') == 'assistant':
            msg_id = msg.get('msg_id', idx)
            boxed_raws = helpers.extract_boxed_actions(msg.get('content', ''))
            actions_per_seq = [r for r in (helpers.parse_action_seq(bx) for bx in boxed_raws) if r is not None]
            for actlist in actions_per_seq:
                for act in actlist:
                    history_actions.append({'act': act, 'msg_id': msg_id})

    history_elements_desc = []
    for item in history_actions:
        act = item['act']
        domtree_for_search = domtree_history.get(item['msg_id'])
        node = helpers.find_node_by_id_for_prompt(domtree_for_search, act['id'])
        desc = str(node)
        if desc.strip():
            history_elements_desc.append(desc)
    history_info_prompt = ""
    if history_elements_desc:
        history_info_prompt = "\n".join(history_elements_desc) + "\n"

    prompt_text = f"""你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，我会发送给你现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。 【历史点击过的dom element】 {history_info_prompt}
        【注意】：请只返回答案！不要有多余的东西！
        页面信息:\n{domtree}\n,"""
    image_content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": "data:image/jpg;base64," + img_b64}}
    ]
    for ex in messages:
        if ex["role"] == "user":
            ex["content"][0]["text"] = f"""你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，请检查我与你的互动历史来判断现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。"""
    print_messages_for_print = []
    for ex in messages:
        new_ex = copy.deepcopy(ex)
        if new_ex["role"] == "user":
            del new_ex["content"][1]
        print_messages_for_print.append(new_ex)

    msg_id = len(messages)
    messages.append({"role": "user", "content": image_content})
    messages_to_get_history_clicked.append({"role": "user", "content": image_content, "msg_id": msg_id})
    response= get_response(messages)
    output = str(response.choices[0].message)
    messages.append({"role": "assistant", "content": output})
    messages_to_get_history_clicked.append({"role": "assistant", "content": output, "msg_id": msg_id})
    domtree_history[msg_id] = copy.deepcopy(domtree)
    boxed_raws = helpers.extract_boxed_actions(output)
    actions_per_seq = [r for r in (helpers.parse_action_seq(bx) for bx in boxed_raws) if r is not None]
    return actions_per_seq, output




# ------------------------------
# 6. 输出json (保持原样)
# ------------------------------

def _dump_partial_result_common(step_records, out_json_path, detailed_timeline, meta_fields: dict):
    try:
        compare_records = [r for r in step_records if r.get("step_type") == "compare"]
        total_trials = len(compare_records)
        total_compare_records = [
            ",".join([a.get('action', '') + str(a.get('id', '')) for a in r.get("actions_seq", [])])
            for r in compare_records
        ]
        pass_records = [r for r in compare_records if r.get("pass")]
        pass_num = len(pass_records)
        pass_texts = [
            ",".join([a.get('action', '') + str(a.get('id', '')) for a in r.get("actions_seq", [])])
            for r in pass_records
        ]
        pass_rate = float(pass_num) / total_trials if total_trials > 0 else 0.0

        def node_signature(node):
            if node is None:
                return "none"
            return f"{node.get('id','')}_{node.get('tag','')}_{json.dumps(node.get('attrs',{}),ensure_ascii=False,sort_keys=True)}_{node.get('visible_text','')}"

        all_action_dom_elements = []
        pass_action_dom_elements = []
        dom_sig_set_all = set()
        dom_sig_set_pass = set()

        for rec in compare_records:
            for a in rec.get("actions_seq", []):
                dom_info = a.get("dom_info")
                sig = node_signature(dom_info)
                all_action_dom_elements.append(dom_info)
                dom_sig_set_all.add(sig)
            if rec.get("pass"):
                for a in rec.get("actions_seq", []):
                    dom_info = a.get("dom_info")
                    sig = node_signature(dom_info)
                    pass_action_dom_elements.append(dom_info)
                    dom_sig_set_pass.add(sig)
        if detailed_timeline:
            result_json = {
                **meta_fields,
                "steps": step_records,
                "total_compare_steps": total_trials,
                "total_compare_records": total_compare_records,
                "pass_num": pass_num,
                "pass_texts": pass_texts,
                "pass_rate": pass_rate,
                "all_action_dom_elements": all_action_dom_elements,
                "pass_action_dom_elements": pass_action_dom_elements,
                "action_frames_timeline_all": detailed_timeline[0],
                "action_frames_timeline_filtered": detailed_timeline[1]
            }
        else:
            result_json = {
                **meta_fields,
                "steps": step_records,
                "total_compare_steps": total_trials,
                "total_compare_records": total_compare_records,
                "pass_num": pass_num,
                "pass_texts": pass_texts,
                "pass_rate": pass_rate,
                "all_action_dom_elements": all_action_dom_elements,
                "pass_action_dom_elements": pass_action_dom_elements,
            }

        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已保存当前进度到 {out_json_path}")
    except Exception as ee:
        print(f"[WARN] 写入部分结果失败: {ee}")


# ------------------------------
# [新增] 7. 日志辅助函数
# ------------------------------
def write_summary_log(all_results, output_dir):
    """
    根据运行结果列表生成 summary_log.txt
    """
    success_list = [r for r in all_results if r['type'] == 'success']
    fail_list = [r for r in all_results if r['type'] == 'fail']
    skip_list = [r for r in all_results if r['type'] == 'skip']

    log_path = os.path.join(output_dir, "summary_log.txt")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    content = []
    content.append(f"====== 执行报告 ({timestamp}) ======")
    content.append(f"总计任务: {len(all_results)}")
    content.append(f"成功完成: {len(success_list)}")
    content.append(f"失败数量: {len(fail_list)}")
    content.append(f"跳过数量: {len(skip_list)}")
    content.append("======================================\n")

    if fail_list:
        content.append(">>> 失败详情列表:")
        for idx, item in enumerate(fail_list, 1):
            content.append(f"{idx}. [{item['target']}]")
            content.append(f"   原因: {item['reason']}")
            content.append("")
    else:
        content.append("无失败任务。\n")
        
    content.append(">>> 成功任务列表 (前20个):")
    for item in success_list[:20]:
        content.append(f" - {item['target']}")
    if len(success_list) > 20:
        content.append(f" ... (以及其他 {len(success_list)-20} 个)")

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"\n[INFO] 总结日志已生成: {log_path}")
        print(f"[INFO] 成功: {len(success_list)}, 失败: {len(fail_list)}, 跳过: {len(skip_list)}")
    except Exception as e:
        print(f"[ERROR] 无法写入总结日志: {e}")


def _run_exploration_core(target, is_url, port, target_name_override=None):
    """
    核心探索器：统一处理 HTML 文件和 URL 探索的冗余逻辑。
    加入了强制超时保护机制和僵尸进程清理逻辑。
    """
    step_records = []
    vid_list = []
    p = None
    fail_reason = None

    # 1. 差异化参数初始化
    if is_url:
        target_name = target_name_override if target_name_override else get_safe_url_name(target)
        max_depth = URL_MAX_DEPTH
    else:
        target_name = os.path.splitext(os.path.basename(target))[0]
        max_depth = HTML_MAX_DEPTH

    try:
        out_dir = f"{IMAGE_DIR}/results/{target_name}"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(BUG_DIR, exist_ok=True)
        helpers.kill_process_on_port(port)

        # 启动环境
        p = subprocess.Popen([sys.executable, INIT_PATH, target, '--port', str(port)])
        helpers.wait_server_up(f"http://localhost:{port}/dom_tree_with_id")

        # URL 特有的加载缓冲逻辑
        if is_url:
            try:
                # 增加 60s 超时，防止遇到毒药网页永久挂起
                requests.post(f"http://localhost:{port}/load_url", json={'url': target}, proxies={"http": None, "https": None}, timeout=60)
            except Exception:
                pass
            time.sleep(2)

        # 获取初始 DOM 和截图 (增加 60s 超时)
        dt = requests.get(f"http://localhost:{port}/dom_tree_with_id", proxies={"http": None, "https": None}, timeout=60).json()
        domtree = dt["domtree"]
        id2xpath = dt["id2xpath"]
        initial_hash = helpers.domtree_hash(domtree)
        
        all_layers = []
        visited_hashes = {initial_hash}
        init_img = helpers.get_current_img_b64(port)
        
        layer0 = [{
            "name": "start",
            "seq_name": "",
            "text": "start",
            "domtree": domtree,
            "domtree_hash": initial_hash,
            "id2xpath": id2xpath,
            "actions": [],
            "img_b64": init_img,
            "messages_chain": [],
            "state_seq": [0], # 统一记录状态追踪序列
        }]
        all_layers.append(layer0)
        helpers.save_image(init_img, f"{out_dir}/start.png")

        # ================= 录制初始全页滚动视频 =================
        try:
            check_scroll = requests.post(f"http://localhost:{port}/check_if_scroll_needed", json={}, proxies={"http": None, "https": None}, timeout=60).json()
            if check_scroll:
                print("正在录制初始化滚动视频...")
                r = requests.post(f"http://localhost:{port}/start_video", json={}, proxies={"http": None, "https": None}, timeout=60)
                pass_flag = False
                if r.status_code == 200 and r.json().get("result"):
                    requests.post(f"http://localhost:{port}/scroll_page", proxies={"http": None, "https": None}, timeout=60)
                    pass_flag = True
                
                scroll_vid = requests.post(f"http://localhost:{port}/end_video", proxies={"http": None, "https": None}, timeout=60).json()
                scroll_vid.update({
                    "name": "init_scroll",
                    "pass_flag": pass_flag,
                    "is_init_scroll": True,
                    "old_moments": [],
                    "new_moments": [],
                    "actions_detail": [None] if is_url else []
                })
                vid_list.append(scroll_vid)
        except Exception as e:
            print(f"[WARN] Init scroll video failed: {e}")
        # ==============================================================

        messages = []
        messages_to_get_history_clicked = []
        domtree_history = {0: copy.deepcopy(domtree)}

        # 主探索循环
        for depth in range(max_depth):
            layer = all_layers[depth]
            next_layer = []

            for idx, s in enumerate(layer):
                if "state_seq" not in s: 
                    s["state_seq"] = [idx]
                print(f"发现state {s['name']} in layer {depth}, index {idx}, state_seq={s['state_seq']}")

            for idx, state in enumerate(layer):
                helpers.set_page_state(state, port)
                dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id", proxies={"http": None, "https": None}, timeout=60).json()
                domtree = dt_info["domtree"]
                id2xpath = dt_info["id2xpath"]
                
                can_operate = bool(
                    helpers.collect_button_ids(domtree) or 
                    helpers.collect_nodes_by_tag(domtree, ["input", "textarea"]) or 
                    helpers.collect_select_values(domtree)
                )

                if can_operate:
                    print(f"本次请求为第{depth}层请求，原图为{state['name']}")
                    actions_seqs, model_raw_output = get_op_actions_by_model_no_history(
                        state["img_b64"], domtree, messages, messages_to_get_history_clicked, domtree_history
                    )

                    detailed_action_seqs = []
                    for actlist in actions_seqs:
                        detailed_actlist = []
                        for a in actlist:
                            a2 = dict(a)
                            a2['dom_info'] = helpers.find_node_by_id_for_prompt(domtree, a.get("id"))
                            detailed_actlist.append(a2)
                        detailed_action_seqs.append(detailed_actlist)

                    step_records.append({
                        "step_type": "list",
                        "img_name": state['name'],
                        "domtree": str(domtree),
                        "actions_seqs": [dict(a) for actlist in detailed_action_seqs for a in actlist],
                        "actions_seqs_detail": detailed_action_seqs,
                        "model_output": model_raw_output,
                        "step_idx": len(step_records),
                    })

                    for seq_idx, actions in enumerate(actions_seqs):
                        elements = helpers.collect_nodes_by_id(domtree, [a["id"] for a in actions])
                        temp_state = {"actions": state["actions"]}

                        # 视频录制启动与重试，这里 timeout 设置得稍微短一点方便快速重试
                        for _ in range(10):
                            try:
                                r = requests.post(f"http://localhost:{port}/start_video", json={}, proxies={"http": None, "https": None}, timeout=10)
                                if r.status_code == 200 and r.json().get("result"):
                                    break
                            except Exception:
                                time.sleep(0.5)
                        else:
                            raise RuntimeError("start_video failed after retries")

                        t0 = r.json().get("t0")

                        # 回放前置状态与执行新操作
                        ok_old, old_op_moments, shift_index = helpers.set_page_state(temp_state, port, t0=t0, label_for_moments="old")
                        new_action_img_list, click_t_or_f, new_op_moments, t_last = helpers.set_page_state_for_each_action(
                            {"actions": actions}, port, shift_index, t0=t0
                        )

                        if not is_url:
                            for res in click_t_or_f:
                                if isinstance(res, dict) and not res.get("result", True):
                                    cp_bad_html(target)

                        time.sleep(4)
                        this_vid = requests.post(f"http://localhost:{port}/end_video", proxies={"http": None, "https": None}, timeout=60).json()
                        this_vid["old_moments"] = [round(x, 3) for x in (old_op_moments or [])]
                        this_vid["new_moments"] = [round(x, 3) for x in (new_op_moments or [])]
                        this_vid["state_seq"] = state.get("state_seq", [idx])

                        actions_detail = []
                        for a in actions:
                            a2 = dict(a)
                            a2["dom_info"] = helpers.find_node_by_id_for_prompt(domtree, a.get("id"))
                            actions_detail.append(a2)

                        combined_actions = state["actions"] + actions_detail
                        vid_actions_detail = list(combined_actions)

                        print("正在准备互动检验请求：", state["name"])
                        step_labels, extracted_labels, step_imgs_b64 = [], [], []

                        for step_action in new_action_img_list:
                            a = step_action["action"]
                            t = ""
                            if a["action"] == "input": t = f"{a['action']}_{a['id']}_{a['input_val']}"
                            elif a["action"] == "select": t = f"{a['action']}_{a['id']}_{a['value']}"
                            elif a["action"] == "click":
                                node = helpers.find_node_by_id_for_prompt(domtree, a["id"])
                                if node:
                                    t = node.get("visible_text") or node.get("attrs", {}).get("text") or \
                                        node.get("attrs", {}).get("value") or node.get("attrs", {}).get("placeholder") or \
                                        a.get("input_val") or a.get("value") or ""
                            
                            label = helpers.clean_text(t) if t else f"{a['action']}_{a['id']}"
                            step_labels.append(label)
                            extracted_labels.append(label)
                            step_imgs_b64.append(step_action["img_b64"])

                        final_names, used_fallback = build_seq_names_or_fallback(
                            state.get("seq_name", ""), step_labels, out_dir, depth
                        )
                        extracted_name = final_names[-1] if used_fallback else ("_".join(extracted_labels) if extracted_labels else "some_where_in_the_picture")
                        next_seq_chain = final_names[-1] if used_fallback else "_".join([p for p in [state.get("seq_name","")] + step_labels if p])

                        img_name_list, img_b64_list = [], [state["img_b64"]]
                        last_image_path = ""
                        for seq_name, img_b64_one in zip(final_names, step_imgs_b64):
                            img_path = os.path.join(out_dir, f"{seq_name}.png")
                            helpers.save_image(img_b64_one, img_path)
                            img_name_list.append(img_path)
                            img_b64_list.append(img_b64_one)
                            last_image_path = img_path

                        output, new_messages_chain = get_compare_response(img_b64_list, extracted_name, state.get("messages_chain", []))
                        terminate_matches = re.findall(r'terminate\{(.*?)\}', output)
                        terminate_flag = (terminate_matches[-1].strip() == "继续") if terminate_matches else False
                        pass_flag = helpers.check_yes_no(helpers.extract_boxed_text(output)) is True
                        imgB = helpers.get_current_img_b64(port)

                        step_records.append({
                            "step_type": "compare",
                            "img_name_before": state['name'],
                            "img_name_list_after": img_name_list,
                            "domtree": str(domtree),
                            "actions_seq": actions_detail,
                            "component_info": elements,
                            "model_output": model_raw_output,
                            "detect_output": output,
                            "pass": pass_flag,
                            "step_idx": len(step_records),
                        })

                        if pass_flag:
                            dt2 = requests.get(f"http://localhost:{port}/dom_tree_with_id", proxies={"http": None, "https": None}, timeout=60).json()
                            next_domtree, next_hash = dt2["domtree"], helpers.domtree_hash(dt2["domtree"])
                            if terminate_flag:
                                child_index = len(next_layer)
                                state_item = {
                                    "name": last_image_path,
                                    "seq_name": next_seq_chain,
                                    "domtree": next_domtree,
                                    "domtree_hash": next_hash,
                                    "id2xpath": id2xpath,
                                    "actions": combined_actions,
                                    "img_b64": imgB,
                                    "messages_chain": new_messages_chain,
                                    "state_seq": state["state_seq"] + [child_index],
                                }
                                next_layer.append(state_item)
                                visited_hashes.add(next_hash)
                                this_vid["state_seq"] = state_item["state_seq"]
                            else:
                                print(f"动作成功完成，页面没有新内容弹出: {last_image_path}，跳过计入下一轮")
                        else:
                            print(f"动作没有显著变化: {last_image_path}，跳过保存。")

                        this_vid.update({
                            "name": next_seq_chain,
                            "pass_flag": bool(pass_flag),
                            "actions_detail": vid_actions_detail
                        })
                        vid_list.append(this_vid)

            if not next_layer:
                break
            all_layers.append(next_layer)

    except Exception as e:
        fail_reason = f"{type(e).__name__}: {str(e)}"
        print(f"Exception in explore {'url' if is_url else 'html'}({target}): {e}")
        if not is_url:
            cp_bad_html(target)

    finally:
        out_json_path = os.path.join(f"{IMAGE_DIR}/results/", f"{target_name}.json")
        detailed_timeline = _save_video_safe(vid_list, IMAGE_DIR, target_name)
        
        meta_info = {"html_file": target}
        _dump_partial_result_common(step_records, out_json_path, detailed_timeline, meta_info)

        # 彻底清理僵尸进程，防止系统资源耗尽
        if p is not None:
            try: 
                p.terminate()
                p.wait(timeout=5)  # 给予5秒优雅退出时间
            except subprocess.TimeoutExpired:
                print(f"[WARN] 进程 {p.pid} 未响应 terminate，强制 kill")
                p.kill()           # 强制物理击杀
            except Exception as e: 
                print(f"[WARN] 结束进程 {p.pid} 时发生异常: {e}")
        helpers.kill_process_on_port(port)

    return fail_reason
# ------------------------------
# 8. 主功能函数 (使用 Core 封装)
# ------------------------------
def explore_html(html_file, port):
    """HTML 本地文件探索入口"""
    return _run_exploration_core(target=html_file, is_url=False, port=port)

def explore_url(url, port, target_name_override=None):
    """URL 在线网页探索入口"""
    return _run_exploration_core(target=url, is_url=True, port=port, target_name_override=target_name_override)

# ---------- 调度 ----------
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]



# ------------------------------
# 修改后的 Worker 函数 (动态获取任务)
# ------------------------------
def worker_html(task_q, port, progress_q):
    signal.signal(signal.SIGALRM, _task_timeout_handler)
    while True:
        try:
            html = task_q.get_nowait()
        except queue.Empty:
            break

        try:
            html_name = os.path.basename(html)
            json_log_path = os.path.join(f"{IMAGE_DIR}/results", f"{os.path.splitext(html_name)[0]}.json")
            if os.path.exists(json_log_path):
                print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {html}")
                progress_q.put({"type": "skip", "target": html, "reason": "Log exists"})
                continue

            signal.alarm(TASK_TIMEOUT_SECS)
            try:
                error_msg = explore_html(html, port)
            finally:
                signal.alarm(0)

            if error_msg is None:
                progress_q.put({"type": "success", "target": html, "reason": "Success"})
            else:
                progress_q.put({"type": "fail", "target": html, "reason": error_msg})
        except Exception as e:
            helpers.kill_process_on_port(port)
            print(f"[ERROR] worker_html 发生未捕获异常: {e}")
            progress_q.put({"type": "fail", "target": html, "reason": f"Worker exception: {e}"})


def worker_url(task_q, port, progress_q):
    signal.signal(signal.SIGALRM, _task_timeout_handler)
    while True:
        try:
            url = task_q.get_nowait()
        except queue.Empty:
            break

        try:
            url_name = get_safe_url_name(url)
            json_log_path = os.path.join(f"{IMAGE_DIR}/results", f"{url_name}.json")

            if os.path.exists(json_log_path):
                print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {url}")
                progress_q.put({"type": "skip", "target": url, "reason": "Log exists"})
                continue

            signal.alarm(TASK_TIMEOUT_SECS)
            try:
                error_msg = explore_url(url, port, target_name_override=url_name)
            finally:
                signal.alarm(0)

            if error_msg is None:
                progress_q.put({"type": "success", "target": url, "reason": "Success"})
            else:
                progress_q.put({"type": "fail", "target": url, "reason": error_msg})
        except Exception as e:
            helpers.kill_process_on_port(port)
            print(f"[ERROR] worker_url 发生未捕获异常: {e}")
            progress_q.put({"type": "fail", "target": url, "reason": f"Worker exception: {e}"})

# ------------------------------
# 修改后的 Main 调度逻辑
# ------------------------------
if __name__ == "__main__":
    try:
        os.makedirs(IMAGE_DIR + "/results", exist_ok=True)
        os.makedirs(IMAGE_DIR + "/videos", exist_ok=True)
        
        all_results = []
        
        if INPUT_TYPE == "html":
            html_files = [
                os.path.join(INPUT_DIR, fn)
                for fn in os.listdir(INPUT_DIR)
                if fn.endswith(".html")
            ]
            if not html_files:
                raise LConfig.ConfigError(f"[INPUT_DIR] 无 .html 文件：{INPUT_DIR}")

            # 1. 创建任务队列并灌入所有任务
            task_q = Queue()
            for html in html_files:
                task_q.put(html)

            effective_workers = max(1, min(NUM_PORT, len(html_files)))
            ports  = [PORT_BASE + i for i in range(effective_workers)]

            progress_q = Queue()
            total = len(html_files)
            pbar = tqdm(total=total, desc="全部HTML", ncols=80)
            procs = []
            
            # 2. 启动 Worker 进程，传入共享的任务队列
            for port in ports:
                p = Process(target=worker_html, args=(task_q, port, progress_q))
                p.start()
                procs.append(p)
                time.sleep(1) # 缩短等待时间

            finished = 0
            # 带超时机制与死进程检测的 while 循环保持不变
            while finished < total:
                try:
                    res = progress_q.get(timeout=5.0)
                    all_results.append(res)
                    finished += 1
                    pbar.update(1)
                except queue.Empty:
                    active_workers = [p for p in procs if p.is_alive()]
                    if not active_workers and finished < total:
                        print(f"\n[ERROR] 致命异常：所有子进程均已死亡，但仍有 {total - finished} 个任务未返回结果！强行终止等待。")
                        break
                    continue

            for p in procs:
                p.join()
            pbar.close()


        else:  # url 模式
            url_list = []
            bad_lines = 0

            with open(URL_FILE, "r", encoding="utf-8") as f:
                for ln_no, ln in enumerate(f, 1):
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue

                    url = obj.get("url", "")
                    if isinstance(url, str):
                        url = url.strip()
                    else:
                        url = ""

                    if url:
                        url_list.append(url)

            # 去重
            seen = set()
            url_list = [u for u in url_list if not (u in seen or seen.add(u))]

            if not url_list:
                raise LConfig.ConfigError(f"[URL_FILE] 未读取到任何有效 url：{URL_FILE} (bad_lines={bad_lines})")

            # 1. 创建任务队列并灌入所有任务
            task_q = Queue()
            for url in url_list:
                task_q.put(url)

            effective_workers = max(1, min(NUM_PORT, len(url_list)))
            ports  = [PORT_BASE + i for i in range(effective_workers)]

            progress_q = Queue()
            total = len(url_list)
            pbar = tqdm(total=total, desc="全部URL", ncols=80)
            procs = []
            
            # 2. 启动 Worker 进程
            for port in ports:
                p = Process(target=worker_url, args=(task_q, port, progress_q))
                p.start()
                procs.append(p)
                time.sleep(1)

            finished = 0
            while finished < total:
                try:
                    res = progress_q.get(timeout=5.0)
                    all_results.append(res)
                    finished += 1
                    pbar.update(1)
                except queue.Empty:
                    active_workers = [p for p in procs if p.is_alive()]
                    if not active_workers and finished < total:
                        print(f"\n[ERROR] 致命异常：所有子进程均已死亡，但仍有 {total - finished} 个任务未返回结果！强行终止等待。")
                        break
                    continue

            for p in procs:
                p.join()
            pbar.close()
            
        # 最终写入统计日志
        write_summary_log(all_results, IMAGE_DIR)

    except LConfig.ConfigError as ce:
        print(str(ce), file=sys.stderr)
        sys.exit(2)
