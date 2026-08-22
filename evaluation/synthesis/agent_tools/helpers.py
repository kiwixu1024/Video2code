
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
from tqdm import tqdm
from multiprocessing import Process, Queue
import openai
from openai import OpenAI
import pathlib
from pathlib import Path
import cv2
import tempfile

import time
from agent_tools import loadConfig as LConfig
import queue


# ------------------------------
# 4. 探索工具
# ------------------------------

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Killing pid={proc.pid}, name={proc.name()}, port={port}")
                    for child in proc.children(recursive=True):
                        print(f"Killing child pid={child.pid}, name={child.name()}")
                        child.kill()
                    proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        except Exception as e:
            print(f"Error checking proc {proc.pid}: {e}")



def extract_boxed_text(text):
    match = re.search(r'\\+boxed\{([^}]*)\}', text)
    if match:
        return match.group(1)
    return ""

def find_node_by_id_for_prompt(domtree, node_id):
    if not domtree:
        return None
    if isinstance(domtree, str):
        try:
            domtree = ast.literal_eval(domtree)
        except Exception:
            print(f"[WARN] domtree字符串转dict失败，domtree:{domtree}")
            return None
            
    if str(domtree.get('id')) == str(node_id):
        # 1. 基础必备信息
        node_info = {
            "id": domtree.get("id"),
            "tag": domtree.get("tag"),
            "attrs": domtree.get("attrs"),
        }
        
        # 2. 动态追加我们在 JS 中提取的高级特征
        # 包含：文本、DOM地址(语义+空间+坐标)、是否可交互、输入框的值、下拉菜单选项
        optional_keys = ["visible_text", "address", "can_interact", "input_value", "options"]
        
        for key in optional_keys:
            # 只有当该字段存在，且不为空（None 或 空字符串/空列表）时才加入
            val = domtree.get(key)
            if val is not None and val != "" and val != []:
                node_info[key] = val
                
        return node_info
        
    for child in domtree.get('children', []):
        res = find_node_by_id_for_prompt(child, node_id)
        if res:
            return res
            
    return None




def clean_text(text):
    if not text:
        return ''
    return ''.join(c if c.isalnum() or c in ('-_') else '_' for c in text.strip())


def wait_server_up(url, timeout=120):
    start_time = time.time()
    while True:
        try:
            res = requests.get(url,  proxies={"http": None, "https": None}, timeout=1)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError("Server not up after {}s".format(timeout))
        time.sleep(0.5)


def get_current_img_b64(port):
    r = requests.get(f"http://localhost:{port}/observe_sized",  proxies={"http": None, "https": None}).json()
    return r["image_b64"]


def domtree_hash(domtree):
    domtree_str = json.dumps(domtree, sort_keys=True)
    return hashlib.md5(domtree_str.encode('utf-8')).hexdigest()


def save_image(img_b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(img_b64))


def deep_diff(a, b):
    return json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


def collect_nodes_by_tag(node, tags):
    res = []
    if "tag" in node and node["tag"].lower() in tags:
        res.append({
            "id": node["id"],
            "tag": node["tag"].lower(),
            "attrs": node.get('attrs', {})
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_tag(child, tags))
    return res


def collect_nodes_by_id(node, ids):
    res = []
    if "id" in node and node["id"] in ids:
        res.append({
            "id": node["id"],
            "attrs": node.get('attrs', {}),
            "visible_text": node.get('visible_text', "")
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_id(child, ids))
    return res


def collect_button_ids(node, button_text_visited=None, no_text_attrs_visited=None):
    if button_text_visited is None:
        button_text_visited = set()
    if no_text_attrs_visited is None:
        no_text_attrs_visited = set()
    res = []
    children = node.get("children", [])
    for child in children:
        if not "tag" in child:
            continue
        tag = child["tag"].lower()
        if tag == "button" or tag == "a" or (tag == "input" and child["attrs"].get("type", "").lower() in ("button", "submit")):
            txt = (child.get("attrs", {}).get("text", None) or child.get("visible_text", None) or "").strip()
            can_interact = child.get("can_interact", True)
            if can_interact:
                if txt:
                    if txt not in button_text_visited:
                        button_text_visited.add(txt)
                        res.append({
                            "id": child["id"], "text": txt, "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
                else:
                    attrs_set = frozenset(child.get("attrs", {}).items())
                    if attrs_set not in no_text_attrs_visited:
                        no_text_attrs_visited.add(attrs_set)
                        res.append({
                            "id": child["id"], "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
        res.extend(collect_button_ids(child, button_text_visited, no_text_attrs_visited))
    return res



def get_node_text(node):
    texts = []
    if "text" in node:
        texts.append(node["text"])
    for c in node.get("children", []):
        texts.append(get_node_text(c))
    return ''.join(texts)


def collect_select_values(node):
    results = []
    def dfs(n):
        if "tag" in n and n["tag"].lower() == "select":
            values = []
            for c in n.get("children", []):
                if c.get("tag", "").lower() == "option":
                    v = c.get("attrs", {}).get("value")
                    if v is None or v == "":
                        v = get_node_text(c)
                    values.append(v)
            results.append({"id": n["id"], "values": values})
        for c in n.get("children", []):
            dfs(c)
    dfs(n=node)
    return results


def check_yes_no(boxed_text):
    if boxed_text == "有":
        return True
    elif boxed_text == "无":
        return False
    else:
        return None


def extract_boxed_actions(text):
    lst = re.findall(r'\\+boxed\{([^}]*)\}', text)
    filtered = []
    for act in lst:
        if not act:
            continue
        filtered.append(act)
    return filtered


def smart_split(s, sep=','):
    res = []
    buf = ''
    level = 0
    for c in s:
        if c == '[':
            level += 1
        elif c == ']':
            level -= 1
        if c == sep and level == 0:
            res.append(buf.strip())
            buf = ''
        else:
            buf += c
    if buf:
        res.append(buf.strip())
    return res


def parse_action_seq(raw_action_seq):
    def strip_quotes(s):
        return re.sub(r'^[\'"""'']+|[\'"""'']+$', '', s)

    def extract_args(s):
        """将 '[arg1][arg2]' 或 '[arg1' 等残缺形式解析为参数列表"""
        args = []
        for m in re.finditer(r'\[([^\[\]]*)\]?', s):
            args.append(strip_quotes(m.group(1).strip()))
        return args

    actions = []
    for act in smart_split(raw_action_seq):
        act = act.strip()
        if not act:
            continue

        # 统一提取动作名和后续参数字符串
        m = re.match(r"(\w+)([\s\S]*)", act, re.I)
        if not m:
            return None

        action_name = m.group(1).lower()
        args_str = m.group(2)
        args = extract_args(args_str)

        if action_name == "click":
            if not args:
                return None
            actions.append({"action": "click", "id": args[0]})

        elif action_name == "enter":
            if len(args) < 2:
                return None
            actions.append({"action": "input", "id": args[0], "input_val": args[1]})

        elif action_name == "select":
            if len(args) < 2:
                return None
            actions.append({"action": "select", "id": args[0], "value": args[1]})

        elif action_name == "scroll":
            if not args:
                return None
            id_raw = args[0]
            id_ = int(id_raw) if id_raw.isdigit() else None
            actions.append({"action": "scroll", "id": id_})

        else:
            return None

    return actions


def set_page_state(state, port, t0: float | None = None, label_for_moments: str | None = None):
    """
    执行 state['actions']，若提供 t0，则记录每一步完成时刻与 t0 的差（秒）。
    返回 (success: bool, moments: list[float])，moments 可能为空。
    label_for_moments 只是占位，方便调用方统一接口（本函数不使用该参数）。
    """
    requests.post(f"http://localhost:{port}/reset", proxies={"http": None, "https": None}).json()
    per_step_moments = []
    ok = True
    shift_index = 1
    for step in state["actions"]:
        action = step["action"]
        dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id", proxies={"http": None, "https": None}).json()
        id2xpath = dt_info["id2xpath"]
        time.sleep(1)
        t1 = time.perf_counter()
        if action == "click":
            resp = requests.post(f"http://localhost:{port}/click",
                                 json={"id": step["id"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "input":
            resp = requests.post(f"http://localhost:{port}/enter",
                                 json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "select":
            resp = requests.post(f"http://localhost:{port}/select",
                                 json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "scroll":
            resp = requests.post(f"http://localhost:{port}/scroll",
                                 json={"id": step.get("id"), "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        else:
            resp = True

        if not resp:
            ok = False

        t_last = time.perf_counter()

        # print("start time: ", t0)
        # print("call action time: ", t1)
        # print("time stamp inside: ", resp.get("time", -1))
        # print("time stamp inside start: ", resp.get("in_time", -1))
        # print("time stamp outside: ", t_last - t1)
        # print("estimated time:" ,resp.get("in_time", -1) - t0 + t_last - t1)
        if t0 is not None:
            per_step_moments.append(max(0.0, t_last - t1 + 0.08))
        shift_index += 1
        time.sleep(1)
    return ok, per_step_moments, shift_index


def set_page_state_for_each_action(state, port, shift_index, t0: float | None = None):
    """
    原返回: img_b64_list, resp, t_last
    新返回: img_b64_list, resp, per_step_moments, t_last
      - per_step_moments: 每一步完成时刻（相对 t0），若 t0=None 则为空
    """
    img_b64_list = []
    resp = []
    per_step_moments = []
    t_last = None
    for step in state["actions"]:
        action = step["action"]
        dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id", proxies={"http": None, "https": None}).json()
        id2xpath = dt_info["id2xpath"]
        time.sleep(1)
        t1 = time.perf_counter()
        if action == "click":
            r = requests.post(f"http://localhost:{port}/click",
                              json={"id": step["id"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "input":
            r = requests.post(f"http://localhost:{port}/enter",
                              json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "select":
            r = requests.post(f"http://localhost:{port}/select",
                              json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        elif action == "scroll":
            r = requests.post(f"http://localhost:{port}/scroll",
                              json={"id": step.get("id"), "id2xpath": id2xpath}, proxies={"http": None, "https": None}).json()
        else:
            r = True
        resp.append(r)
        t_last = time.perf_counter()
        img_b64 = get_current_img_b64(port)
        img_b64_list.append({"action": step, "img_b64": img_b64})
        # print("call action time: ", t1)
        # print("time stamp request start: ", r.get("time", -1))
        # print("time stamp inside start: ", r.get("in_time", -1))
        # print("time stamp outside: ", t_last - t1)
        # print("estimated time:" ,r.get("in_time", -1) - t0 + t_last - t1)
        if t0 is not None:
            per_step_moments.append(max(0.0, t_last - t1 + 0.08))
        shift_index += 1

    return img_b64_list, resp, per_step_moments, t_last
