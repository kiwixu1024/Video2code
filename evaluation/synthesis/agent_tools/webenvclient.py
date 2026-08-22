import os
import sys
import json
import copy
import time
import re
import base64
import argparse
import subprocess
import requests as http_requests  # 避免与 openai 冲突

from agent_tools import helpers
PROXIES = {"http": None, "https": None}
REQUEST_TIMEOUT = 120  # 秒


class WebEnvClient:
    """
    通过 HTTP 与 webenv.py FastAPI 服务通信的客户端。
    严格对齐 webenv.py 中定义的 API 端点和请求/响应格式。
    """

    def __init__(self, port: int, webenv_py: str):
        self.port = port
        self.webenv_py = webenv_py
        self.base_url = f"http://localhost:{port}"
        self.process = None
        # 缓存最近一次获取的 id2xpath，用于 click/enter/select
        self._cached_id2xpath = {}

    # ---------- 生命周期 ----------

    def start(self, target: str, is_url: bool = False):
        """启动 webenv.py 子进程并等待就绪。"""
        helpers.kill_process_on_port(self.port)

        # webenv.py 接受 html_or_url 作为第一个位置参数，--port 作为端口
        cmd = [sys.executable, self.webenv_py, target, "--port", str(self.port)]
        print(f"[WebEnv] 启动命令: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

        # 等待服务就绪（轮询 /dom_tree_with_id 端点）
        helpers.wait_server_up(f"{self.base_url}/dom_tree_with_id")
        print(f"[WebEnv] 服务已就绪 @ port {self.port}")

        # URL 模式需要额外加载
        if is_url:
            try:
                http_requests.post(
                    f"{self.base_url}/load_url",
                    json={"url": target},
                    proxies=PROXIES,
                    timeout=60,
                )
                print(f"[WebEnv] 已加载 URL: {target}")
            except Exception as e:
                print(f"[WARN] load_url 请求异常: {e}")
            time.sleep(2)

    def shutdown(self):
        """关闭 webenv 子进程并清理端口。"""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[WARN] 进程 {self.process.pid} 未响应 terminate，强制 kill")
                self.process.kill()
            except Exception as e:
                print(f"[WARN] 关闭进程异常: {e}")
        helpers.kill_process_on_port(self.port)
        print("[WebEnv] 已关闭")

    # ---------- DOM & 截图 ----------

    def get_dom_info(self) -> tuple:
        """
        GET /dom_tree_with_id
        返回: (domtree, id2xpath)
        domtree 结构包含: id, tag, attrs, children, can_interact, visible_text,
                         address ("zone--[x,y,w,h]"), input_value, options 等
        """
        resp = http_requests.get(
            f"{self.base_url}/dom_tree_with_id",
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()
        domtree = data["domtree"]
        id2xpath = data["id2xpath"]
        self._cached_id2xpath = id2xpath  # 缓存
        return domtree, id2xpath

    def get_screenshot_b64(self) -> str:
        """
        GET /observe_sized → 视口截图 (base64)
        """
        resp = http_requests.get(
            f"{self.base_url}/observe_sized",
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()["image_b64"]

    def get_full_screenshot_b64(self) -> str:
        """
        GET /observe → 全页截图 (base64)
        """
        resp = http_requests.get(
            f"{self.base_url}/observe",
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()["image_b64"]

    # ---------- 操作执行 ----------

    def click(self, element_id: int, id2xpath: dict = None) -> dict:
        """
        POST /click
        请求体: ClickRequest {"id": int, "id2xpath": dict}
        响应: {"result": bool|tuple, "time": float, "in_time": ...}
        注意: webenv 的 click 内部会自动执行鼠标移动动画、滚动到视口、
              debugbar 闪烁、视觉变化检测等全套流程
        """
        if id2xpath is None:
            id2xpath = self._cached_id2xpath
        resp = http_requests.post(
            f"{self.base_url}/click",
            json={"id": element_id, "id2xpath": id2xpath},
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def enter(self, element_id: int, text: str, id2xpath: dict = None) -> dict:
        """
        POST /enter
        请求体: EnterRequest {"id": int, "text": str, "id2xpath": dict}
        响应: {"result": bool, "time": float, "in_time": ...}
        注意: webenv 的 enter 内部会自动执行鼠标移动、点击聚焦、
              逐字符输入动画（含中文特殊延迟）、range slider 特判等
        """
        if id2xpath is None:
            id2xpath = self._cached_id2xpath
        resp = http_requests.post(
            f"{self.base_url}/enter",
            json={"id": element_id, "text": text, "id2xpath": id2xpath},
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def select(self, element_id: int, value: str, id2xpath: dict = None) -> dict:
        """
        POST /select
        请求体: SelectRequest {"id": int, "value": str, "id2xpath": dict}
        响应: {"result": bool|tuple, "time": float, "in_time": ...}
        注意: webenv 的 select 内部支持 <select> 下拉框（含模糊匹配）、
              radio 按钮、date input 等多种类型
        """
        if id2xpath is None:
            id2xpath = self._cached_id2xpath
        resp = http_requests.post(
            f"{self.base_url}/select",
            json={"id": element_id, "value": value, "id2xpath": id2xpath},
            proxies=PROXIES,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def execute_action(self, action: dict, id2xpath: dict = None) -> dict:
        """
        统一的操作执行入口，根据 action["action"] 分派到 click/enter/select。

        action 格式（与 helpers.parse_action_seq 的输出对齐）:
          {"action": "click", "id": 42}
          {"action": "input", "id": 42, "input_val": "hello"}
          {"action": "select", "id": 42, "value": "option1"}
        """
        act_type = action.get("action", "").lower()
        raw_id = action.get("id")
        elem_id = int(raw_id) if raw_id is not None else None

        if act_type == "click":
            return self.click(elem_id, id2xpath)
        elif act_type in ("input", "enter"):
            text = action.get("input_val") or action.get("text") or action.get("value") or ""
            return self.enter(elem_id, str(text), id2xpath)
        elif act_type == "select":
            value = action.get("value") or action.get("input_val") or ""
            return self.select(elem_id, str(value), id2xpath)
        elif act_type == "scroll":
            if elem_id == -1:
                return self.scroll_to_top()
            return self.scroll_to_element(elem_id, id2xpath)
        else:
            print(f"[WARN] 未知操作类型: {act_type}")
            return {"result": False, "error": f"unknown action type: {act_type}"}

    def execute_action_sequence(self, actions: list[dict], id2xpath: dict = None) -> list[dict]:
        """
        顺序执行一组操作，每步之后截图。
        每步操作前重新获取 DOM（因为页面可能因上一步操作而变化，DOM id 会重新分配）。
        返回: [{"action": dict, "api_response": dict, "img_b64_after": str}, ...]
        """
        results = []
        for action in actions:
            # 关键：每步前刷新 DOM，因为 webenv 的 DOM id 是动态分配的
            domtree_fresh, id2xpath_fresh = self.get_dom_info()

            # 如果 action 中的 id 在新的 id2xpath 中不存在，
            # 尝试通过 dom_info 中的文本/标签重新匹配
            act_id = action.get("id")
            # -1 is the scroll-to-top sentinel; skip DOM existence check for it
            if act_id is not None and int(act_id) != -1 and str(act_id) not in id2xpath_fresh and int(act_id) not in [int(k) for k in id2xpath_fresh.keys()]:
                print(f"  [WARN] 元素 id={act_id} 在刷新后的 DOM 中不存在，尝试重新匹配...")
                new_id = self._rematch_element(action, domtree_fresh)
                if new_id is not None:
                    print(f"  [INFO] 重新匹配成功: {act_id} → {new_id}")
                    action = dict(action)
                    action["id"] = new_id
                else:
                    print(f"  [WARN] 重新匹配失败，使用原 id 尝试")

            api_resp = self.execute_action(action, id2xpath_fresh)
            time.sleep(2)  # 等待页面完全加载后再截图
            img_b64 = self.get_screenshot_b64()
            results.append({
                "action": action,
                "api_response": api_resp,
                "img_b64_after": img_b64,
            })
        return results

    def _rematch_element(self, action: dict, domtree) -> int | None:
        """
        当 DOM 刷新后原 id 失效时，通过 dom_info 中的文本和标签
        在新的 DOM Tree 中尝试重新找到对应元素。
        """
        dom_info = action.get("dom_info")
        if not dom_info or not isinstance(dom_info, dict):
            return None

        target_text = (dom_info.get("visible_text") or "").strip()
        target_tag = (dom_info.get("tag") or "").upper()

        if not target_text and not target_tag:
            return None

        def search_tree(node):
            if not isinstance(node, dict):
                return None
            node_text = (node.get("visible_text") or "").strip()
            node_tag = (node.get("tag") or "").upper()

            if target_text and target_tag:
                if node_text == target_text and node_tag == target_tag:
                    return node.get("id")
            elif target_text:
                if node_text == target_text:
                    return node.get("id")

            for child in node.get("children", []):
                found = search_tree(child)
                if found is not None:
                    return found
            return None

        return search_tree(domtree)

    # ---------- 页面控制 ----------

    def reset(self) -> dict:
        """POST /reset"""
        resp = http_requests.post(
            f"{self.base_url}/reset", json={},
            proxies=PROXIES, timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def load_url(self, url: str) -> dict:
        """POST /load_url"""
        resp = http_requests.post(
            f"{self.base_url}/load_url",
            json={"url": url},
            proxies=PROXIES, timeout=60,
        )
        return resp.json()

    def check_scroll_needed(self) -> bool:
        """POST /check_if_scroll_needed → bool"""
        resp = http_requests.post(
            f"{self.base_url}/check_if_scroll_needed",
            json={}, proxies=PROXIES, timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def scroll_page(self) -> dict:
        """POST /scroll_page → 触发人工模拟滚动浏览"""
        resp = http_requests.post(
            f"{self.base_url}/scroll_page",
            proxies=PROXIES, timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def scroll_to_element(self, elem_id, id2xpath: dict) -> dict:
        """POST /scroll → scroll to element (or to bottom if elem_id is None)"""
        resp = http_requests.post(
            f"{self.base_url}/scroll",
            json={"id": elem_id, "id2xpath": id2xpath},
            proxies=PROXIES, timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    def scroll_to_top(self) -> dict:
        """POST /scroll_to_top → instantly scroll the page back to position (0, 0)"""
        resp = http_requests.post(
            f"{self.base_url}/scroll_to_top",
            json={}, proxies=PROXIES, timeout=REQUEST_TIMEOUT,
        )
        return resp.json()

    # ---------- 视频录制 ----------

    def start_video_cdp(self, width: int = None, height: int = None) -> dict:
        """
        POST /start_video_cdp
        使用 CDP 无感录制，不重建页面 context，直接在当前页面上附加截帧流。
        返回: {"result": bool, "recording": bool, "t0": float}
        """
        payload = {}
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height
        resp = http_requests.post(
            f"{self.base_url}/start_video_cdp",
            json=payload, proxies=PROXIES, timeout=30,
        )
        return resp.json()

    def end_video_cdp(self) -> dict:
        """
        POST /end_video_cdp
        返回: {"result": bool, "frames_count": int, "frames": [base64 JPEG str, ...]}
        注意: frames 是 CDP Screencast 捕获的 JPEG 帧列表，可直接用于视觉验证，
              无需 ffmpeg 或 opencv 解码。
        """
        resp = http_requests.post(
            f"{self.base_url}/end_video_cdp",
            proxies=PROXIES, timeout=120,
        )
        return resp.json()
