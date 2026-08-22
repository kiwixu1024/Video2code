import os
import asyncio
import base64
import difflib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import psutil  # 需 pip install psutil
import aiofiles
import uvicorn
import sys
from gym import spaces
from playwright.async_api import async_playwright
import argparse

import math
import io
import re  # ✅ 你的 enter() 里用到了 re，这里补上
import tempfile
import shutil
from PIL import Image
from pydantic import BaseModel


class WebHtmlGymEnv:
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, html_file="page.html", viewport_width=1920, viewport_height=1080):
        self.html_file = html_file
        self.is_url = str(html_file).startswith("http://") or str(html_file).startswith("https://")
        self.action_space = spaces.Discrete(3)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.viewport_height, self.viewport_width, 3), dtype=np.uint8)
        self.p = None
        self.browser = None
        self.context = None
        self.page = None
        self.click_history = []

        # === 录制状态 ===
        self.recording: bool = False
        self._video_tmpdir: Optional[str] = None
        self._last_url: Optional[str] = None  # 记录当前页面 URL，用于录制切换/结束后恢复

        # === 鼠标
        self._mouse_x = None
        self._mouse_y = None

    # ============ 鼠标可视化：注入 / 定位 / 匀速移动 ============

    async def _inject_cursor_overlay(self):
        """在页面里放一个可见的鼠标小圆点，id=___webenv_mouse（可重复调用，幂等）。"""
        js = """
        () => {
          if (document.getElementById('___webenv_mouse')) return true;
          const tip = document.createElement('div');
          tip.id = '___webenv_mouse';
          Object.assign(tip.style, {
            position: 'fixed',
            left: '0px',
            top: '0px',
            width: '16px',
            height: '16px',
            margin: '0',
            padding: '0',
            borderRadius: '50%',
            border: '2px solid rgba(0,0,0,0.7)',
            background: 'rgba(255,255,255,0.9)',
            boxShadow: '0 0 8px rgba(0,0,0,0.35)',
            pointerEvents: 'none',
            zIndex: '2147483647',
            transform: 'translate(-9999px,-9999px)'
          });
          document.documentElement.appendChild(tip);
          return true;
        }
        """
        try:
            await self.page.evaluate(js)
        except Exception:
            pass

    async def _cursor_exists(self) -> bool:
        try:
            return await self.page.evaluate("()=>!!document.getElementById('___webenv_mouse')")
        except Exception:
            return False

    async def _set_cursor_pos(self, x: float, y: float):
        """移动可视化鼠标（不触发真实鼠标事件）。"""
        await self.page.evaluate(
            """({x, y}) => {
                const el = document.getElementById('___webenv_mouse');
                if (!el) return;
                el.style.transform = `translate(${x-8}px, ${y-8}px)`;
            }""",
            {"x": x, "y": y}
        )

    async def _ensure_cursor_ready(self):
        if not await self._cursor_exists():
            await self._inject_cursor_overlay()

    async def _get_viewport_center(self):
        # Playwright Python: viewport_size 是属性，不是方法
        vp = self.page.viewport_size  # ✅ 直接取属性，切记不加 ()
        if not vp:
            # 有些情况下可能是 None，用浏览器窗口尺寸兜底
            vp = await self.page.evaluate(
                "() => ({ width: window.innerWidth, height: window.innerHeight })"
            )
        return (int(vp["width"] // 2), int(vp["height"] // 2))


    async def _ensure_mouse_origin(self):
        """若未记录当前鼠标位置，则将其设为视窗中心（真实与可视都放到中点）。"""
        if self._mouse_x is None or self._mouse_y is None:
            cx, cy = await self._get_viewport_center()
            await self.page.mouse.move(cx, cy)
            await self._ensure_cursor_ready()
            await self._set_cursor_pos(cx, cy)
            self._mouse_x, self._mouse_y = cx, cy

    async def _center_of_xpath(self, xpath: str):
        """滚动到可见并返回元素中心坐标（相对视窗）。"""
        loc = self.page.locator(f"xpath={xpath}")
        await loc.scroll_into_view_if_needed()
        box = await loc.bounding_box()
        if not box:
            return None
        return (box["x"] + box["width"]/2, box["y"] + box["height"]/2)

    async def _move_mouse_uniform_to(self, x: float, y: float, duration_ms: int = 600, fps: int = 90):
        """匀速直线：同时推进 Playwright 真鼠标与可视化鼠标。"""
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        sx, sy = float(self._mouse_x), float(self._mouse_y)
        tx, ty = float(x), float(y)
        frames = max(1, round((duration_ms/1000) * fps))

        # 首帧把真实鼠标移到起点，避免跨页初始化偏差
        await self.page.mouse.move(sx, sy)

        for i in range(1, frames+1):
            t = i / frames
            cx = sx + (tx - sx) * t
            cy = sy + (ty - sy) * t
            await self.page.mouse.move(cx, cy, steps=1)
            await self._set_cursor_pos(cx, cy)
            await asyncio.sleep(1 / fps)

        # 最后一滴定在目标点
        await self.page.mouse.move(tx, ty, steps=1)
        await self._set_cursor_pos(tx, ty)
        self._mouse_x, self._mouse_y = tx, ty

    async def setup(self):
        # 初始化网页
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )
        self.page = await self.context.new_page()
        # 可视化鼠标：注入并放到视窗中央
        await self._inject_cursor_overlay()
        cx, cy = await self._get_viewport_center()
        await self.page.mouse.move(cx, cy)
        await self._set_cursor_pos(cx, cy)
        self._mouse_x, self._mouse_y = cx, cy


    async def _goto_current_target(self):
        """根据 self.html_file / self.is_url 导航并更新 _last_url。"""
        if self.is_url:
            await self.page.goto(self.html_file, timeout=60000)
        else:
            await self.page.goto(f"file:///{os.path.abspath(self.html_file)}", timeout=60000)
        self._last_url = self.page.url

    async def reset(self):
        # 重置网页（保持同一 context，不会影响录制）
        if self.page is None:
            await self.setup()
        await self.page.set_viewport_size({
            "width": self.viewport_width,
            "height": self.viewport_height
        })
        await self._goto_current_target()
        self.click_history = []

    async def load_url(self, url: str):
        self.html_file = url
        self.is_url = True
        if self.page is None:
            await self.setup()
        await self.page.goto(url, timeout=60000)
        self._last_url = self.page.url
        self.click_history = []
        print(f"已加载新页面: {url}")

    async def render(self, mode="human"):
        await asyncio.sleep(1)  # 可调
        size = await self.page.evaluate("""() => {
            return {
                width: Math.max(
                    document.documentElement.clientWidth || 1920,
                    document.body ? document.body.scrollWidth : 1920,
                    document.documentElement.scrollWidth || 1920,
                    document.documentElement.offsetWidth || 1920
                ),
                height: Math.max(
                    document.documentElement.clientHeight || 1080,
                    document.body ? document.body.scrollHeight : 1080,
                    document.documentElement.scrollHeight || 1080,
                    document.documentElement.offsetHeight || 1080
                )
            };
        }""")
        await self.page.set_viewport_size({
            "width": max(size["width"], 1920),
            "height": max(size["height"], 1080)
        })
        obs = await self.page.screenshot()
        return obs  # bytes

    async def render_sized(self, jpeg_quality=95):
        """
        用于模型输入的：渲染后将宽、高都等比缩小到原来的1/1.3倍，返回bytes。
        """
        obs = await self.render()   # type: bytes
        image = Image.open(io.BytesIO(obs))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size

        scale = 1/1.3
        w_bar = int(w * scale)
        h_bar = int(h * scale)

        image = image.resize((w_bar, h_bar), Image.BICUBIC)
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG', quality=jpeg_quality)
        buffered.seek(0)
        return buffered.getvalue()

    async def render_sized_small(self, jpeg_quality=85):
        """补齐接口：更小尺寸（例如 0.5x）。"""
        obs = await self.render()
        image = Image.open(io.BytesIO(obs))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size
        scale = 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG', quality=jpeg_quality)
        buffered.seek(0)
        return buffered.getvalue()

    async def get_all_dom_elements(self):
        tag_names = [ "input", "button", "textarea", "a", "[role=button]", "[role=checkbox]", "[role=radio]", ]
        doms = []
        for tag in tag_names:
            elems = await self.page.query_selector_all(tag)
            for elem in elems:
                try:
                    props = dict()
                    props['tag'] = await self.page.evaluate('el=>el.tagName.toLowerCase()', elem)
                    props['id'] = await self.page.evaluate('el=>el.id', elem)
                    props['class'] = await self.page.evaluate('el=>el.className', elem)
                    props['name'] = await self.page.evaluate('el=>el.getAttribute("name")', elem)
                    props['type'] = await self.page.evaluate('el=>el.getAttribute("type")', elem)
                    props['placeholder'] = await self.page.evaluate('el=>el.getAttribute("placeholder")', elem)
                    props['text'] = await self.page.evaluate('(el)=>(el.innerText||"")', elem)
                    doms.append(props)
                except Exception as e:
                    print(f"get_all_dom_elements err: {e}")
        return doms

    async def get_dom_tree_with_id(self):
        js_code = """/* 省略：与原版一致，这里保持不变 */"""  # 为压缩篇幅，保留你原来那段 JS（完全兼容）
        # —— 为了保持完整性，直接放回你原代码块 —— 
        js_code = """
        () => {
            let nodeId = 0;
            let id2xpath = {};
            function getXPath(node) {
                if (node.nodeType !== 1) return '';
                if (!node.parentNode || node === document.documentElement) return '/' + node.tagName;
                let ix = 1;
                let sib = node.previousSibling;
                while (sib) {
                    if (sib.nodeType === 1 && sib.tagName === node.tagName) ix++;
                    sib = sib.previousSibling;
                }
                return getXPath(node.parentNode) + '/' + node.tagName + '[' + ix + ']';
            }
            function isButtonVisibleAndEnabled(elem) {
                if (!elem) return false;
                try { elem.scrollIntoView({block: 'center', inline: 'center'}); } catch(e) {}
                const style = window.getComputedStyle(elem);
                if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
                const rect = elem.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return false;
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;
                const topElem = document.elementFromPoint(cx, cy);
                let current = topElem;
                while (current) {
                    if (current === elem) return true;
                    current = current.parentElement;
                }
                return false;
            }
            function isInteractiveElement(node) {
                if (!node || !node.tagName) return false;
                const tag = node.tagName.toLowerCase();
                if (['button', 'a', 'select', 'textarea', 'label'].includes(tag)) return true;
                if (tag === 'input') {
                    const t = (node.getAttribute('type') || 'text').toLowerCase();
                    return [
                        'button', 'submit', 'reset', 
                        'radio', 'checkbox', 'range', 
                        'file', 'color', 'date', 
                        'time', 'month', 'week', 
                        'email', 'tel', 'password', 'text', 'search', 'number', 'url'
                    ].includes(t);
                }
                return false;
            }
            function isTextInputLike(node) {
                if (!node || !node.tagName) return false;
                const tag = node.tagName.toLowerCase();
                if (tag === 'textarea') return true;
                if (tag === 'input') {
                    const t = (node.getAttribute('type') || 'text').toLowerCase();
                    return (
                        t === 'text' ||
                        t === 'password' ||
                        t === 'search' ||
                        t === 'number' ||
                        t === 'email' ||
                        t === 'tel' ||
                        t === 'url'
                    );
                }
                return false;
            }
            function getVisibleText(node) {
                if (!node || node.nodeType !== 1) return undefined;
                const tag = node.tagName.toLowerCase();
                if (tag === 'input' || tag === 'textarea') {
                    return node.value || '';
                } else if (tag === 'select') {
                    const selected = node.selectedOptions && node.selectedOptions[0];
                    return selected ? selected.textContent : '';
                } else {
                    return node.innerText || '';
                }
            }
            function traverse(node, parents=[]) {
                if (node.tagName && node.tagName.toUpperCase() === 'SCRIPT') {
                    return null;
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return null;
                const tag = node.tagName ? node.tagName.toLowerCase() : '';
                let can_interact = false;
                let input_value = undefined;
                let visible_text = undefined;
                let should_keep = false;
                let options = undefined;
                if (isInteractiveElement(node)) {
                    should_keep = true;
                    try {
                        can_interact = isButtonVisibleAndEnabled(node);
                    } catch (e) {
                        can_interact = false;
                    }
                }
                if (isTextInputLike(node)) {
                    try {
                        input_value = node.value;
                    } catch (e) {
                        input_value = null;
                    }
                }
                if (tag === 'select') {
                    should_keep = true;
                    try {
                        options = [];
                        for (let opt of node.options) {
                            options.push({
                                value: opt.value,
                                text: opt.text,
                                selected: opt.selected
                            });
                        }
                    } catch (e) { options = null; }
                }
                const new_parents = parents.slice();
                if (node.tagName) new_parents.push(node.tagName);
                const children = [];
                for (let child of node.childNodes) {
                    const childResult = traverse(child, new_parents);
                    if (childResult !== null) {
                        children.push(childResult);
                    }
                }
                if (!should_keep && children.length === 0) return null;
                const id = nodeId++;
                const xpath = getXPath(node);
                id2xpath[id] = xpath;
                const attrs = {};
                if (node.attributes) {
                    for (let attr of node.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                }
                let result = {
                    id: id,
                    tag: node.tagName,
                    attrs: attrs,
                    children: children
                };
                if (should_keep) {
                    result['can_interact'] = can_interact;
                    if (isTextInputLike(node)) {
                        result['input_value'] = input_value;
                    }
                    if (['button', 'input', 'textarea', 'select', 'a', 'label'].includes(tag)) {
                        result['visible_text'] = getVisibleText(node);
                    }
                    if (tag === 'select') {
                        result['options'] = options;
                    }
                }
                return result;
            }
            const domtree = traverse(document.body, []);
            return {domtree, id2xpath};
        }
        """
        result = await self.page.evaluate(js_code)
        return result["domtree"], result["id2xpath"]

    async def click(self, unique_id: int, id2xpath: dict):
        xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
        if not xpath:
            print(f"[CLICK] No xpath found for id={unique_id}")
            return False
        elem = await self.page.query_selector(f'xpath={xpath}')
        if elem is None:
            print(f"[CLICK] No element found by xpath {xpath} for id={unique_id}")
            return False
        try:
            await elem.scroll_into_view_if_needed()
            center = await self._center_of_xpath(xpath)
            if not center:
                print(f"[CLICK] Element not visible/bbox missing: {xpath}")
                return False

            tx, ty = center
            # 匀速移动 → 点击
            await self._move_mouse_uniform_to(tx, ty, duration_ms=600, fps=90)
            await self.page.mouse.click(tx, ty)

            self._last_url = self.page.url
            self.click_history.append(("click", unique_id))
            return True
        except Exception as e:
            print(f"[CLICK] Failed click: {e}")
            return False

    async def enter(self, unique_id, content, id2xpath):
        xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
        if not xpath:
            print(f"[ENTER] No xpath found for id={unique_id}")
            return False
        elem = await self.page.query_selector(f'xpath={xpath}')
        if elem is None:
            print(f"[ENTER] No element found by xpath {xpath} for id={unique_id}")
            return False
        try:
            tag = await elem.evaluate("el => el.tagName.toLowerCase()")
            if tag not in ["input", "textarea"]:
                print(f"[ENTER] Element id={unique_id} is {tag}, not input/textarea, cannot enter text.")
                return False

            # 先平滑移动并单击获取焦点
            await elem.scroll_into_view_if_needed()
            center = await self._center_of_xpath(xpath)
            if not center:
                print(f"[ENTER] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center
            await self._move_mouse_uniform_to(tx, ty, duration_ms=600, fps=90)
            await self.page.mouse.click(tx, ty)

            is_number_input = await elem.evaluate(
                """el => {
                    if (el.tagName.toLowerCase() === 'input' && el.type && el.type.toLowerCase() === 'number') return true;
                    if (el.pattern && el.pattern.match(/^\\d+$/)) return true;
                    if (el.inputMode && el.inputMode.toLowerCase().includes('numeric')) return true;
                    if ((el.className || '').match(/(number|num)/i)) return true;
                    return false;
                }"""
            )

            fill_content = str(content)
            if is_number_input:
                try:
                    fill_content = str(float(content))
                except Exception:
                    m = re.search(r"[-+]?\d*\.?\d+", str(content))
                    fill_content = str(float(m.group())) if m else "0"

            await elem.fill(fill_content)
            self._last_url = self.page.url
            self.click_history.append(("enter", unique_id, fill_content))
            return True
        except Exception as e:
            print(f"[ENTER] Failed to enter text: {e}")
            return False

    async def select(self, unique_id: Any, value: Any, id2xpath: Dict[Any, str]) -> bool:
        xpath: Optional[str] = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id)) if isinstance(unique_id, (str, int)) else None
        if not xpath:
            print(f"[SELECT] No xpath found for id={unique_id}")
            return False

        elem = await self.page.query_selector(f"xpath={xpath}")
        if elem is None:
            print(f"[SELECT] No element found by xpath {xpath} for id={unique_id}")
            return False

        try:
            tag = await elem.evaluate("el => el.tagName.toLowerCase()")
            await elem.scroll_into_view_if_needed()
            center = await self._center_of_xpath(xpath)
            if not center:
                print(f"[SELECT] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center
            # 移动并轻点获取焦点（对 select 展开下拉、radio/日期控件更自然）
            await self._move_mouse_uniform_to(tx, ty, duration_ms=600, fps=90)
            await self.page.mouse.click(tx, ty)

            if tag == "select":
                try:
                    result = await elem.select_option(str(value))
                    if result and result != []:
                        self._last_url = self.page.url
                        self.click_history.append(("select", unique_id, value))
                        return True

                    options = await elem.evaluate(
                        "el => Array.from(el.options).map(o => ({value: o.value, text: o.text}))"
                    )
                    option_texts = [opt["text"] for opt in options]
                    closest = difflib.get_close_matches(str(value), option_texts, n=1, cutoff=0.0)
                    if not closest:
                        print(f"[SELECT] No similar option (by text) for value: {value}")
                        return False

                    closest_text = closest[0]
                    for opt in options:
                        if opt["text"] == closest_text:
                            result = await elem.select_option(opt["value"])
                            if result and result != []:
                                self._last_url = self.page.url
                                self.click_history.append(("select-closest", unique_id, opt["value"]))
                                return True
                    print("[SELECT] Failed to select even after closest-match logic.")
                    return False
                except Exception as e:
                    print(f"[SELECT] Exception during <select> handling: {e}")
                    return False

            elif tag == "input":
                input_type = await elem.get_attribute("type")

                if input_type == "radio":
                    # 已经移动并点击过一次，保险起见再点一下
                    await self.page.mouse.click(tx, ty)
                    self._last_url = self.page.url
                    self.click_history.append(("select-radio", unique_id))
                    return True

                elif input_type == "date":
                    await elem.fill(str(value))
                    self._last_url = self.page.url
                    self.click_history.append(("select-date", unique_id, value))
                    return True

                else:
                    print(f"[SELECT] Element id={unique_id} is input but not type=date/radio, skipping.")
                    return False

            else:
                print(f"[SELECT] Element id={unique_id} is {tag}, not select/input[type=date|radio], cannot select value.")
                return False

        except Exception as e:
            print(f"[SELECT] Failed to select: {e}")
            return False

    # ========= 录制：核心新增 =========
    async def start_video(self, width: Optional[int] = None, height: Optional[int] = None) -> bool:
        """
        开始录制（切换到启用 record_video 的新 context/page），后续 reset/click/enter/select/load_url 等都不会中断录制。
        """
        print("正在准备启动视频录制")
        if self.recording:
            print("[VIDEO] Recording already started.")
            return True

        if self.page is None:
            await self.setup()
            await self._goto_current_target()

        # 记录当前 URL（切换 context 前）
        try:
            self._last_url = self.page.url
        except Exception:
            self._last_url = None

        # 为视频建立临时目录
        self._video_tmpdir = tempfile.mkdtemp(prefix="webenv_video_")

        # 关闭旧 context/page（释放干净），并创建新 context 开启录制
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
        except Exception:
            pass

        record_size = None
        if width and height:
            record_size = {"width": int(width), "height": int(height)}
        else:
            # 默认用当前 viewport
            record_size = {"width": self.viewport_width, "height": self.viewport_height}

        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1,
            record_video_dir=self._video_tmpdir,
            record_video_size=record_size
        )
        self.page = await self.context.new_page()
    
        # 回到之前页面
        if self._last_url:
            await self.page.goto(self._last_url, timeout=60000)
        else:
            await self._goto_current_target()
        # 录像用的 context 中也注入/归位光标
        await self._inject_cursor_overlay()
        cx, cy = await self._get_viewport_center()
        await self.page.mouse.move(cx, cy)
        await self._set_cursor_pos(cx, cy)
        self._mouse_x, self._mouse_y = cx, cy

        self.recording = True
        print(f"[VIDEO] Started. dir={self._video_tmpdir}, size={record_size}")
        return True

    async def end_video(self) -> Optional[bytes]:
        """
        结束录制：关闭 page/context 获取视频文件，读入 bytes 返回；
        然后自动恢复到非录制 context/page（保持原 URL），以便继续使用。
        """
        if not self.recording:
            print("[VIDEO] Not recording.")
            return None

        # 记录要恢复的 URL
        try:
            current_url = self.page.url
        except Exception:
            current_url = self._last_url or None

        # 关闭以触发视频写出
        video_bytes: Optional[bytes] = None
        try:
            video_path = None
            try:
                # 只有关闭 page 后才能拿到路径
                await self.page.close()
                # Playwright: page.video.path() 需在 page.close() 后调用
                # 这里无法再调用 self.page.video，改成从目录中读取唯一文件更稳妥
            except Exception as e:
                print(f"[VIDEO] page.close() error: {e}")

            try:
                if self.context:
                    await self.context.close()
            except Exception as e:
                print(f"[VIDEO] context.close() error: {e}")

            # 从临时目录里找视频文件（一般是 .webm）
            if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                candidates = [os.path.join(self._video_tmpdir, f) for f in os.listdir(self._video_tmpdir)]
                candidates = [p for p in candidates if os.path.isfile(p)]
                if candidates:
                    # 取最新/唯一的那个
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    video_path = candidates[0]
                    with open(video_path, "rb") as vf:
                        video_bytes = vf.read()
                else:
                    print("[VIDEO] No video file found in temp dir.")
        finally:
            # 清理临时目录
            try:
                if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                    shutil.rmtree(self._video_tmpdir, ignore_errors=True)
            except Exception:
                pass
            self._video_tmpdir = None
            self.recording = False

            # 恢复普通（非录制）context/page
            self.context = await self.browser.new_context(
                viewport={'width': self.viewport_width, 'height': self.viewport_height},
                device_scale_factor=1
            )
            self.page = await self.context.new_page()
            if current_url:
                await self.page.goto(current_url, timeout=60000)
                self._last_url = self.page.url
            else:
                await self._goto_current_target()
                        # 恢复后也要注入/归位光标
            await self._inject_cursor_overlay()
            cx, cy = await self._get_viewport_center()
            await self.page.mouse.move(cx, cy)
            await self._set_cursor_pos(cx, cy)
            self._mouse_x, self._mouse_y = cx, cy

        return video_bytes

    async def save_state(self):
        return list(self.click_history)

    async def restore_state(self, click_path):
        await self.reset()
        for item in click_path:
            if isinstance(item, int):
                await self.click(item)
            elif isinstance(item, tuple):
                if item[0] == "pixel":
                    _, x, y, button = item
                    await self.click_pixel(x, y, button)
                elif item[0] == "type":
                    _, x, y, text = item
                    await self.type_pixel(x, y, text)
                elif item[0] == "selector":
                    _, selector = item
                    await self.click_by_selector(selector)

    async def save_current_html(self, file_name="my_new_page.html"):
        html = await self.page.evaluate('document.documentElement.outerHTML')
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"{file_name} 保存完成 (from evaluate)")

    async def close(self):
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass
        if self.browser:
            await self.browser.close()
        if self.p:
            await self.p.stop()


# ================== FastAPI service ===================

app = FastAPI()
env: Optional[WebHtmlGymEnv] = None

class ClickRequest(BaseModel):
    id: int
    id2xpath: dict

class EnterRequest(BaseModel):
    id: int
    text: str
    id2xpath: dict

class SelectRequest(BaseModel):
    id: int
    value: str
    id2xpath: dict

class LoadUrlRequest(BaseModel):
    url: str

class StartVideoRequest(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    global env
    if len(sys.argv) > 1:
        html_file = sys.argv[1]  # 允许 python server.py xxx.html 这样传参
    else:
        html_file = "page.html"
    env = WebHtmlGymEnv(html_file=html_file)
    await env.reset()
    print(f"WebHtmlGymEnv启动并初始化完成。加载html: {html_file}")

@app.post("/click")
async def api_click(req: ClickRequest):
    global env
    result = await env.click(req.id, req.id2xpath)
    return {"result": result}

@app.post("/enter")
async def api_enter(req: EnterRequest):
    global env
    result = await env.enter(req.id, req.text, req.id2xpath)
    return {"result": result}

@app.post("/select")
async def api_select(req: SelectRequest):
    global env
    result = await env.select(req.id, req.value, req.id2xpath)
    return {"result": result}

@app.get("/observe")
async def api_observe():
    global env
    obs_bytes = await env.render()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}

@app.get("/observe_sized")
async def api_observe_sized():
    global env
    obs_bytes = await env.render_sized()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}

@app.get("/observe_sized_small")
async def api_observe_sized_small():
    global env
    obs_bytes = await env.render_sized_small()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}

@app.get("/all_dom_elements")
async def api_all_dom_elements():
    global env
    doms = await env.get_all_dom_elements()
    return {"doms": doms}

@app.get("/dom_tree_with_id")
async def api_dom_tree_with_id():
    global env
    domtree, id2xpath = await env.get_dom_tree_with_id()
    return {"domtree": domtree, "id2xpath": id2xpath}

@app.post("/reset")
async def api_reset():
    global env
    await env.reset()
    return {"result": True}

@app.get("/save_state")
async def api_save_state():
    global env
    state = await env.save_state()
    return {"state": state}

@app.post("/restore_state")
async def api_restore_state(request: Request):
    global env
    data = await request.json()
    click_path = data.get("click_path", [])
    await env.restore_state(click_path)
    return {"result": True}

# ===== 新增：视频录制 API =====
@app.post("/start_video")
async def api_start_video(req: StartVideoRequest):
    global env
    print("正在准备启动视频录制")
    ok = await env.start_video(width=req.width, height=req.height)
    return {"result": ok, "recording": env.recording}

# @app.post("/end_video")
# async def api_end_video():
#     global env
#     video_bytes = await env.end_video()
#     if not video_bytes:
#         return {"result": False, "error": "no_video"}
#     video_b64 = base64.b64encode(video_bytes).decode("utf-8")
#     # Chromium 输出通常为 .webm
#     return {"result": True, "content_type": "video/webm", "video_b64": video_b64}

import tempfile
import subprocess
import base64
import io
from fastapi.responses import FileResponse

@app.post("/end_video")
async def api_end_video():
    global env
    video_bytes = await env.end_video()
    if not video_bytes:
        return {"result": False, "error": "no_video"}

    # --- 写临时 webm 文件 ---
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(video_bytes)
        webm_path = f.name
    mp4_path = webm_path.replace(".webm", ".mp4")

    # --- 用 ffmpeg 转成 mp4 ---
    subprocess.run([
        "ffmpeg", "-y", "-i", webm_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        mp4_path
    ], check=True)

    # --- 返回 MP4 文件（二选一）---
    with open(mp4_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("ascii")
    return {"result": True, "content_type": "video/mp4", "video_b64": video_b64}

@app.on_event("shutdown")
async def shutdown_event():
    global env
    await env.close()
    print("已关闭WebHtmlGymEnv。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("html_or_url", nargs='?', default="page.html")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        "webenv:app",
        host='0.0.0.0',
        port=args.port,
        reload=True,
        reload_dirs=["./webenv-init"],
    )
