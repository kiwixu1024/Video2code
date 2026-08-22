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
import subprocess  # ✅ 用于 ffmpeg
from fastapi.responses import FileResponse  # 保留
import time  # ✅ 新增：用于高精度计时


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

        # === NEW: 保存所有段的视频（b64 列表）===
        # 每个元素：{"content_type": "video/mp4" | "video/webm", "video_b64": "..."}
        self.video_segments: List[Dict[str, str]] = []   # ### NEW
            # 在 __init__ 里加：
        self._t0: Optional[float] = None
        self._first_event_monotonic: Optional[float] = None  # 可用于后面物理裁切


    async def start_live_timer(self, label: str = "Click"):
        """
        在页面右上角启动一个实时计时器，每 ~100ms 刷新。
        如果之前已存在计时器，会复用并重置为运行状态（非红色）。
        """
        try:
            await self.page.evaluate(
                """
                (label) => {
                  // 容器
                  let el = document.getElementById('___webenv_time');
                  if (!el) {
                    el = document.createElement('div');
                    el.id = '___webenv_time';
                    Object.assign(el.style, {
                      position: 'fixed',
                      top: '12px',
                      right: '12px',
                      zIndex: '2147483647',
                      background: 'rgba(0,0,0,0.75)',
                      color: '#fff',
                      padding: '6px 10px',
                      borderRadius: '12px',
                      font: '500 14px/1.2 -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif',
                      boxShadow: '0 2px 8px rgba(0,0,0,.25)',
                      opacity: '0',
                      transition: 'opacity .2s ease'
                    });
                    document.documentElement.appendChild(el);
                  }

                  // 状态对象
                  if (!window.__webenvTimer) window.__webenvTimer = {};
                  const T = window.__webenvTimer;

                  // 重置样式（确保不是红色停止态）
                  el.style.background = 'rgba(0,0,0,0.75)';
                  el.style.color = '#fff';
                  el.style.border = 'none';

                  // 记录起点 & 启动刷新
                  T.label = label || 'Click';
                  T.startEpoch = performance.now();
                  if (T.tid) clearInterval(T.tid);
                  const fmt = (ms) => (ms/1000).toFixed(3) + 's';
                  const update = () => {
                    const elapsed = performance.now() - T.startEpoch;
                    el.textContent = `${T.label}: ${fmt(elapsed)}`;
                  };
                  update();
                  T.tid = setInterval(update, 100);

                  // 显现
                  requestAnimationFrame(() => { el.style.opacity = '1'; });
                }
                """,
                label
            )
        except Exception:
            pass

    async def stop_live_timer(self, final_seconds: float, label: str = "Click"):
        """
        停止右上角计时器：清除刷新定时器，固定显示最终用时，并将背景标红。
        如页面在计时期间发生导航、导致浮层丢失，则会重建后直接显示最终结果。
        """
        try:
            await self.page.evaluate(
                """
                (sec, label) => {
                  const ensureEl = () => {
                    let el = document.getElementById('___webenv_time');
                    if (!el) {
                      el = document.createElement('div');
                      el.id = '___webenv_time';
                      Object.assign(el.style, {
                        position: 'fixed',
                        top: '12px',
                        right: '12px',
                        zIndex: '2147483647',
                        padding: '6px 10px',
                        borderRadius: '12px',
                        font: '500 14px/1.2 -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif',
                        boxShadow: '0 2px 8px rgba(0,0,0,.25)',
                        opacity: '0',
                        transition: 'opacity .2s ease'
                      });
                      document.documentElement.appendChild(el);
                      requestAnimationFrame(() => { el.style.opacity = '1'; });
                    }
                    return el;
                  };

                  if (!window.__webenvTimer) window.__webenvTimer = {};
                  const T = window.__webenvTimer;
                  const el = ensureEl();

                  // 停止刷新
                  if (T.tid) { clearInterval(T.tid); T.tid = null; }

                  // 标红并显示最终结果
                  el.textContent = `${label || 'Click'}: ${Number(sec).toFixed(3)}s`;
                  el.style.background = '#d93025';   // 红色
                  el.style.color = '#fff';
                  el.style.border = '1px solid rgba(0,0,0,0.25)';

                  // 可选：几秒后淡出（如果你不想自动消失，注释掉即可）
                  if (window.__webenv_time_hide) clearTimeout(window.__webenv_time_hide);
                  window.__webenv_time_hide = setTimeout(() => { el.style.opacity = '0'; }, 2500);
                }
                """,
                float(final_seconds), label
            )
        except Exception:
            pass

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
        vp = self.page.viewport_size
        if not vp:
            vp = await self.page.evaluate(
                "() => ({ width: window.innerWidth, height: window.innerHeight })"
            )
        return (int(vp["width"] // 2), int(vp["height"] // 2))

    async def _ensure_mouse_origin(self):
        if self._mouse_x is None or self._mouse_y is None:
            cx, cy = await self._get_viewport_center()
            await self.page.mouse.move(cx, cy)
            await self._ensure_cursor_ready()
            await self._set_cursor_pos(cx, cy)
            self._mouse_x, self._mouse_y = cx, cy

    async def _center_of_xpath(self, xpath: str):
        loc = self.page.locator(f"xpath={xpath}")
        await loc.scroll_into_view_if_needed()
        box = await loc.bounding_box()
        if not box:
            return None
        return (box["x"] + box["width"]/2, box["y"] + box["height"]/2)

    async def _move_mouse_uniform_to(self, x: float, y: float, duration_ms: int = 600, fps: int = 90):
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        sx, sy = float(self._mouse_x), float(self._mouse_y)
        tx, ty = float(x), float(y)
        frames = max(1, round((duration_ms/1000) * fps))

        await self.page.mouse.move(sx, sy)

        for i in range(1, frames+1):
            t = i / frames
            cx = sx + (tx - sx) * t
            cy = sy + (ty - sy) * t
            await self.page.mouse.move(cx, cy, steps=1)
            await self._set_cursor_pos(cx, cy)
            await asyncio.sleep(1 / fps)

        await self.page.mouse.move(tx, ty, steps=1)
        await self._set_cursor_pos(tx, ty)
        self._mouse_x, self._mouse_y = tx, ty

    async def setup(self):
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch(headless=False)
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )
        self.page = await self.context.new_page()
        await self._inject_cursor_overlay()
        cx, cy = await self._get_viewport_center()
        await self.page.mouse.move(cx, cy)
        await self._set_cursor_pos(cx, cy)
        self._mouse_x, self._mouse_y = cx, cy

    async def _goto_current_target(self):
        if self.is_url:
            await self.page.goto(self.html_file, timeout=60000)
        else:
            await self.page.goto(f"file:///{os.path.abspath(self.html_file)}", timeout=60000)
        self._last_url = self.page.url

    async def reset(self):
        if self.page is None:
            await self.setup()
        await self.page.set_viewport_size({
            "width": self.viewport_width,
            "height": self.viewport_height
        })
        await self._goto_current_target()
        self.click_history = []

        # ✅ 新增：重注入并归位到中心
        await self._inject_cursor_overlay()
        cx, cy = await self._get_viewport_center()
        await self.page.mouse.move(cx, cy)
        await self._set_cursor_pos(cx, cy)
        self._mouse_x, self._mouse_y = cx, cy


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
        await asyncio.sleep(1)
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
        return obs

    async def render_sized(self, jpeg_quality=95):
        obs = await self.render()
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
            # 1) 确保元素可见并取中心
            await elem.scroll_into_view_if_needed()
            center = await self._center_of_xpath(xpath)
            if not center:
                print(f"[CLICK] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center

            # 2) 在 window 上预置一次性 Promise（精准到“点击后一帧”）
            await self.page.evaluate("""
                () => {
                    // 创建/重置一次性等待器
                    let resolveFn;
                    window.__webenvClickFrame = {
                        ts: null,
                        promise: new Promise(res => { resolveFn = res; })
                    };
                    // 捕获阶段监听 click，确保尽可能早地捕获到
                    const handler = () => {
                        try {
                            requestAnimationFrame((t) => {
                                // 记录点击完成的那一帧的 rAF 时间戳
                                window.__webenvClickFrame.ts = t;
                                // 触发 promise 完成
                                resolveFn(t);
                            });
                        } catch (e) {
                            // 兜底：即便 rAF 抛错也要 resolve，避免悬挂
                            try { resolveFn(performance.now()); } catch(_) {}
                        }
                    };
                    window.addEventListener('click', handler, { once: true, capture: true });
                }
            """)

            # 3) 平滑移动到目标位置
            await self._move_mouse_uniform_to(tx, ty, duration_ms=600, fps=90)

            # # 4) 可能发生的导航（无导航也不会卡），给足一点时间
            # nav_task = asyncio.create_task(
            #     self.page.wait_for_event("framenavigated", timeout=10000)
            # )

            # 5) 执行真实鼠标点击（down→up）
            await self.page.mouse.click(tx, ty)

            # 6) 等待“点击后一帧”的 Promise（精准到帧）
            try:
                await self.page.evaluate("() => window.__webenvClickFrame && window.__webenvClickFrame.promise")
            except Exception:
                # 若立即导航导致旧页面无 rAF，promise 可能被丢弃：交由导航/加载兜底
                pass

            # 7) 稳定性等待：若发生了导航，等加载；否则等 networkidle（把异步请求也计入时间）
            nav_happened = False
            # try:
            #     await nav_task
            #     nav_happened = True
            # except Exception:
            #     nav_happened = False
                
            if nav_happened:
                # 导航已发生：再等到 load（你也可以换成 'domcontentloaded' 或 'networkidle'）
                try:
                    await self.page.wait_for_load_state("load", timeout=10000)
                except Exception:
                    # 某些 SPA 只会 'domcontentloaded'，再兜底试 networkidle
                    try:
                        await self.page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
            else:
                # 未导航：等网络空闲一段时间，涵盖 XHR/fetch 的处理
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    # 没有显式的 load state 变化也没关系，再退一步至少让一帧过去
                    await self.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")

            # 8) 最后对齐微任务，记录并返回
            await asyncio.sleep(0)
            try:
                self._last_url = self.page.url
            except Exception:
                pass
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

        try:
            self._last_url = self.page.url
        except Exception:
            self._last_url = None

        self._video_tmpdir = tempfile.mkdtemp(prefix="webenv_video_")

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
            record_size = {"width": self.viewport_width, "height": self.viewport_height}
        self._t0 = time.perf_counter()  # ← 统一基准：页面“准备好之后”的服务端单调时钟
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1,
            record_video_dir=self._video_tmpdir,
            record_video_size=record_size
        )
        self.page = await self.context.new_page()
    
        if self._last_url:
            await self.page.goto(self._last_url, timeout=60000)
        else:
            await self._goto_current_target()

        await self._inject_cursor_overlay()
        cx, cy = await self._get_viewport_center()
        await self.page.mouse.move(cx, cy)
        await self._set_cursor_pos(cx, cy)
        self._mouse_x, self._mouse_y = cx, cy

        self.recording = True
        print(f"[VIDEO] Started. dir={self._video_tmpdir}, size={record_size}")
        return True

    async def end_video(self) -> Optional[Dict[str, str]]:
        """
        结束录制：关闭 page/context 获取视频文件，读入 bytes 返回；
        然后自动恢复到非录制 context/page（保持原 URL），以便继续使用。
        返回: {"content_type": "...", "video_b64": "..."} 或 None
        """
        if not self.recording:
            print("[VIDEO] Not recording.")
            return None

        try:
            current_url = self.page.url
        except Exception:
            current_url = self._last_url or None

        video_bytes: Optional[bytes] = None
        output: Optional[Dict[str, str]] = None

        try:
            try:
                await self.page.close()
            except Exception as e:
                print(f"[VIDEO] page.close() error: {e}")

            try:
                if self.context:
                    await self.context.close()
            except Exception as e:
                print(f"[VIDEO] context.close() error: {e}")

            webm_path = None
            if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                candidates = [os.path.join(self._video_tmpdir, f) for f in os.listdir(self._video_tmpdir)]
                candidates = [p for p in candidates if os.path.isfile(p)]
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    webm_path = candidates[0]

            if not webm_path:
                print("[VIDEO] No video file found in temp dir.")
                return None

            # 先尝试转 mp4；失败则回退 webm
            mp4_b64 = None
            try:
                mp4_path = webm_path.replace(".webm", ".mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-i", webm_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    mp4_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(mp4_path, "rb") as f:
                    mp4_b64 = base64.b64encode(f.read()).decode("ascii")
                output = {"content_type": "video/mp4", "video_b64": mp4_b64}
            except Exception as e:
                print(f"[VIDEO] ffmpeg convert failed, fallback to webm. err={e}")
                with open(webm_path, "rb") as f:
                    webm_b64 = base64.b64encode(f.read()).decode("ascii")
                output = {"content_type": "video/webm", "video_b64": webm_b64}

        finally:
            try:
                if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                    shutil.rmtree(self._video_tmpdir, ignore_errors=True)
            except Exception:
                pass
            self._video_tmpdir = None
            self.recording = False

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
            await self._inject_cursor_overlay()
            cx, cy = await self._get_viewport_center()
            await self.page.mouse.move(cx, cy)
            await self._set_cursor_pos(cx, cy)
            self._mouse_x, self._mouse_y = cx, cy

        # ### NEW: 将本段保存到实例列表
        if output:
            self.video_segments.append(output)
        return output

    # ### NEW: 获取当前已保存的视频段列表（浅拷贝）
    def get_video_segments(self) -> List[Dict[str, str]]:
        return list(self.video_segments)

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
        html_file = sys.argv[1]
    else:
        html_file = "page.html"
    env = WebHtmlGymEnv(html_file=html_file)
    await env.reset()
    print(f"WebHtmlGymEnv启动并初始化完成。加载html: {html_file}")

@app.post("/click")
async def api_click(req: ClickRequest):
    global env
    await env.start_live_timer(label="Click")
    start = time.perf_counter()
    result = await env.click(req.id, req.id2xpath)
    elapsed_sec = round(time.perf_counter() - start, 3)

    now_mono = time.perf_counter()
    if env._first_event_monotonic is None:
        env._first_event_monotonic = now_mono  # 记录第一步发生的绝对时刻

    await env.stop_live_timer(elapsed_sec, label="Click")
    return {
        "result": result,
        "time": elapsed_sec,               # 本次操作自身耗时
        "t0": env._t0            # 也回给你，方便上层不用单独保存
    }

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

# ===== 视频录制 API =====
@app.post("/start_video")
async def api_start_video(req: StartVideoRequest):
    global env
    ok = await env.start_video(width=req.width, height=req.height)
    return {"result": ok, "recording": env.recording, "t0": env._t0}

@app.post("/end_video")
async def api_end_video():
    """
    结束当前录制，将该段视频追加进实例列表，并返回本段（同时列表里也会有）。
    """
    global env
    output = await env.end_video()  # {"content_type": "...", "video_b64": "..."} 或 None
    if not output:
        return {"result": False, "error": "no_video"}
    # 返回本段，同时保证它已经被写入 env.video_segments
    return {"result": True, **output, "segments_count": len(env.video_segments)}

# ===== NEW: “merge_video” 接口（按你的要求：返回列表本身）=====
@app.get("/merge_video")
async def api_merge_video():
    """
    当前设计：按照你的描述“返回这个列表”，并未做真正码流级合并。
    需要合成一个单独 mp4 时，可再加 ffmpeg 的 concat 逻辑。
    """
    global env
    videos = env.get_video_segments()
    return {
        "result": True,
        "count": len(videos),
        "videos": videos  # 每个元素: {"content_type": "...", "video_b64": "..."}
    }

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
