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

from fastapi import HTTPException
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

        # === NEW: 保存所有段的视频（b64 列表）===
        # 每个元素：{"content_type": "video/mp4" | "video/webm", "video_b64": "..."}
        self.video_segments: List[Dict[str, str]] = []   # ### NEW
            # 在 __init__ 里加：
        self._t0: Optional[float] = None
        self._first_event_monotonic: Optional[float] = None  # 可用于后面物理裁切

    async def _debug_bar_init(self):
        """
        确保页面底部有一条常驻蓝色调试条。
        采用“顽固模式”：如果 DOM 没准备好，会轮询直到注入成功；
        并挂载一个守护进程，防止它被页面删除。
        """
        print("[BAR] INIT (Robust)")
        js = """
        () => {
            // 唯一 ID
            const BAR_ID = '___webenv_debugbar';
            
            // 定义核心注入逻辑
            const ensureBar = () => {
                // 1. 寻找挂载点：首选 body，次选 documentElement
                const mountPoint = document.body || document.documentElement;
                if (!mountPoint) {
                    // 如果连根节点都没有，说明太早了，下一帧再试
                    return false;
                }

                let bar = document.getElementById(BAR_ID);
                
                // 2. 如果不存在，创建它
                if (!bar) {
                    bar = document.createElement('div');
                    bar.id = BAR_ID;
                    // 初始挂载
                    try {
                        mountPoint.appendChild(bar);
                    } catch (e) {
                        return false; 
                    }
                } else {
                    // 如果存在，但父节点不是当前的挂载点（可能被移除了，或者在 detached DOM 里），重新挂载
                    if (bar.parentNode !== mountPoint) {
                        mountPoint.appendChild(bar);
                    }
                }

                // 3. 强制赋予“霸权”样式 (使用 !important 防止被覆盖)
                // 每次检查都重新赋值，防止页面 JS 修改了它的样式
                bar.style.cssText = `
                    position: fixed !important;
                    left: 0 !important;
                    bottom: 0 !important;
                    width: 100vw !important;
                    height: 10px !important;
                    background-color: #0044ff !important;
                    z-index: 2147483647 !important; /* Playwright 也就这么大，拉满 */
                    pointer-events: none !important;
                    box-shadow: 0 -1px 4px rgba(0,0,0,0.25) !important;
                    transition: background-color 80ms ease-out !important;
                    display: block !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                `;
                
                return true;
            };

            // 4. 初始化执行
            const success = ensureBar();

            // 5. 设置守护逻辑 (如果在 window 上已经有了定时器，先清理，避免重复)
            if (window.__webenvDebugBarTimer) {
                clearInterval(window.__webenvDebugBarTimer);
            }

            // 每 500ms 检查一次：
            // 1. 如果 bar 被删了，重新加回来
            // 2. 如果 styles 被改了，强制改回来
            window.__webenvDebugBarTimer = setInterval(() => {
                ensureBar();
            }, 500);

            // 如果第一次失败了（DOM未就绪），用 requestAnimationFrame 疯狂重试直到成功
            if (!success) {
                const retryLoop = () => {
                    if (ensureBar()) {
                        // 成功后退出循环，交给 setInterval 守护
                        return;
                    }
                    requestAnimationFrame(retryLoop);
                };
                requestAnimationFrame(retryLoop);
            }
            
            // 初始化状态管理（用于脉冲变色）
            if (!window.__webenvDebugBarState) {
                window.__webenvDebugBarState = { timer: null };
            }

            return true;
        }
        """
        try:
            if not self.page.is_closed():
                await self.page.evaluate(js)
        except Exception as e:
            # 只有当真的无法执行 JS 时才报错
            if "Target closed" not in str(e) and "Execution context was destroyed" not in str(e):
                print(f"[debugbar] init error: {e}")

    async def _debug_bar_pulse(self, label: str = "EVENT", active_ms: int = 160):
        """
        将底部调试条短暂变成红色。
        采用“高频霸权”模式：在 active_ms 期间，每一帧都强制写入红色样式，
        以覆盖 _debug_bar_init 中守护进程的重置干扰。
        """
        js = """
        ({ label, activeMs }) => {
            const BAR_ID = '___webenv_debugbar';
            let bar = document.getElementById(BAR_ID);

            // 1. 如果 bar 不见了，尝试复活它 (为了稳健)
            if (!bar) {
                const mountPoint = document.body || document.documentElement;
                if (!mountPoint) return; 
                bar = document.createElement('div');
                bar.id = BAR_ID;
                try { mountPoint.appendChild(bar); } catch(e){}
            }

            // 2. 定义样式模板 (必须与 init 中的 !important 级别保持一致，否则会覆盖失败)
            // 只有 background-color 是变量
            const getStyle = (color) => `
                position: fixed !important;
                left: 0 !important;
                bottom: 0 !important;
                width: 100vw !important;
                height: 10px !important;
                background-color: ${color} !important;
                z-index: 2147483647 !important;
                pointer-events: none !important;
                box-shadow: 0 -1px 4px rgba(0,0,0,0.25) !important;
                transition: background-color 80ms ease-out !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            `;

            const RED_STYLE = getStyle('#d93025');
            const BLUE_STYLE = getStyle('#0044ff');

            // 3. 启动霸权循环
            const start = performance.now();
            
            const frame = () => {
                const now = performance.now();
                if (now - start >= activeMs) {
                    // 时间到：恢复蓝色，并退出循环
                    if (bar) bar.style.cssText = BLUE_STYLE;
                    return;
                }
                
                // 时间没到：强制设为红色
                // 即使 init 的守护进程在某一帧把它改蓝了，下一帧我们立马改回来
                if (bar) bar.style.cssText = RED_STYLE;
                
                requestAnimationFrame(frame);
            };

            // 立即启动
            frame();
        }
        """
        try:
            if not self.page.is_closed():
                await self.page.evaluate(js, {"label": label, "activeMs": int(active_ms)})
        except Exception as e:
            if "Target closed" not in str(e) and "Execution context was destroyed" not in str(e):
                print(f"[debugbar] pulse error: {e}")

    async def _inject_cursor_overlay(self):
            """
            在页面注入 PNG 鼠标图案。
            增加安全检查：如果 document.documentElement 不存在则跳过。
            """
            import base64
            import pathlib

            png_path = pathlib.Path(__file__).parent / "cursor.png"
            if not png_path.exists():
                # 静默失败或只打印一次，避免干扰
                return

            with open(png_path, "rb") as f:
                png_b64 = base64.b64encode(f.read()).decode("ascii")

            js = """
            (png) => {
            // 1. 安全检查
            if (!document.documentElement) return false;

            let el = document.getElementById('___webenv_mouse');
            if (!el) {
                el = document.createElement('div');
                el.id = '___webenv_mouse';
                
                // 2. 尝试挂载
                try {
                    document.documentElement.appendChild(el);
                } catch(e) {
                    return false;
                }
            }

            Object.assign(el.style, {
                position: 'fixed',
                left: '0px',
                top: '0px',
                width: '24px',
                height: '24px',
                backgroundImage: `url("data:image/png;base64,${png}")`,
                backgroundSize: 'contain',
                backgroundRepeat: 'no-repeat',
                pointerEvents: 'none',
                zIndex: '2147483647',
                transform: 'translate(-9999px,-9999px)'
            });

            return true;
            }
            """

            try:
                if not self.page.is_closed():
                    await self.page.evaluate(js, png_b64)
            except Exception as e:
                if "Target closed" not in str(e) and "Execution context was destroyed" not in str(e):
                    print(f"[cursor inject error] {e}")

    async def _element_view_info(self, elem):
        """
        返回元素相对 viewport 的位置 + viewport 尺寸。
        """
        return await elem.evaluate(
            """el => {
                const rect = el.getBoundingClientRect();
                return {
                    top: rect.top,
                    bottom: rect.bottom,
                    left: rect.left,
                    right: rect.right,
                    width: rect.width,
                    height: rect.height,
                    vh: window.innerHeight,
                    vw: window.innerWidth
                };
            }"""
        )

    async def _smooth_scroll_into_view(
        self,
        elem,
        margin_px: int = 20,
        step_px: int = 320,
        max_rounds: int = 15,
        mouse_speed: float = 1200.0,
        fps: int = 90
    ):
        """
        丝滑版 scroll-into-view：
        - 如果元素已经在视口内：直接返回其中心 (cx, cy)
        - 否则：
          * 根据元素在上/下方，决定滚动方向
          * 把鼠标匀速移动到视口上/下方的一个“锚点”
          * 用鼠标滚轮一步一步滚动页面
          * 每次滚动后重新获取元素 rect，直到进入视口或超出 max_rounds

        返回:
          (cx, cy) 屏幕坐标（在 viewport 内）
          或 None（多次滚动仍不可见）
        """
        # 先拿一次 rect
        info = await self._element_view_info(elem)
        vh, vw = info["vh"], info["vw"]

        def is_visible(inf):
            return (
                inf["bottom"] > margin_px and
                inf["top"] < inf["vh"] - margin_px and
                inf["height"] > 0 and inf["width"] > 0
            )

        # 如果已经可见，直接返回中心
        if is_visible(info):
            cx = info["left"] + info["width"] / 2
            cy = info["top"] + info["height"] / 2
            # 限制在视口里
            cx = max(1, min(vw - 1, cx))
            cy = max(1, min(vh - 1, cy))
            return (cx, cy)

        # 元素不在视口内：开始滚动
        for _ in range(max_rounds):
            info = await self._element_view_info(elem)
            vh, vw = info["vh"], info["vw"]

            if is_visible(info):
                cx = info["left"] + info["width"] / 2
                cy = info["top"] + info["height"] / 2
                cx = max(1, min(vw - 1, cx))
                cy = max(1, min(vh - 1, cy))
                return (cx, cy)

            # 决定滚动方向：向下 / 向上
            #   元素在下面：rect.top > vh -> 向下
            #   元素在上面：rect.bottom < 0 -> 向上
            if info["top"] >= vh - margin_px:
                direction = 1   # 往下滚
            elif info["bottom"] <= margin_px:
                direction = -1  # 往上滚
            elif info["top"] < margin_px:
                direction = -1
            elif info["bottom"] > vh - margin_px:
                direction = 1
            else:
                direction = 0

            if direction == 0:
                # 理论上不会进来，保险兜底
                cx = info["left"] + info["width"] / 2
                cy = info["top"] + info["height"] / 2
                cx = max(1, min(vw - 1, cx))
                cy = max(1, min(vh - 1, cy))
                return (cx, cy)

            # 鼠标的“滚动锚点”：
            #   向下滚：鼠标移到视口偏下 0.8 * vh
            #   向上滚：鼠标移到视口偏上 0.2 * vh
            anchor_y = int(vh * (0.8 if direction > 0 else 0.2))
            # 横向尽量靠近元素中心，但限制在一点安全区内
            target_cx = info["left"] + info["width"] / 2
            target_cx = max(40, min(vw - 40, target_cx))

            # ✨ 用你已有的匀速鼠标移动过去
            await self._move_mouse_uniform_to(
                target_cx,
                anchor_y,
                speed_px_per_sec=mouse_speed,
                fps=fps
            )

            # ✨ 用鼠标滚轮一步滚动页面
            await self.page.mouse.wheel(0, direction * step_px)
            # 给浏览器一点时间渲染
            await asyncio.sleep(0.12)

        # 滚了很多次还不可见，就放弃
        return None


    async def _cursor_exists(self) -> bool:
        try:
            return await self.page.evaluate("()=>!!document.getElementById('___webenv_mouse')")
        except Exception:
            return False
        
    async def _set_cursor_pos(self, x: float, y: float):
        """移动 PNG 鼠标箭头（不触发真实鼠标事件）。"""
        await self.page.evaluate(
            """({x, y}) => {
                const el = document.getElementById('___webenv_mouse');
                if (!el) return;

                // PNG 的箭头尖端一般在左上角，不需要偏移。如需偏移可以调下面数字
                el.style.transform = `translate(${x}px, ${y}px)`;
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

    async def _move_mouse_uniform_to(
        self,
        x: float,
        y: float,
        speed_px_per_sec: float = 1200.0,  # 鼠标移动速度（像素/秒），可以自己调
        fps: int = 90
    ):
        """
        以恒定速度移动鼠标：时间 = 距离 / 速度
        speed_px_per_sec 越大，移动越快；越小，移动越慢。
        """
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        sx, sy = float(self._mouse_x), float(self._mouse_y)
        tx, ty = float(x), float(y)

        # 计算欧几里得距离
        dist = math.hypot(tx - sx, ty - sy)

        # 防止距离太小时间为 0，这里给一个下限（比如 0.08s）
        min_duration = 0.08
        if dist <= 1e-3:
            duration_sec = min_duration
        else:
            duration_sec = max(dist / speed_px_per_sec, min_duration)

        # 根据时长和 fps 计算需要的帧数
        frames = max(1, round(duration_sec * fps))

        # 确保 Playwright 的“真实鼠标”从起点开始
        await self.page.mouse.move(sx, sy)

        for i in range(1, frames + 1):
            t = i / frames
            cx = sx + (tx - sx) * t
            cy = sy + (ty - sy) * t
            await self.page.mouse.move(cx, cy, steps=1)
            await self._set_cursor_pos(cx, cy)
            await asyncio.sleep(1 / fps)

        # 最后一帧对齐到目标点，防止累计误差
        await self.page.mouse.move(tx, ty, steps=1)
        await self._set_cursor_pos(tx, ty)
        self._mouse_x, self._mouse_y = tx, ty

    async def _reload_tools_if_needed(self, start_url: str):
        """
        检测页面是否发生了导航。如果发生了，必须先等待 DOM 就绪，再注入工具。
        """
        try:
            # 1. 简单的现状快照
            current_url = self.page.url
            
            # 检查1: JS环境是否重置 (强刷新)
            # 这里的 try-except 很重要，因为如果页面正在跳转中，evaluate 可能会直接抛出 "Execution context destroyed"
            try:
                is_context_lost = await self.page.evaluate("() => typeof window.__webenvClickFrame === 'undefined'")
            except Exception:
                # 如果 evaluate 报错，说明上下文已经炸了，肯定发生了导航
                is_context_lost = True

            # 检查2: URL 是否变化 (SPA / 跳转)
            is_url_changed = current_url != start_url

            if is_context_lost or is_url_changed:
                print(f"[NAV] Navigation detected (ContextLost={is_context_lost}, URLChanged={is_url_changed}). Waiting for DOM...")
                
                # === 🌟 核心修复点：强制等待页面加载 🌟 ===
                # 在注入之前，必须死等 DOM 结构出来。
                # 'domcontentloaded' 比 'load' 快，但足够保证 document.documentElement 存在
                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception as e:
                    print(f"[NAV] Wait for domcontentloaded warning: {e}")
                    # 即使超时，也稍微硬等一下，防止空白
                    await asyncio.sleep(0.5)

                # 额外等待：针对某些由 JS 渲染的 SPA，DOM 出来后可能还需要一点点时间
                # await asyncio.sleep(0.2) 

                # === 现在注入才是安全的 ===
                await self._inject_cursor_overlay()
                await self._debug_bar_init()
                
                # 尝试恢复鼠标位置
                if self._mouse_x is not None and self._mouse_y is not None:
                    await self._set_cursor_pos(self._mouse_x, self._mouse_y)
                
                return True
        except Exception as e:
            print(f"[NAV] Error during reload checks: {e}")
            # 兜底：如果出错，假设发生了导航，并强制等待后注入
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                await self._inject_cursor_overlay()
                await self._debug_bar_init()
            except:
                pass
            return True
        
        return False

    async def setup(self):
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )
        self.page = await self.context.new_page()
        
        # --- 这里的顺序很重要 ---
        # 1. 先去加载一个页面 (如果是本地文件或 about:blank)
        # 2. 等待 DOM 准备好
        # 3. 再注入工具
        
        # 你的 _goto_current_target 可能指向 file:// 或 http://
        # 如果是初始状态，这里建议先不做 _goto，或者在 reset 里做
        # 但为了保持兼容，如果你这里要注入，必须确保有页面
        
        # 示例：如果不跳转，page 是 about:blank，有时 dom 也不全
        # 所以最好移到 reset() 里统一处理，这里只做基础初始化
        
        # 为了修复你的报错，建议 setup 简单化，把注入放到 reset
        pass 

    async def reset(self):
        if self.page is None:
            await self.setup()
        
        # 1. 确保尺寸
        await self.page.set_viewport_size({
            "width": self.viewport_width,
            "height": self.viewport_height
        })
        
        # 2. 导航到目标页面
        await self._goto_current_target()
        
        # 3. [关键] 等待 DOM 结构加载完成
        # 这一步能保证 document.documentElement 存在
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass # 超时也继续尝试，反正我们加了 JS 空值检查

        self.click_history = []

        # 4. 此时注入才是安全的
        await self._inject_cursor_overlay()
        await self._debug_bar_init()

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

    async def load_url(self, url: str):
        self.html_file = url
        self.is_url = True
        if self.page is None:
            await self.setup()
        await self.page.goto(url, timeout=60000)
        self._last_url = self.page.url
        self.click_history = []
        print(f"已加载新页面: {url}")

        # 🌟 保证新 URL 里也有调试条
        await self._debug_bar_init()

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

    async def render_viewport(self, mode="human"):
        # 不等待页面扩大，不自动调整大小
        # 直接截图当前浏览器视口
        obs = await self.page.screenshot(full_page=False)
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
            # === 0) 记录点击前的 URL ===
            start_url = self.page.url

            # 1) 确保元素可见并取中心
            center = await self._smooth_scroll_into_view(elem)
            if not center:
                print(f"[CLICK] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center

            # 2) 注入 Promise 监听器
            await self.page.evaluate("""
                () => {
                    let resolveFn;
                    window.__webenvClickFrame = {
                        ts: null,
                        promise: new Promise(res => { resolveFn = res; })
                    };
                    const handler = () => {
                        try {
                            requestAnimationFrame((t) => {
                                window.__webenvClickFrame.ts = t;
                                resolveFn(t);
                            });
                        } catch (e) {
                            try { resolveFn(performance.now()); } catch(_) {}
                        }
                    };
                    window.addEventListener('click', handler, { once: true, capture: true });
                }
            """)
            
            t_start = time.perf_counter()
            
            # 3) 移动鼠标
            await self._move_mouse_uniform_to(tx, ty)

            # 4) 视觉反馈
            await self._debug_bar_pulse("CLICK", active_ms=160)
            
            # 5) ⚡️ 真实点击 ⚡️
            # 注意：如果点击触发强跳转，这一行可能会抛出 disconnect 错误，需要忽略
            try:
                await self.page.mouse.click(tx, ty)
            except Exception:
                pass 

            # 6) 尝试等待点击动画帧（如果页面没立即跳的话）
            try:
                await self.page.evaluate("() => window.__webenvClickFrame && window.__webenvClickFrame.promise")
            except Exception:
                # 这一步报错通常意味着页面已经开始跳转了，Context Destroyed
                pass

            # === 7) 关键点：给浏览器一点时间去“开始”跳转 ===
            # 如果不 sleep，代码跑太快，URL 可能还没变，context 也没销毁
            await asyncio.sleep(0.1)

            # === 8) 智能等待与重载 ===
            # 这里调用上面修复过的函数。
            # 如果它检测到 URL 变了，或者 Context 没了，它会在内部执行 wait_for_load_state('domcontentloaded')
            # 从而实现“导航后等待”。
            did_reload = await self._reload_tools_if_needed(start_url)

            # 如果没检测到导航（did_reload 为 False），说明是普通点击，
            # 我们只需要等待网络空闲即可，不用等 heavy load
            if not did_reload:
                try:
                    # 原地操作（如展开下拉框），等待网络空闲或至少一帧
                    await self.page.wait_for_load_state("networkidle", timeout=2000)
                except Exception:
                    await asyncio.sleep(0.05)
            
            # 9) 收尾
            try:
                self._last_url = self.page.url
            except Exception:
                pass
            self.click_history.append(("click", unique_id))
            return True, t_start

        except Exception as e:
            print(f"[CLICK] Failed click: {e}")
            return False, t_start
        
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
            # === 0) 记录操作前的状态 ===
            start_url = self.page.url
            # 确保标记变量存在，防止 evaluate 报错
            await self.page.evaluate("() => { if(!window.__webenvClickFrame) window.__webenvClickFrame = {}; }")

            tag = await elem.evaluate("el => el.tagName.toLowerCase()")
            if tag not in ["input", "textarea"]:
                print(f"[ENTER] Element id={unique_id} is {tag}, not input/textarea, cannot enter text.")
                return False

            # 确保可见并取中心点
            center = await self._smooth_scroll_into_view(elem)
            if not center:
                print(f"[ENTER] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center

            # 平滑移动鼠标到目标位置
            await self._move_mouse_uniform_to(tx, ty)

            # 点击元素以聚焦
            await self._debug_bar_pulse("ENTER", active_ms=160)
            await self.page.mouse.click(tx, ty)

            # 判断是否为数字输入框
            is_number_input = await elem.evaluate(
                """el => {
                    if (el.tagName.toLowerCase() === 'input' && el.type && el.type.toLowerCase() === 'number') return true;
                    if (el.pattern && el.pattern.match(/^\\d+$/)) return true;
                    if (el.inputMode && el.inputMode.toLowerCase().includes('numeric')) return true;
                    if ((el.className || '').match(/(number|num)/i)) return true;
                    return false;
                }"""
            )

            # 处理数字输入内容
            fill_content = str(content)
            if is_number_input:
                try:
                    fill_content = str(float(content))
                except Exception:
                    m = re.search(r"[-+]?\d*\\.?\\d+", str(content))
                    fill_content = str(float(m.group())) if m else "0"

            # 填充内容
            await elem.fill(fill_content)

            # === 导航检测 ===
            await asyncio.sleep(0.1) # 缓冲
            await self._reload_tools_if_needed(start_url) # 内部包含 wait_for_load_state
            # 记录历史
            try:
                self._last_url = self.page.url
            except Exception:
                pass
            self.click_history.append(("enter", unique_id, fill_content))
            
            return True

        except Exception as e:
            print(f"[ENTER] Failed to enter text: {e}")
            return False
        
    async def select(self, unique_id: Any, value: Any, id2xpath: Dict[Any, str]) -> bool:
        xpath: Optional[str] = id2xpath.get(str(unique_id)) or (id2xpath.get(int(unique_id)) if isinstance(unique_id, (str, int)) else None)
        if not xpath:
            print(f"[SELECT] No xpath found for id={unique_id}")
            return False

        elem = await self.page.query_selector(f"xpath={xpath}")
        if elem is None:
            print(f"[SELECT] No element found by xpath {xpath} for id={unique_id}")
            return False

        try:
            # === 0) 记录操作前的状态 ===
            start_url = self.page.url
            
            tag = await elem.evaluate("el => el.tagName.toLowerCase()")
            center = await self._smooth_scroll_into_view(elem)
            if not center:
                print(f"[SELECT] Element not visible/bbox missing: {xpath}")
                return False
            tx, ty = center

            # 预置 Promise
            await self.page.evaluate("""
                () => {
                    let resolveFn;
                    window.__webenvClickFrame = {
                        ts: null,
                        promise: new Promise(res => { resolveFn = res; })
                    };
                    const handler = () => {
                        try {
                            requestAnimationFrame((t) => {
                                window.__webenvClickFrame.ts = t;
                                resolveFn(t);
                            });
                        } catch (e) {
                            try { resolveFn(performance.now()); } catch(_){}
                        }
                    };
                    window.addEventListener('click', handler, { once: true, capture: true });
                }
            """)

            t_start = time.perf_counter()
            # 平滑移动并真实点击
            await self._move_mouse_uniform_to(tx, ty)
            await self._debug_bar_pulse("SELECTT", active_ms=160)
            await self.page.mouse.click(tx, ty)

            # 等待“点击完成后一帧”
            try:
                await self.page.evaluate("() => window.__webenvClickFrame && window.__webenvClickFrame.promise")
            except Exception:
                pass

            # 定义一个统一的成功退出处理函数
            async def finish_action(action_key, val=None):      
                # === 核心等待 ===
                await asyncio.sleep(0.1) # 给浏览器喘息时间
                await self._reload_tools_if_needed(start_url) # 如果跳了，这里会死等 DOMContentLoaded
                
                try:
                    self._last_url = self.page.url
                except:
                    pass
                
                rec = (action_key, unique_id) if val is None else (action_key, unique_id, val)
                self.click_history.append(rec)
                return True, t_start
            # —— 根据元素类型进行选择 ——
            if tag == "select":
                try:
                    # 先尝试按 value 选
                    result = await elem.select_option(str(value))
                    if result and result != []:
                        return await finish_action("select", value)

                    # 回退：按可见文本近似匹配
                    options = await elem.evaluate(
                        "el => Array.from(el.options).map(o => ({value: o.value, text: o.text}))"
                    )
                    option_texts = [opt["text"] for opt in options]
                    closest = difflib.get_close_matches(str(value), option_texts, n=1, cutoff=0.0)
                    if not closest:
                        print(f"[SELECT] No similar option (by text) for value: {value}")
                        return False, t_start

                    closest_text = closest[0]
                    for opt in options:
                        if opt["text"] == closest_text:
                            result = await elem.select_option(opt["value"])
                            if result and result != []:
                                return await finish_action("select-closest", opt["value"])
                                
                    print("[SELECT] Failed to select even after closest-match logic.")
                    return False, t_start

                except Exception as e:
                    print(f"[SELECT] Exception during <select> handling: {e}")
                    return False, t_start

            elif tag == "input":
                input_type = (await elem.get_attribute("type")) or ""
                input_type = input_type.lower()

                if input_type == "radio":
                    # 单选框点击已经在上面完成了，直接返回
                    return await finish_action("select-radio")

                elif input_type == "date":
                    await elem.fill(str(value))
                    return await finish_action("select-date", value)

                else:
                    print(f"[SELECT] Element id={unique_id} is input but not type=date/radio, skipping.")
                    return False, t_start

            else:
                print(f"[SELECT] Element id={unique_id} is {tag}, not select/input[type=date|radio], cannot select value.")
                return False, t_start

        except Exception as e:
            print(f"[SELECT] Failed to select: {e}")
            return False, t_start

    
    # ========= 录制：核心新增 =========
    async def start_video(self, width: Optional[int] = None, height: Optional[int] = None) -> bool:
        """
        开始录制（切换到启用 record_video 的新 context/page），后续 reset/click/enter/select/load_url 等都不会中断录制。
        """
        print("正在准备启动视频录制")
        self._t0 = time.perf_counter()  # ← 统一基准：页面“准备好之后”的服务端单调时钟
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

        # === [新增] 等待 DOM ===
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass    
        await self._inject_cursor_overlay()
        await self._debug_bar_init()  # 🌟 录制用的 context 里也要有调试条

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


class LoadUrlRequest(BaseModel):
    url: str


    
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
    start = time.perf_counter()
    result= await env.click(req.id, req.id2xpath)
    elapsed_sec = round(time.perf_counter() - start, 3)
    t_start = None

    now_mono = time.perf_counter()
    if env._first_event_monotonic is None:
        env._first_event_monotonic = now_mono  # 记录第一步发生的绝对时刻

    return {"result": result, "time": start, "in_time": t_start}

@app.post("/enter")
async def api_enter(req: EnterRequest):
    global env
    start = time.perf_counter()
    result = await env.enter(req.id, req.text, req.id2xpath)
    end_t = round(time.perf_counter() - start, 3)
    t_start = None
    return {"result": result, "time": start, "in_time": t_start}

@app.post("/select")
async def api_select(req: SelectRequest):
    global env
    start = time.perf_counter()
    result = await env.select(req.id, req.value, req.id2xpath)
    t_start = None
    end_t = round(time.perf_counter() - start, 3)
    return {"result": result, "time": start, "in_time": t_start}


@app.get("/observe")
async def api_observe():
    global env
    obs_bytes = await env.render()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}

@app.get("/observe_sized")
async def api_observe_sized():
    global env
    obs_bytes = await env.render_viewport() #这里改了，应该是observe_sized
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


@app.post("/load_url")
async def api_load_url(req: LoadUrlRequest):
    """
    加载指定 URL 到当前 env 中。
    请求格式：
    POST /load_url
    {
        "url": "https://example.com"
    }
    返回：
    {
        "result": true,
        "url": "https://example.com"
    }
    """
    global env
    if not req.url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")

    try:
        # 假设 env 提供了异步的 load_url 方法
        # 如果是同步的，就去掉 await
        await env.load_url(req.url)
        return {"result": True, "url": req.url}
    except Exception as e:
        # 这里可以按需打印 log
        raise HTTPException(status_code=500, detail=f"Failed to load url: {e}")
    
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
