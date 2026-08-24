#修改代码，让init滚动速度更慢

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
import re
import tempfile
import shutil
from PIL import Image
from pydantic import BaseModel
import subprocess
from fastapi.responses import FileResponse
import time
import random
from fastapi import HTTPException
from pydantic import BaseModel


class WebHtmlGymEnv:
    metadata = {'render_modes': ['human'], "render_fps": 4}



    #INIT
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

        # === 鼠标 ===
        # 初始化为 None，表示尚未有位置记录
        self._mouse_x = None
        self._mouse_y = None

        # === NEW: 保存所有段的视频（b64 列表）===
        # 每个元素：{"content_type": "video/mp4" | "video/webm", "video_b64": "..."}
        self.video_segments: List[Dict[str, str]] = []

        self._t0: Optional[float] = None
        self._first_event_monotonic: Optional[float] = None

        # ✅ NEW: 全局互斥锁
        self._page_lock = asyncio.Lock()




    #DEBUG BAR
    async def _debug_bar_init(self):
        """
        确保页面底部有一条常驻蓝色调试条。
        """
        # print("[BAR] INIT (Robust)")
        js = """
        () => {
            const BAR_ID = '___webenv_debugbar';

            const ensureBar = () => {
                const mountPoint = document.body || document.documentElement;
                if (!mountPoint) return false;

                let bar = document.getElementById(BAR_ID);

                if (!bar) {
                    bar = document.createElement('div');
                    bar.id = BAR_ID;
                    try {
                        mountPoint.appendChild(bar);
                    } catch (e) {
                        return false;
                    }
                } else {
                    if (bar.parentNode !== mountPoint) {
                        mountPoint.appendChild(bar);
                    }
                }

                bar.style.cssText = `
                    position: fixed !important;
                    left: 0 !important;
                    bottom: 0 !important;
                    width: 100vw !important;
                    height: 10px !important;
                    background-color: #0044ff !important;
                    z-index: 2147483647 !important;
                    pointer-events: none !important;
                    box-shadow: 0 -1px 4px rgba(0,0,0,0.25) !important;
                    transition: background-color 80ms ease-out !important;
                    display: block !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                `;

                return true;
            };

            const success = ensureBar();

            if (window.__webenvDebugBarTimer) {
                clearInterval(window.__webenvDebugBarTimer);
            }

            window.__webenvDebugBarTimer = setInterval(() => {
                ensureBar();
            }, 500);

            if (!success) {
                const retryLoop = () => {
                    if (ensureBar()) return;
                    requestAnimationFrame(retryLoop);
                };
                requestAnimationFrame(retryLoop);
            }

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
            if "Target closed" not in str(e) and "Execution context was destroyed" not in str(e):
                print(f"[debugbar] init error: {e}")

    async def _debug_bar_pulse(self, label: str = "EVENT", active_ms: int = 160, color: str = "#d93025"):
        """
        将底部调试条变成指定颜色 (默认红)。
        """
        js = """
        ({ label, activeMs, color }) => {
            const BAR_ID = '___webenv_debugbar';
            let bar = document.getElementById(BAR_ID);

            if (!bar) {
                const mountPoint = document.body || document.documentElement;
                if (!mountPoint) return;
                bar = document.createElement('div');
                bar.id = BAR_ID;
                try { mountPoint.appendChild(bar); } catch(e){}
            }

            const getStyle = (c) => `
                position: fixed !important;
                left: 0 !important;
                bottom: 0 !important;
                width: 100vw !important;
                height: 10px !important;
                background-color: ${c} !important;
                z-index: 2147483647 !important;
                pointer-events: none !important;
                box-shadow: 0 -1px 4px rgba(0,0,0,0.25) !important;
                transition: background-color 80ms ease-out !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            `;

            const ACTIVE_STYLE = getStyle(color);
            const BLUE_STYLE = getStyle('#0044ff');

            const start = performance.now();

            const frame = () => {
                const now = performance.now();
                if (now - start >= activeMs) {
                    if (bar) bar.style.cssText = BLUE_STYLE;
                    return;
                }

                if (bar) bar.style.cssText = ACTIVE_STYLE;
                requestAnimationFrame(frame);
            };

            frame();
        }
        """

        try:
            if not self.page.is_closed():
                await self.page.evaluate(js, {"label": label, "activeMs": int(active_ms), "color": color})
        except Exception as e:
            if "Target closed" not in str(e) and "Execution context was destroyed" not in str(e):
                print(f"[debugbar] pulse error: {e}")

    async def _inject_cursor_overlay(self):
        """
        在页面注入 PNG 鼠标图案。
        """
        import base64
        import pathlib

        png_path = pathlib.Path(__file__).parent / "cursur_blue.png"
        if not png_path.exists():
            return

        with open(png_path, "rb") as f:
            png_b64 = base64.b64encode(f.read()).decode("ascii")

        js = """
        (png) => {
        if (!document.documentElement) return false;

        let el = document.getElementById('___webenv_mouse');
        if (!el) {
            el = document.createElement('div');
            el.id = '___webenv_mouse';
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
        info = await self._element_view_info(elem)
        vh, vw = info["vh"], info["vw"]

        def is_visible(inf):
            return (
                inf["bottom"] > margin_px and
                inf["top"] < inf["vh"] - margin_px and
                inf["height"] > 0 and inf["width"] > 0
            )

        if is_visible(info):
            cx = info["left"] + info["width"] / 2
            cy = info["top"] + info["height"] / 2
            cx = max(1, min(vw - 1, cx))
            cy = max(1, min(vh - 1, cy))
            return (cx, cy)

        for _ in range(max_rounds):
            info = await self._element_view_info(elem)
            vh, vw = info["vh"], info["vw"]

            if is_visible(info):
                cx = info["left"] + info["width"] / 2
                cy = info["top"] + info["height"] / 2
                cx = max(1, min(vw - 1, cx))
                cy = max(1, min(vh - 1, cy))
                return (cx, cy)

            if info["top"] >= vh - margin_px:
                direction = 1
            elif info["bottom"] <= margin_px:
                direction = -1
            elif info["top"] < margin_px:
                direction = -1
            elif info["bottom"] > vh - margin_px:
                direction = 1
            else:
                direction = 0

            if direction == 0:
                cx = info["left"] + info["width"] / 2
                cy = info["top"] + info["height"] / 2
                cx = max(1, min(vw - 1, cx))
                cy = max(1, min(vh - 1, cy))
                return (cx, cy)

            anchor_y = int(vh * (0.8 if direction > 0 else 0.2))
            target_cx = info["left"] + info["width"] / 2
            target_cx = max(40, min(vw - 40, target_cx))

            await self._move_mouse_uniform_to(
                target_cx,
                anchor_y,
                speed_px_per_sec=mouse_speed,
                fps=fps
            )

            await self.page.mouse.wheel(0, direction * step_px)
            await asyncio.sleep(0.12)

        return None

    async def _cursor_exists(self) -> bool:
        try:
            return await self.page.evaluate("""() => {
                const el = document.getElementById('___webenv_mouse');
                // 1. 检查元素是否被找到
                if (!el) return false;
                
                // 2. 检查元素是否真正连接在 DOM 树上 (处理元素被移除但 JS 引用还在的情况)
                // isConnected 是现代浏览器标准属性，Playwright 的 Chromium 环境完全支持
                return el.isConnected; 
            }""")
        except Exception:
            return False

    async def _set_cursor_pos(self, x: float, y: float):
        await self.page.evaluate(
            """({x, y}) => {
                window.__webenvMousePos = { x, y };
                const el = document.getElementById('___webenv_mouse');
                if (el) {
                    el.style.transform = `translate(${x}px, ${y}px)`;
                }
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
        speed_px_per_sec: float = 1200.0,
        fps: int = 90
    ):
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        sx, sy = float(self._mouse_x), float(self._mouse_y)
        tx, ty = float(x), float(y)

        dist = math.hypot(tx - sx, ty - sy)

        min_duration = 0.08
        if dist <= 1e-3:
            duration_sec = min_duration
        else:
            duration_sec = max(dist / speed_px_per_sec, min_duration)

        frames = max(1, round(duration_sec * fps))

        await self.page.mouse.move(sx, sy)

        for i in range(1, frames + 1):
            t = i / frames
            cx = sx + (tx - sx) * t
            cy = sy + (ty - sy) * t
            await self.page.mouse.move(cx, cy, steps=1)
            await self._set_cursor_pos(cx, cy)
            await asyncio.sleep(1 / fps)

        await self.page.mouse.move(tx, ty, steps=1)
        await self._set_cursor_pos(tx, ty)
        self._mouse_x, self._mouse_y = tx, ty

    async def _reload_tools_if_needed(self, start_url: str):
        try:
            current_url = self.page.url

            try:
                is_context_lost = await self.page.evaluate("() => typeof window.__webenvClickFrame === 'undefined'")
            except Exception:
                is_context_lost = True

            is_url_changed = current_url != start_url

            if is_context_lost or is_url_changed:
                print(f"[NAV] Navigation detected (ContextLost={is_context_lost}, URLChanged={is_url_changed}). Waiting for DOM...")

                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception as e:
                    print(f"[NAV] Wait for domcontentloaded warning: {e}")
                    await asyncio.sleep(0.5)

                await self._debug_bar_init()
                await self._inject_cursor_overlay()

                # 这里保持原样，因为导航后可能需要重置到中心，
                # 但如果你想在普通点击跳转后也保持鼠标位置，可以修改这里。
                # 目前保持 "SETTING TO CENTER" 以作为明确的视觉提示页面变了。
                centX, centY = await self._get_viewport_center()
                # print("[MOUSE] SETTING TO CENTER")
                await self._set_cursor_pos(centX, centY)
                self._mouse_x, self._mouse_y = centX, centY
                return True
        except Exception as e:
            print(f"[NAV] Error during reload checks: {e}")
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                await self._inject_cursor_overlay()
                await self._debug_bar_init()
            except:
                pass
            return True

        return False

    def _ease_out_cubic(self, t: float) -> float:
        """
        比 Quartic 更 '冲' 一点，尾部减速时间更短，更符合急停的感觉。
        """
        return 1 - pow(1 - t, 3)

    async def _move_mouse_natural(self, target_x: float, target_y: float):
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        start_x, start_y = float(self._mouse_x), float(self._mouse_y)
        total_dist = math.hypot(target_x - start_x, target_y - start_y)

        # 1. 极短距离直接跳过，防止在那蹭来蹭去
        if total_dist < 15:
            await self.page.mouse.move(target_x, target_y)
            await self._set_cursor_pos(target_x, target_y)
            self._mouse_x, self._mouse_y = target_x, target_y
            return

        # 2. 计算过头（Overshoot）逻辑
        overshoot_amount = 0
        # 只有距离够长才触发过头，模拟快速移动惯性
        if total_dist > 200: 
            if random.random() > 0.15: # 提高过头概率
                # 过头量稍微加大，让修正动作更明显
                overshoot_amount = random.uniform(15, min(total_dist * 0.15, 80))
        
        if overshoot_amount > 0:
            dx = (target_x - start_x) / total_dist
            dy = (target_y - start_y) / total_dist
            # 角度抖动，模拟手滑
            angle_jitter = random.uniform(-0.10, 0.10)
            
            rotated_dx = dx * math.cos(angle_jitter) - dy * math.sin(angle_jitter)
            rotated_dy = dx * math.sin(angle_jitter) + dy * math.cos(angle_jitter)

            aim_x = target_x + rotated_dx * overshoot_amount
            aim_y = target_y + rotated_dy * overshoot_amount
        else:
            aim_x, aim_y = target_x, target_y

        # === 内部函数：贝塞尔移动 ===
        async def perform_bezier_move(s_x, s_y, e_x, e_y, speed_factor=1.0):
            dist = math.hypot(e_x - s_x, e_y - s_y)
            # 提高基础速度，让人感觉操作者很熟练
            base_speed = random.uniform(5000, 9000) * speed_factor
            duration = max(0.1, dist / base_speed) 
            
            # 减少步数：步数越少，最后一次移动的跨度越大，看起来“急停”感越强
            # 原来乘 120，现在乘 90，降低采样率
            steps = max(3, int(duration * 90))

            mid_x = (s_x + e_x) / 2
            mid_y = (s_y + e_y) / 2
            # 贝塞尔控制点稍微收敛一点，让轨迹更直，更像有目的性的快速移动
            offset = min(dist * 0.25, 120) * random.choice([1, -1])
            ctrl_x = mid_x + random.uniform(-offset, offset)
            ctrl_y = mid_y + random.uniform(-offset, offset)

            for i in range(steps + 1):
                t = i / steps
                # 使用 Cubic 缓动，尾部减速更晚
                eased_t = self._ease_out_cubic(t)

                u = 1 - eased_t
                tt = eased_t * eased_t
                uu = u * u

                curr_x = (uu * s_x) + (2 * u * eased_t * ctrl_x) + (tt * e_x)
                curr_y = (uu * s_y) + (2 * u * eased_t * ctrl_y) + (tt * e_y)
                
                await self.page.mouse.move(curr_x, curr_y)
                await self._set_cursor_pos(curr_x, curr_y)
                
                # 只有非常短的 sleep，保证浏览器能跟上，但不要有拖泥带水感
                await asyncio.sleep(duration / steps)
            
            return e_x, e_y

        curr_x, curr_y = start_x, start_y

        # 3. 如果距离很远，中间加一个很小的停顿或变速（模拟抬手或换姿势），但要快
        if total_dist > 1500 and random.random() > 0.4:
            stop_ratio = random.uniform(0.6, 0.8)
            stop_x = start_x + (aim_x - start_x) * stop_ratio + random.uniform(-30, 30)
            stop_y = start_y + (aim_y - start_y) * stop_ratio + random.uniform(-30, 30)
            
            # 第一段冲刺更猛 (speed_factor=1.3)
            curr_x, curr_y = await perform_bezier_move(curr_x, curr_y, stop_x, stop_y, speed_factor=1.3)
            # 极短停顿
            await asyncio.sleep(random.uniform(0.01, 0.05))

        # 4. 执行主要移动（冲向目标或过头点）
        curr_x, curr_y = await perform_bezier_move(curr_x, curr_y, aim_x, aim_y, speed_factor=1.1)

        # 5. === 修正阶段 (The Correction) ===
        # 这里是根据你的要求修改的核心：极速回弹
        dist_to_real = math.hypot(target_x - curr_x, target_y - curr_y)
        
        if dist_to_real > 3:
            # 反应时间：人类意识到过头后的反应时间。
            # 想要“极快修正”，这里的 sleep 必须非常短，模拟下意识的肌肉反射。
            await asyncio.sleep(random.uniform(0.02, 0.06)) 
            
            # 修正不需要复杂的 ease-out，因为这是一个极其短促的动作（Snap）。
            # 步数极少 (2-4步)，直接拉回去。
            correction_steps = random.randint(2, 4)
            
            for i in range(1, correction_steps + 1):
                t = i / correction_steps
                # 使用简单的线性插值或极弱的缓动，显得干脆利落
                # 越接近 1，动作越线性。这里稍微给一点点曲线，防止看起来像瞬移。
                t_linear = t 
                
                final_x = curr_x + (target_x - curr_x) * t_linear
                final_y = curr_y + (target_y - curr_y) * t_linear
                
                await self.page.mouse.move(final_x, final_y)
                await self._set_cursor_pos(final_x, final_y)
                
                # 几乎不等待，利用代码执行本身的微小延迟即可
                # 如果必须要加，加一个极其微小的值
                if i < correction_steps:
                    await asyncio.sleep(0.002)

        # 确保最终精准落在目标点
        await self.page.mouse.move(target_x, target_y)
        await self._set_cursor_pos(target_x, target_y)
        self._mouse_x, self._mouse_y = target_x, target_y
    
    async def setup(self):
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )
        self.page = await self.context.new_page()
        pass

    async def reset(self):
        async with self._page_lock:
            if self.page is None:
                await self.setup()

            await self.page.set_viewport_size({
                "width": self.viewport_width,
                "height": self.viewport_height
            })

            await self._goto_current_target()

            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            except:
                pass

            self.click_history = []

            await self._inject_cursor_overlay()
            await self._debug_bar_init()

            # === ✅ MODIFIED: 记忆鼠标位置 ===
            # 如果之前有记录鼠标位置，则使用之前的坐标
            # 如果是第一次运行 (None)，则使用屏幕中心
            if self._mouse_x is not None and self._mouse_y is not None:
                target_x, target_y = self._mouse_x, self._mouse_y
                # 确保坐标在当前视口范围内 (防止 reset 改变分辨率后鼠标越界)
                target_x = max(0, min(self.viewport_width, target_x))
                target_y = max(0, min(self.viewport_height, target_y))
            else:
                target_x, target_y = await self._get_viewport_center()

            await self.page.mouse.move(target_x, target_y)
            await self._set_cursor_pos(target_x, target_y)
            self._mouse_x, self._mouse_y = target_x, target_y
            # === END MODIFICATION ===

    async def _goto_current_target(self):
        if self.is_url:
            await self.page.goto(self.html_file, timeout=60000)
        else:
            await self.page.goto(f"file:///{os.path.abspath(self.html_file)}", timeout=60000)
        self._last_url = self.page.url

    async def load_url(self, url: str):
        async with self._page_lock:
            self.html_file = url
            self.is_url = True
            if self.page is None:
                await self.setup()
            await self.page.goto(url, timeout=60000)
            self._last_url = self.page.url
            self.click_history = []
            print(f"已加载新页面: {url}")

            await self._debug_bar_init()

    async def render(self, mode="human"):
        async with self._page_lock:
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
        async with self._page_lock:
            obs = await self.page.screenshot(full_page=False)
            return obs

    async def render_sized(self, jpeg_quality=95):
        async with self._page_lock:
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
        async with self._page_lock:
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



#DOM TREE 
    async def get_dom_tree_with_id(self):
        async with self._page_lock:
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
                    const style = window.getComputedStyle(elem);
                    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
                    const rect = elem.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return false;
                    // 注意：elementFromPoint 需要的是相对于视口的坐标，这里保持原样
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
                    
                    // --- 1. 原版的判断逻辑完全保留 ---
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
                    
                    // --- 2. 新增的宽容逻辑：放过 span 等伪装元素 ---
                    // 检查 CSS 类名
                    const className = (typeof node.className === 'string') ? node.className.toLowerCase() : '';
                    if (className.includes('cursor-pointer') || className.includes('btn') || className.includes('button')) {
                        return true;
                    }
                    // 检查 ARIA 角色
                    const role = node.getAttribute('role');
                    if (role === 'button' || role === 'link' || role === 'menuitem' || role === 'tab') {
                        return true;
                    }
                    // 检查是否有直接绑定的点击事件
                    if (node.hasAttribute('onclick')) {
                        return true;
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
                            t === 'text' || t === 'password' || t === 'search' || 
                            t === 'number' || t === 'email' || t === 'tel' || t === 'url'
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
                        // 对于 span 等元素，提取内部文本
                        return node.innerText || node.textContent || '';
                    }
                }

                // --- 语义区域解析器 ---
                function getSemanticZone(node, parentZone) {
                    if (!node || node.nodeType !== 1) return parentZone;
                    const tag = node.tagName.toLowerCase();
                    const idClass = ((node.id || '') + ' ' + (typeof node.className === 'string' ? node.className : '')).toLowerCase();

                    if (tag === 'header' || idClass.includes('header')) return 'header';
                    if (tag === 'footer' || idClass.includes('footer')) return 'footer';
                    if (tag === 'nav' || idClass.includes('nav')) return 'nav';
                    if (tag === 'aside' || idClass.includes('sidebar')) return 'sidebar';
                    if (tag === 'dialog' || idClass.includes('modal') || idClass.includes('dialog')) return 'modal';
                    if (tag === 'main' || idClass.includes('main')) return 'main';
                    if (tag === 'form' || idClass.includes('form')) return 'form';
                    
                    return parentZone;
                }

                // --- 空间坐标解析器 (基于页面文档左上角的绝对坐标) ---
                function getSpatialCoordinates(node) {
                    if (!node || node.nodeType !== 1) return '[0,0,0,0]';
                    const rect = node.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return '[0,0,0,0]';
                    
                    // rect.left/top 是基于视口(Viewport)的坐标
                    // 加上 window.scrollX/Y 即可转换为基于整个页面文档左上角的坐标
                    const absoluteLeft = Math.round(rect.left + window.scrollX);
                    const absoluteTop = Math.round(rect.top + window.scrollY);
                    const rw = Math.round(rect.width);
                    const rh = Math.round(rect.height);

                    // 返回格式 [x, y, width, height]
                    return `[${absoluteLeft},${absoluteTop},${rw},${rh}]`;
                }

                function traverse(node, currentZone = 'body') {
                    if (node.tagName && node.tagName.toUpperCase() === 'SCRIPT') {
                        return null;
                    }
                    if (node.nodeType !== Node.ELEMENT_NODE) return null;
                    
                    const myZone = getSemanticZone(node, currentZone);
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

                    const children = [];
                    for (let child of node.childNodes) {
                        const childResult = traverse(child, myZone);
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
                        // --- 核心挂载：语义域 + 基于页面左上角的坐标 ---
                        result['address'] = `${myZone}--${getSpatialCoordinates(node)}`;
                        
                        result['can_interact'] = can_interact;
                        if (isTextInputLike(node)) {
                            result['input_value'] = input_value;
                        }
                        
                        // --- 修改点：不再限制只有特定的 tag 才能提取 visible_text ---
                        // 既然 should_keep 是 true，说明它是我们关心的元素（包括伪装的 span），直接提取文本
                        let text = getVisibleText(node);
                        if (text) {
                            // 稍微清理一下两端空白，防止全是空格的无用文本
                            result['visible_text'] = text.trim(); 
                        }

                        if (tag === 'select') {
                            result['options'] = options;
                        }
                    }
                    return result;
                }
                
                const domtree = traverse(document.body, 'body');
                return {domtree, id2xpath};
            }
            """
            result = await self.page.evaluate(js_code)
            return result["domtree"], result["id2xpath"]

    async def _capture_visual_state(self):
        """
        在交互操作前调用。
        通过“特征指纹”记录页面所有可见元素的信息，不再向 DOM 注入自定义属性，
        以防止 React/Vue 等框架重绘 DOM 导致标记丢失。
        """
        js_code = """
        () => {
            window.__webenv_snapshot_sigs = [];
            window.__webenv_scroll = { x: window.scrollX, y: window.scrollY };
            
            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                // 忽略面积过小的噪点元素 (例如 < 4平方像素)
                if (rect.width <= 2 || rect.height <= 2) return false;
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
            };

            const traverse = (node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.id === '___webenv_debugbar' || node.id === '___webenv_mouse') return;
                    
                    if (isVisible(node)) {
                        let directText = '';
                        for (let child of node.childNodes) {
                            if (child.nodeType === Node.TEXT_NODE) {
                                directText += child.textContent.trim();
                            }
                        }
                        if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') {
                            directText += node.value || '';
                        }
                        
                        // 只记录有实质性内容（文本或输入框）或特定视觉标签（图片、视频、SVG）的元素
                        // 过滤掉纯粹的透明排版容器，大幅减少噪点
                        const isVisualElement = directText.length > 0 || ['IMG', 'SVG', 'VIDEO', 'CANVAS'].includes(node.tagName);

                        if (isVisualElement) {
                            const rect = node.getBoundingClientRect();
                            window.__webenv_snapshot_sigs.push({
                                text: directText,
                                tag: node.tagName,
                                docX: Math.round(rect.x + window.scrollX),
                                docY: Math.round(rect.y + window.scrollY),
                                w: Math.round(rect.width),
                                h: Math.round(rect.height)
                            });
                        }
                    }
                    
                    for (let child of node.childNodes) {
                        traverse(child);
                    }
                }
            };
            
            traverse(document.body);
        }
        """
        try:
            if not self.page.is_closed():
                await self.page.evaluate(js_code)
        except Exception as e:
            print(f"[CAPTURE STATE] 记录视觉快照失败: {e}")

    async def _review_visual_changes(self):
        """
        在交互操作后调用。
        通过模糊匹配比对“视觉指纹”，找出真正新出现或内容改变的区域。
        如果变化区域在原始视口之外，则平滑/顿挫滚动过去展示，最后复位。
        """
        print("@@@等待页面渲染和动画完成...")
        await asyncio.sleep(0.8) # 给 React/Vue 的入场动画一点时间
        print("@@@正在分析页面视觉变化...")
        js_code = """
        () => {
            // 1. 检测是否页面发生了全局刷新或跳转 (变量丢失)
            if (!window.__webenv_snapshot_sigs) {
                return { is_refreshed: true };
            }
            
            const origScroll = window.__webenv_scroll;
            const changes = [];
            const oldSigs = window.__webenv_snapshot_sigs;
            
            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                if (rect.width <= 2 || rect.height <= 2) return false;
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
            };

            const traverse = (node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.id === '___webenv_debugbar' || node.id === '___webenv_mouse') return;
                    
                    if (isVisible(node)) {
                        let directText = '';
                        for (let child of node.childNodes) {
                            if (child.nodeType === Node.TEXT_NODE) {
                                directText += child.textContent.trim();
                            }
                        }
                        if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') {
                            directText += node.value || '';
                        }

                        const isVisualElement = directText.length > 0 || ['IMG', 'SVG', 'VIDEO', 'CANVAS'].includes(node.tagName);

                        if (isVisualElement) {
                            const rect = node.getBoundingClientRect();
                            const docX = Math.round(rect.x + window.scrollX);
                            const docY = Math.round(rect.y + window.scrollY);
                            const w = Math.round(rect.width);
                            const h = Math.round(rect.height);
                            
                            const isMatch = oldSigs.some(old => {
                                return old.text === directText &&
                                       old.tag === node.tagName &&
                                       Math.abs(old.docX - docX) <= 2 &&
                                       Math.abs(old.docY - docY) <= 2 &&
                                       Math.abs(old.w - w) <= 2 &&
                                       Math.abs(old.h - h) <= 2;
                            });

                            if (!isMatch) {
                                changes.push({ x: docX, y: docY, w: rect.width, h: rect.height });
                            }
                        }
                    }
                    for (let child of node.childNodes) {
                        traverse(child);
                    }
                }
            };
            traverse(document.body);
            
            delete window.__webenv_snapshot_sigs;
            delete window.__webenv_scroll;
            
            return { is_refreshed: false, changedRects: changes, origScroll: origScroll };
        }
        """
        
        # --- 基础的通用拟人化分段平滑滚动 (用于大段位移或复位) ---
        def get_human_scroll_js(target_x, target_y):
            return f"""
            async () => {{
                const sleep = ms => new Promise(r => setTimeout(r, ms));
                const startX = window.scrollX;
                const startY = window.scrollY;
                const endX = {target_x};
                const endY = {target_y};

                const distanceX = endX - startX;
                const distanceY = endY - startY;
                const totalDist = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

                if (totalDist <= 0) {{
                    window.scrollTo(endX, endY);
                    return;
                }}

                let scrolledDist = 0;

                const swipe = async (chunkDist, durationMs) => {{
                    const beginDist = scrolledDist;
                    const targetDist = Math.min(beginDist + chunkDist, totalDist);
                    const actualChunkDist = targetDist - beginDist;

                    const startTime = performance.now();
                    return new Promise(resolve => {{
                        const step = (time) => {{
                            let progress = (time - startTime) / durationMs;
                            if (progress > 1) progress = 1;

                            // 段内依然使用阻尼缓动
                            const ease = 1 - Math.pow(1 - progress, 3);
                            const currentRunDist = beginDist + actualChunkDist * ease;
                            const ratio = currentRunDist / totalDist;

                            window.scrollTo(startX + distanceX * ratio, startY + distanceY * ratio);

                            if (progress < 1) {{
                                window.requestAnimationFrame(step);
                            }} else {{
                                scrolledDist = targetDist;
                                resolve();
                            }}
                        }};
                        window.requestAnimationFrame(step);
                    }});
                }};

                while (scrolledDist < totalDist) {{
                    const dist = 300 + Math.random() * 300;
                    const duration = 200 + Math.random() * 200;
                    
                    await swipe(dist, duration);

                    if (scrolledDist < totalDist) {{
                        await sleep(50 + Math.random() * 100);
                    }}
                }}
                window.scrollTo(endX, endY);
            }}
            """


        def get_search_scroll_js(target_x, target_y):
            return f"""
            async () => {{
                const sleep = ms => new Promise(r => setTimeout(r, ms));
                const startX = window.scrollX;
                const startY = window.scrollY;
                const endX = {target_x};
                const endY = {target_y};

                const distanceX = endX - startX;
                const distanceY = endY - startY;
                const totalDist = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

                if (totalDist <= 0) {{
                    window.scrollTo(endX, endY);
                    return;
                }}

                let scrolledDist = 0;

                // 定义每次小幅度滚动的平滑动画
                const swipeChunk = async (chunkDist, durationMs) => {{
                    const beginDist = scrolledDist;
                    const targetDist = Math.min(beginDist + chunkDist, totalDist);
                    const actualChunkDist = targetDist - beginDist;

                    const startTime = performance.now();
                    return new Promise(resolve => {{
                        const step = (time) => {{
                            let progress = (time - startTime) / durationMs;
                            if (progress > 1) progress = 1;

                            // 使用 ease-out (减速) 缓动，模拟滚轮拨动后停下的摩擦力感
                            const ease = 1 - Math.pow(1 - progress, 3);
                            const currentRunDist = beginDist + actualChunkDist * ease;
                            const ratio = currentRunDist / totalDist;

                            window.scrollTo(startX + distanceX * ratio, startY + distanceY * ratio);

                            if (progress < 1) {{
                                window.requestAnimationFrame(step);
                            }} else {{
                                scrolledDist = targetDist;
                                resolve();
                            }}
                        }};
                        window.requestAnimationFrame(step);
                    }});
                }};

                while (scrolledDist < totalDist) {{
                    // 模拟一次滚轮拨动或短距离拖拽 (150~250px)
                    const dist = 150 + Math.random() * 100;
                    // 每次拨动的耗时，保证有足够的帧数被录制下来 (200~300ms)
                    const duration = 200 + Math.random() * 100;

                    await swipeChunk(dist, duration);

                    if (scrolledDist < totalDist) {{
                        // 停顿时间变长，模拟人眼在阅读/确认页面内容 (300~600ms)
                        await sleep(300 + Math.random() * 300);
                    }}
                }}
                
                // 确保最终精确落位
                window.scrollTo(endX, endY);
            }}
            """
        try:
            if self.page.is_closed():
                return
                
            res = await self.page.evaluate(js_code)
            
            if not res:
                print("@@@无法获取页面信息，停止。")
                return
            
            scroll_speed_px_per_sec = 800

            # --- 2. 拦截全局刷新逻辑 ---
            if res.get('is_refreshed'):
                print("@@@检测到页面发生了跳转或整体刷新！调用全屏人类拟态滚动展示...")
                
                initial_scroll = await self.page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")
                
                await self._scroll_page_human_like_internal(bar_color="#fbbc05")
                
                current_scroll_after_scan = await self.page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")
                distance_back = abs(initial_scroll['y'] - current_scroll_after_scan['y'])
                
                if distance_back > 0:
                    await self.page.evaluate(get_human_scroll_js(initial_scroll['x'], initial_scroll['y']))
                    
                return 
            
            # --- 3. 继续原本的局部变化检测逻辑 ---
            changes = res.get('changedRects', [])
            orig_scroll = res.get('origScroll', {'x': 0, 'y': 0})
            
            if not changes:
                print("@@@页面无实质性视觉变化，无需展示")
                return

            vp_w = getattr(self, 'viewport_width', 1920)
            vp_h = getattr(self, 'viewport_height', 1080)
            
            margin = 100
            vp_top = orig_scroll['y'] - margin
            vp_bottom = orig_scroll['y'] + vp_h + margin
            
            outside_rects = []
            for r in changes:
                cy = r['y'] + r['h'] / 2
                if cy < vp_top or cy > vp_bottom:
                    outside_rects.append(r)
            
            if not outside_rects:
                print("@@@页面变化都在当前视口内，无需滚动展示")
                return

            clusters = []
            outside_rects.sort(key=lambda r: r['y'])
            
            for r in outside_rects:
                if not clusters:
                    clusters.append([r])
                else:
                    last_cluster = clusters[-1]
                    last_r = last_cluster[-1]
                    if r['y'] - (last_r['y'] + last_r['h']) < vp_h * 0.5:
                        clusters[-1].append(r)
                    else:
                        clusters.append([r])
                        
            current_y = orig_scroll['y']
            
            for cluster in clusters:
                min_y = min(r['y'] for r in cluster)
                max_y = max(r['y'] + r['h'] for r in cluster)
                center_y = min_y + (max_y - min_y) / 2
                
                target_y = max(0, center_y - vp_h / 2)
                target_x = orig_scroll['x'] 
                
                distance_y = abs(target_y - current_y)
                
                scroll_ms = max((distance_y / scroll_speed_px_per_sec) * 1000, 500) if distance_y > 0 else 0
                observe_ms = 1000 
                total_ms = int(scroll_ms + observe_ms)
                
                if hasattr(self, '_debug_bar_pulse'):
                    await self._debug_bar_pulse("OBSERVING CHANGES", active_ms=total_ms, color="#fbbc05")

                # --- 4. 前往观察点使用“寻找式顿挫滚动”，模拟找变化 ---
                if distance_y > 0:
                    await self.page.evaluate(get_search_scroll_js(target_x, target_y))

                await asyncio.sleep(observe_ms / 1000.0)
                current_y = target_y

            # --- 5. 观察完毕后，复位使用原有的“平滑拟人滚动”回去 ---
            distance_back = abs(orig_scroll['y'] - current_y)
            if distance_back > 0:
                await self.page.evaluate(get_human_scroll_js(orig_scroll['x'], orig_scroll['y']))

        except Exception as e:
            print(f"[REVIEW CHANGES] 浏览变化区域时出错: {e}")

    async def _review_visual_changes_safe(self, timeout: float = 45.0):
        try:
            await asyncio.wait_for(self._review_visual_changes(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[REVIEW CHANGES] 超时 ({timeout}s)，跳过视觉变化展示")
        except Exception as e:
            print(f"[REVIEW CHANGES] _review_visual_changes_safe 异常: {e}")


    async def check_if_scroll_needed(self):
        async with self._page_lock:
            # 综合对比页面的真实内容尺寸与当前视口 (Viewport) 尺寸
            # 增加对 body/documentElement 的兼容，并加入 2px 容差处理浮点数渲染精度问题
            check_scroll_js = """
            () => {
                const vh = window.innerHeight || document.documentElement.clientHeight;
                const vw = window.innerWidth || document.documentElement.clientWidth;

                // 取各类高度的极大值，确保不管哪个容器被撑开都能捕捉到
                const scrollHeight = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.offsetHeight,
                    document.body.clientHeight,
                    document.documentElement.clientHeight
                );

                const scrollWidth = Math.max(
                    document.body.scrollWidth,
                    document.documentElement.scrollWidth,
                    document.body.offsetWidth,
                    document.documentElement.offsetWidth,
                    document.body.clientWidth,
                    document.documentElement.clientWidth
                );

                // 只要垂直或水平方向上内容超出了视口（加上 2px 的容错），就认为需要滚动
                const needVertical = (scrollHeight - vh) > 2;
                const needHorizontal = (scrollWidth - vw) > 2;

                // 如果你的后续自动化逻辑 (scroll_page_human_like) 只做垂直滚动，可以直接 return needVertical;
                return needVertical || needHorizontal;
            }
            """
            
            is_scroll_needed = await self.page.evaluate(check_scroll_js)
            print("@@@SCROLL RESULT: ", is_scroll_needed)
            return is_scroll_needed


    async def scroll_page_human_like(self, bar_color="#00cc00"):
        """
        供外部 API 调用的包装方法，负责加锁。
        """
        async with self._page_lock:
            await self._scroll_page_human_like_internal(bar_color)

    async def _scroll_page_human_like_internal(self, bar_color="#00cc00"):
        """
        内部核心逻辑，不包含加锁逻辑（调用方必须已经持有 _page_lock）。
        """
        try:
            if self.page.is_closed():
                return

            get_scroll_height_js = "() => Math.max(0, document.documentElement.scrollHeight - window.innerHeight)"
            max_scroll = await self.page.evaluate(get_scroll_height_js)

            if max_scroll <= 0:
                return

            # 每次滚动距离: 200 + random()*150  → 平均 275px
            # 每次滚动耗时: 800 + random()*600  → 平均 1100ms
            # 每次间隔休眠: 100 + random()*150  → 平均  175ms
            # 每轮合计:                          → 平均 1275ms
            # 最后一次休眠: 300 + random()*400   → 平均  500ms
            avg_scroll_per_iter = 275
            avg_ms_per_iter = 1275
            num_iters = max_scroll / avg_scroll_per_iter
            estimated_ms = int(num_iters * avg_ms_per_iter + 500)

            await self._debug_bar_pulse("FAST SCROLLING", active_ms=estimated_ms, color=bar_color)
            await asyncio.sleep(0.3)

            js_code = """
            async () => {
                const maxScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
                const sleep = ms => new Promise(r => setTimeout(r, ms));

                const swipe = async (dist, durationMs) => {
                    const beginY = window.scrollY;
                    let targetY = beginY + dist;
                    if (targetY > maxScroll) targetY = maxScroll;
                    const startTime = performance.now();

                    return new Promise(resolve => {
                        const step = (time) => {
                            let progress = (time - startTime) / durationMs;
                            if (progress > 1) progress = 1;

                            const ease = 1 - Math.pow(1 - progress, 3);
                            window.scrollTo(0, beginY + (targetY - beginY) * ease);

                            if (progress < 1) {
                                requestAnimationFrame(step);
                            } else {
                                resolve();
                            }
                        };
                        requestAnimationFrame(step);
                    });
                };

                let currentY = window.scrollY;
                let maxIter = 300;
                let noProgressCount = 0;
                while (currentY < maxScroll && maxIter-- > 0) {
                    const prevY = currentY;
                    const dist = 200 + Math.random() * 150;
                    const duration = 800 + Math.random() * 600;

                    await swipe(dist, duration);
                    currentY = window.scrollY;

                    if (currentY <= prevY) {
                        noProgressCount++;
                        if (noProgressCount >= 3) break;
                    } else {
                        noProgressCount = 0;
                    }

                    if (currentY < maxScroll) {
                        await sleep(100 + Math.random() * 150);
                    }
                }
                await sleep(300 + Math.random() * 400);
            }
            """

            await self.page.evaluate(js_code)
            self.click_history.append(("auto_scroll_page", None))

        except Exception as e:
            print(f"[AUTO SCROLL] 页面自动化滚动失败: {e}")

    async def click(self, unique_id: int, id2xpath: dict):
        async with self._page_lock:
            xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
            if not xpath:
                print(f"[CLICK] No xpath found for id={unique_id}")
                return False

            elem = await self.page.query_selector(f'xpath={xpath}')
            if elem is None:
                print(f"[CLICK] No element found by xpath {xpath} for id={unique_id}")
                return False

            try:
                start_url = self.page.url

                center = await self._smooth_scroll_into_view(elem)
                if not center:
                    print(f"[CLICK] Element not visible/bbox missing: {xpath}")
                    return False
                tx, ty = center

                if not await self._cursor_exists():
                    print("[CLICK] Cursor visually missing, resetting...")
                    await self._inject_cursor_overlay()
                    cx, cy = self._mouse_x, self._mouse_y
                    if cx is None or cy is None:
                        cx, cy = await self._get_viewport_center()
                    await self.page.mouse.move(cx, cy)
                    await self._set_cursor_pos(cx, cy)
                    self._mouse_x, self._mouse_y = cx, cy

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
                await self._capture_visual_state()
                t_start = time.perf_counter()

                await self._move_mouse_natural(tx, ty)

                await self._debug_bar_pulse("CLICK", active_ms=160)
                time.sleep(1)

                try:
                    await self.page.mouse.click(tx, ty)
                except Exception:
                    pass

                try:
                    await self.page.evaluate("() => window.__webenvClickFrame && window.__webenvClickFrame.promise")
                except Exception:
                    pass

                await asyncio.sleep(0.5)

                did_reload = await self._reload_tools_if_needed(start_url)

                if not did_reload:
                    try:
                        await self.page.wait_for_load_state("networkidle", timeout=2000)
                    except Exception:
                        await asyncio.sleep(0.05)

                try:
                    self._last_url = self.page.url
                except Exception:
                    pass
                # ==========================================
                # [新增] 2. 如果页面没有发生跨域跳转，比对并浏览视觉变化
                # ==========================================
                await self._review_visual_changes_safe()

                self.click_history.append(("click", unique_id))
                return True, t_start

            except Exception as e:
                print(f"[CLICK] Failed click: {e}")
                return False, t_start

    async def enter(self, unique_id, content, id2xpath):
        async with self._page_lock:
            xpath = id2xpath.get(str(unique_id)) or id2xpath.get(int(unique_id))
            if not xpath:
                print(f"[ENTER] No xpath found for id={unique_id}")
                return False
            elem = await self.page.query_selector(f'xpath={xpath}')
            if elem is None:
                print(f"[ENTER] No element found by xpath {xpath} for id={unique_id}")
                return False
            try:
                start_url = self.page.url
                await self.page.evaluate("() => { if(!window.__webenvClickFrame) window.__webenvClickFrame = {}; }")
                tag = await elem.evaluate("el => el.tagName.toLowerCase()")
                if tag not in ["input", "textarea"]:
                    print(f"[ENTER] Element id={unique_id} is {tag}, not input/textarea, cannot enter text.")
                    return False

                input_type = (await elem.get_attribute("type") or "").lower()

                # 平滑滚动到元素
                center = await self._smooth_scroll_into_view(elem)
                if not center:
                    print(f"[ENTER] Element not visible/bbox missing: {xpath}")
                    return False
                tx, ty = center

                # 确保鼠标存在
                if not await self._cursor_exists():
                    print("[ENTER] Cursor visually missing, resetting...")
                    await self._inject_cursor_overlay()
                    cx, cy = self._mouse_x, self._mouse_y
                    if cx is None or cy is None:
                        cx, cy = await self._get_viewport_center()
                    await self.page.mouse.move(cx, cy)
                    await self._set_cursor_pos(cx, cy)
                    self._mouse_x, self._mouse_y = cx, cy

                # === range 类型特殊处理参考 click ===
                if input_type == "range":
                    # 注入 click 逻辑的视觉状态捕捉
                    await self._capture_visual_state()

                    # 移动鼠标到目标位置
                    await self._move_mouse_natural(tx, ty)

                    # 红色闪烁 1秒（click 的逻辑是 160ms，这里改成 1000ms）
                    await self._debug_bar_pulse("SET RANGE", active_ms=160)

                    # 等待红条闪完
                    time.sleep(1)

                    # 点击一下（参考 click）
                    try:
                        await self.page.mouse.click(tx, ty)
                    except Exception:
                        pass

                    # 瞬间设置 slider 值并触发输入/变更事件
                    await elem.evaluate(
                        """(el, val) => {
                            el.value = val;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        }""",
                        str(content)
                    )

                    await asyncio.sleep(0.1)

                    # 检测是否需要重载工具（参考 click）
                    await self._reload_tools_if_needed(start_url)
                    try:
                        self._last_url = self.page.url
                    except Exception:
                        pass

                    # 浏览视觉变化（参考 click）
                    await self._review_visual_changes_safe()

                    # 记录历史
                    self.click_history.append(("enter-range", unique_id, str(content)))
                    return True

                # 非 Range 类型按原逻辑
                await self._move_mouse_natural(tx, ty)
                await self._capture_visual_state()
                await self.page.mouse.click(tx, ty)

                # 数值输入特判
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
                        m = re.search(r"[-+]?\d*\\.?\\d+", str(content))
                        fill_content = str(float(m.group())) if m else "0"

                await elem.fill("")
                typing_plan = []
                total_duration_ms = 500
                for char in fill_content:
                    if '\u4e00' <= char <= '\u9fff':
                        delay = random.uniform(0.5, 2.0)
                    else:
                        delay = random.uniform(0.03, 0.3)
                    typing_plan.append((char, delay))
                    total_duration_ms += (delay * 1000) + 50

                await self._debug_bar_pulse("TYPING", active_ms=total_duration_ms, color="#00cc00")
                for char, delay in typing_plan:
                    await elem.type(char)
                    await asyncio.sleep(delay)
                await asyncio.sleep(0.1)

                await self._reload_tools_if_needed(start_url)
                try:
                    self._last_url = self.page.url
                except Exception:
                    pass
                self.click_history.append(("enter", unique_id, fill_content))
                await self._review_visual_changes_safe()
                return True

            except Exception as e:
                print(f"[ENTER] Failed to enter text: {e}")
                return False
        
    async def select(self, unique_id: Any, value: Any, id2xpath: Dict[Any, str]) -> bool:
        async with self._page_lock:
            xpath: Optional[str] = id2xpath.get(str(unique_id)) or (id2xpath.get(int(unique_id)) if isinstance(unique_id, (str, int)) else None)
            if not xpath:
                print(f"[SELECT] No xpath found for id={unique_id}")
                return False

            elem = await self.page.query_selector(f"xpath={xpath}")
            if elem is None:
                print(f"[SELECT] No element found by xpath {xpath} for id={unique_id}")
                return False

            try:
                start_url = self.page.url

                tag = await elem.evaluate("el => el.tagName.toLowerCase()")
                center = await self._smooth_scroll_into_view(elem)
                if not center:
                    print(f"[SELECT] Element not visible/bbox missing: {xpath}")
                    return False
                tx, ty = center

                if not await self._cursor_exists():
                    print("[CLICK] Cursor visually missing, resetting...")
                    await self._inject_cursor_overlay()
                    cx, cy = self._mouse_x, self._mouse_y
                    if cx is None or cy is None:
                        cx, cy = await self._get_viewport_center()
                    await self.page.mouse.move(cx, cy)
                    await self._set_cursor_pos(cx, cy)
                    self._mouse_x, self._mouse_y = cx, cy

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

                await self._capture_visual_state()

                t_start = time.perf_counter()

                async def finish_action(action_key, val=None):
                    await asyncio.sleep(0.1)
                    await self._reload_tools_if_needed(start_url)

                    try:
                        self._last_url = self.page.url
                    except:
                        pass

                    rec = (action_key, unique_id) if val is None else (action_key, unique_id, val)
                    self.click_history.append(rec)
                    await self._review_visual_changes_safe()
                    return True, t_start

                if tag == "select":
                    try:
                        # 1. 预先获取所有选项及索引，用于文本匹配 (此时下拉框还未展开)
                        options = await elem.evaluate(
                            "el => Array.from(el.options).map((o, i) => ({index: i, value: o.value, text: o.text}))"
                        )

                        target_str = str(value)
                        target_index = -1
                        target_value = target_str
                        target_display_text = target_str
                        found_exact = False

                        # 2. 精确匹配
                        for opt in options:
                            if opt["value"] == target_str or opt["text"] == target_str:
                                target_index = opt["index"]
                                target_value = opt["value"]
                                target_display_text = opt["text"] 
                                found_exact = True
                                break

                        # 3. 模糊匹配
                        if not found_exact:
                            option_texts = [opt["text"] for opt in options]
                            closest = difflib.get_close_matches(target_str, option_texts, n=1, cutoff=0.0)
                            if not closest:
                                print(f"[SELECT] No similar option (by text) for value: {target_str}")
                                return False, t_start

                            closest_text = closest[0]
                            for opt in options:
                                if opt["text"] == closest_text:
                                    target_index = opt["index"]
                                    target_value = opt["value"]
                                    target_display_text = opt["text"]
                                    break

                        # 5. 开始执行物理动作序列：移动并点击原始 select 框
                        await self._move_mouse_natural(tx, ty)


                        # 4. === 触发全局长绿条 ===
                        # 预估总耗时: 
                        # 移动到 select(0) -> 展开等待(0.8s) -> 滚动判断(0.3s) -> 移动到 option(0) -> 点击等待(0.2s) -> 展示高亮(0.5s)
                        # 保守估计总耗时约 2500ms
                        total_duration_ms = 2500 
                        await self._debug_bar_pulse(f"SELECTING: {target_display_text}", active_ms=total_duration_ms, color="#00cc00")

                        await self.page.mouse.click(tx, ty)
                        
                        # 展开下拉框 (转换为 listbox)
                        await elem.evaluate("""el => {
                            el.dataset.originalSize = el.getAttribute('size') || '';
                            el.setAttribute('size', Math.min(6, el.options.length)); 
                        }""")
                        await asyncio.sleep(0.8) # 等待展开动画

                        # 6. 防弹级定位选项元素
                        opt_js_handle = await elem.evaluate_handle(f"el => el.options[{target_index}]")
                        target_opt_elem = opt_js_handle.as_element()

                        if not target_opt_elem:
                            print(f"[SELECT] Cannot find option element at index: {target_index}")
                            await elem.evaluate("el => el.setAttribute('size', el.dataset.originalSize || 1)")
                            return False, t_start

                        # 7. 确保选项在视野内
                        await target_opt_elem.scroll_into_view_if_needed()
                        await asyncio.sleep(0.3)

                        opt_box = await target_opt_elem.bounding_box()
                        if opt_box:
                            opt_tx = opt_box["x"] + opt_box["width"] / 2
                            opt_ty = opt_box["y"] + opt_box["height"] / 2

                            # 8. 移动鼠标到具体选项并点击
                            await self._move_mouse_natural(opt_tx, opt_ty)
                            await asyncio.sleep(0.2)
                            await self.page.mouse.click(opt_tx, opt_ty)
                            await asyncio.sleep(0.5) # 停留，展示高亮

                            # 9. 收尾：恢复下拉框状态并触发事件
                            await elem.evaluate(f"""el => {{
                                el.selectedIndex = {target_index};
                                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                if (el.dataset.originalSize) {{
                                    el.setAttribute('size', el.dataset.originalSize);
                                }} else {{
                                    el.removeAttribute('size');
                                }}
                            }}""")

                            action_type = "select" if found_exact else "select-closest"
                            return await finish_action(action_type, target_value)

                        else:
                            print(f"[SELECT] Cannot get bounding box for option index: {target_index}")
                            await elem.evaluate("el => el.setAttribute('size', el.dataset.originalSize || 1)")
                            return False, t_start

                    except Exception as e:
                        print(f"[SELECT] Exception during <select> handling: {e}")
                        await elem.evaluate("el => el.setAttribute('size', el.dataset.originalSize || 1)")
                        return False, t_start

                elif tag == "input":
                    input_type = (await elem.get_attribute("type")) or ""
                    input_type = input_type.lower()

                    if input_type == "radio":
                        await self._debug_bar_pulse("SELECT RADIO", active_ms=1000, color="#00cc00")
                        await self._move_mouse_natural(tx, ty)
                        await asyncio.sleep(0.2)
                        await self.page.mouse.click(tx, ty)
                        await asyncio.sleep(0.5)
                        return await finish_action("select-radio")

                    elif input_type == "date":
                        await self._debug_bar_pulse(f"SET DATE: {value}", active_ms=1000, color="#00cc00")
                        await asyncio.sleep(0.3)
                        await elem.fill(str(value))
                        await asyncio.sleep(0.5)
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
        async with self._page_lock:
            """
            开始录制（切换到启用 record_video 的新 context/page）
            """
            print("正在准备启动视频录制")
            self._t0 = time.perf_counter()
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

            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            except:
                pass
            await self._inject_cursor_overlay()
            await self._debug_bar_init()

            # === ✅ MODIFIED: 录制开始时保持鼠标位置 ===
            if self._mouse_x is not None and self._mouse_y is not None:
                tx, ty = self._mouse_x, self._mouse_y
                tx = max(0, min(self.viewport_width, tx))
                ty = max(0, min(self.viewport_height, ty))
            else:
                tx, ty = await self._get_viewport_center()

            await self.page.mouse.move(tx, ty)
            await self._set_cursor_pos(tx, ty)
            self._mouse_x, self._mouse_y = tx, ty
            # === END MODIFICATION ===

            self.recording = True
            print(f"[VIDEO] Started. dir={self._video_tmpdir}, size={record_size}")
            return True

    async def end_video(self) -> Optional[Dict[str, str]]:
        async with self._page_lock:
            """
            结束录制并恢复环境
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
                
                # === ✅ MODIFIED: 恢复非录制模式时保持鼠标位置 ===
                if self._mouse_x is not None and self._mouse_y is not None:
                    tx, ty = self._mouse_x, self._mouse_y
                    tx = max(0, min(self.viewport_width, tx))
                    ty = max(0, min(self.viewport_height, ty))
                else:
                    tx, ty = await self._get_viewport_center()

                await self.page.mouse.move(tx, ty)
                await self._set_cursor_pos(tx, ty)
                self._mouse_x, self._mouse_y = tx, ty
                # === END MODIFICATION ===

            if output:
                self.video_segments.append(output)
            return output



    # ========= 录制：CDP 动态截帧方案 =========
    async def start_video_cdp(self, width: Optional[int] = None, height: Optional[int] = None) -> bool:
        async with self._page_lock:
            print("正在准备启动 CDP 视频录制...")
            self._t0 = time.perf_counter()
            if self.recording:
                return True

            if self.page is None:
                await self.setup()
                await self._goto_current_target()

            self._cdp_frames = []
            self._cdp_session = await self.page.context.new_cdp_session(self.page)

            async def on_screencast_frame(event):
                self._cdp_frames.append(event["data"])
                try:
                    await self._cdp_session.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
                except Exception:
                    pass

            self._cdp_session.on("Page.screencastFrame", on_screencast_frame)

            # --- 核心黑科技：注入不可见的动画，强制 Chrome 持续发帧 ---
            async def inject_force_paint():
                try:
                    await self.page.evaluate("""() => {
                        if (document.getElementById('cdp-keep-alive')) return;
                        const style = document.createElement('style');
                        style.id = 'cdp-keep-alive';
                        // 一个透明度在 1 和 0.99 之间微小闪烁的动画，肉眼完全看不见，但足以骗过 Chrome
                        style.innerHTML = `@keyframes cdpForcePaint { 0% { opacity: 1; } 100% { opacity: 0.99; } } body { animation: cdpForcePaint 0.1s infinite alternate; }`;
                        document.head.appendChild(style);
                    }""")
                    print("[VIDEO] 已注入强制渲染动画")
                except Exception:
                    pass

            async def _start_screencast():
                try:
                    await asyncio.sleep(0.2) # 微小缓冲
                    await self._cdp_session.send("Page.startScreencast", {
                        "format": "jpeg",
                        "quality": 80,
                        "everyNthFrame": 30 # 建议改小一点，比如5，保证流畅度
                    })
                    print("[VIDEO] CDP Screencast 指令已发送")
                    await inject_force_paint() # 启动时注入一次
                except Exception as e:
                    print(f"[VIDEO] Screencast 启动失败: {e}")

            # --- 改回 framenavigated，专门对付 SPA 路由跳转 ---
            async def on_frame_navigated(frame):
                if frame == self.page.main_frame and self.recording:
                    print(f"[VIDEO] 检测到路由变化: {frame.url}，正在重启截帧...")
                    asyncio.create_task(_start_screencast())

            self.page.on("framenavigated", on_frame_navigated)

            # 初次启动
            await _start_screencast()

            # 保持鼠标位置
            if self._mouse_x is not None and self._mouse_y is not None:
                tx, ty = max(0, min(self.viewport_width, self._mouse_x)), max(0, min(self.viewport_height, self._mouse_y))
                await self.page.mouse.move(tx, ty)
            
            self.recording = True
            print("[VIDEO] CDP Screencast Started seamlessly.")
            return True
        
    async def end_video_cdp(self):
        async with self._page_lock:
            if not self.recording:
                return None
                
            print("正在停止 CDP 视频录制...")
            # 停止截屏流并断开 CDP 会话
            try:
                await self._cdp_session.send("Page.stopScreencast")
                await self._cdp_session.detach()
            except Exception as e:
                print(f"停止 CDP 失败: {e}")

            self.recording = False
            
            # 返回抓取到的 base64 图像序列
            # 你可以将这些 base64 传回主脚本，直接用 cv2.VideoWriter 合成视频，
            # 也可以直接丢给你原有的 video_tools 处理
            return self._cdp_frames
        

    def get_video_segments(self) -> List[Dict[str, str]]:
        return list(self.video_segments)

    async def save_state(self):
        return list(self.click_history)

    async def restore_state(self, click_path):
        async with self._page_lock:
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
        async with self._page_lock:
            html = await self.page.evaluate('document.documentElement.outerHTML')
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"{file_name} 保存完成 (from evaluate)")

    async def close(self):
        async with self._page_lock:
            if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                print(f"[CLEANUP] Removing residual video dir: {self._video_tmpdir}")
                try:
                    shutil.rmtree(self._video_tmpdir, ignore_errors=True)
                except Exception as e:
                    print(f"[CLEANUP] Failed to remove dir: {e}")
            self._video_tmpdir = None
            self.recording = False

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
    result = await env.click(req.id, req.id2xpath)
    elapsed_sec = round(time.perf_counter() - start, 3)
    t_start = None

    now_mono = time.perf_counter()
    if env._first_event_monotonic is None:
        env._first_event_monotonic = now_mono

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
    obs_bytes = await env.render_viewport()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}


@app.get("/observe_sized_small")
async def api_observe_sized_small():
    global env
    obs_bytes = await env.render_sized_small()
    obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
    return {"image_b64": obs_b64}



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


@app.post("/start_video")
async def api_start_video(req: StartVideoRequest):
    global env
    ok = await env.start_video(width=req.width, height=req.height)
    return {"result": ok, "recording": env.recording, "t0": env._t0}


@app.post("/end_video")
async def api_end_video():
    global env
    output = await env.end_video()
    if not output:
        return {"result": False, "error": "no_video"}
    return {"result": True, **output, "segments_count": len(env.video_segments)}


@app.post("/start_video_cdp")
async def api_start_video_cdp(req: StartVideoRequest):
    """
    使用 Chrome DevTools Protocol (CDP) 启动无感录制
    """
    global env
    if env is None or env.page is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized.")
    
    # 注意：确保在 WebHtmlGymEnv 中你将上一步的方法命名为了 start_video_cdp
    # 如果你直接覆盖了原有的 start_video，这里改成 await env.start_video(width=req.width, height=req.height)
    ok = await env.start_video_cdp(width=req.width, height=req.height)
    
    return {
        "result": ok, 
        "recording": getattr(env, "recording", False), 
        "t0": getattr(env, "_t0", None)
    }


@app.post("/end_video_cdp")
async def api_end_video_cdp():
    """
    停止 CDP 录制，并直接返回 base64 图像帧数组
    """
    global env
    if env is None or env.page is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized.")
        
    # 注意：确保在 WebHtmlGymEnv 中你将上一步的方法命名为了 end_video_cdp
    frames = await env.end_video_cdp()
    
    if frames is None:
        return {"result": False, "error": "not_recording"}
    
    if len(frames) == 0:
        return {"result": False, "error": "no_frames_captured"}
        
    return {
        "result": True, 
        "frames_count": len(frames),
        "frames": frames  # 这就是一个包含所有截帧 base64 字符串的列表
    }

@app.post("/load_url")
async def api_load_url(req: LoadUrlRequest):
    global env
    if not req.url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")

    try:
        await env.load_url(req.url)
        return {"result": True, "url": req.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load url: {e}")


@app.get("/merge_video")
async def api_merge_video():
    global env
    videos = env.get_video_segments()
    return {
        "result": True,
        "count": len(videos),
        "videos": videos
    }


@app.post("/check_if_scroll_needed")
async def api_check_if_scroll_page():
    global env
    r = await env.check_if_scroll_needed()
    print("@@@API RESULT: ", r)
    return r

@app.post("/scroll_page")
async def api_scroll_page():
    """
    触发浏览器执行一次全局的“人工模拟”滚动浏览，并返回原位。
    常用于在视频录制阶段主动展示长页面的全貌。
    """
    global env
    if env is None or env.page is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized.")
        
    start_time = time.perf_counter()
    
    await env.scroll_page_human_like()
    
    elapsed_sec = round(time.perf_counter() - start_time, 3)
    return {
        "result": True, 
        "time_taken_sec": elapsed_sec,
        "message": "Human-like scroll completed."
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
        reload=False
    )