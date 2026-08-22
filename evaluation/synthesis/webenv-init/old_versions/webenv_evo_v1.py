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
import random
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

        # ✅ NEW: 全局互斥锁，避免 start/end_video 关闭/重建 page/context 时与其它请求并发冲突
        self._page_lock = asyncio.Lock()

    async def _debug_bar_init(self):
        """
        确保页面底部有一条常驻蓝色调试条。
        采用“顽固模式”：如果 DOM 没准备好，会轮询直到注入成功；
        并挂载一个守护进程，防止它被页面删除。
        """
        print("[BAR] INIT (Robust)")

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
        color: hex 颜色字符串 (如 '#00cc00' 为绿, '#d93025' 为红)
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
                # ✅ 传入 color 参数
                await self.page.evaluate(js, {"label": label, "activeMs": int(active_ms), "color": color})
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

        png_path = pathlib.Path(__file__).parent / "cursur_blue.png"
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
        """
        检查鼠标是否“视觉上有效存在”。
        """
        try:
            return await self.page.evaluate("""() => {
                const el = document.getElementById('___webenv_mouse');
                if (!el) return false;
                const mountPoint = document.body || document.documentElement;
                return el.parentNode === mountPoint;
            }""")
        except Exception:
            return False

    async def _set_cursor_pos(self, x: float, y: float):
        """
        移动 PNG 鼠标箭头。
        """
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

                centX, centY = await self._get_viewport_center()
                print("[MOUSE] SETTING TO CENTER")
                await self._set_cursor_pos(centX, centY)
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


    def _ease_out_quart(self, t: float) -> float:
        """
        缓动函数：急停效果 (Quart Out)
        这种曲线起步非常快，后段减速明显，适合模拟快速甩鼠标的感觉
        """
        return 1 - pow(1 - t, 4)

    async def _move_mouse_natural(self, target_x: float, target_y: float):
        """
        高级拟人移动逻辑：
        1. 极速起步
        2. 长距离自动触发"手指重定位" (Clutching) 停顿
        3. 模拟过冲 (Overshoot) 和回正 (Correction)
        """
        await self._ensure_mouse_origin()
        await self._ensure_cursor_ready()

        start_x, start_y = float(self._mouse_x), float(self._mouse_y)
        total_dist = math.hypot(target_x - start_x, target_y - start_y)

        # === 1. 极短距离直接瞬移 (小于15px) ===
        if total_dist < 15:
            await self.page.mouse.move(target_x, target_y)
            await self._set_cursor_pos(target_x, target_y)
            self._mouse_x, self._mouse_y = target_x, target_y
            return

        # === 2. 计算"过冲"目标点 (Overshoot Target) ===
        # 距离越远，速度越快，过冲概率和距离越大
        # 我们故意让鼠标瞄准目标稍微偏后的位置
        overshoot_amount = 0
        if total_dist > 100:
            # 随机过冲 10px - 50px，或者偶尔不过冲
            if random.random() > 0.2: 
                overshoot_amount = random.uniform(10, min(total_dist * 0.1, 60))
        
        # 计算过冲坐标
        if overshoot_amount > 0:
            # 向量归一化
            dx = (target_x - start_x) / total_dist
            dy = (target_y - start_y) / total_dist
            # 加上一点随机角度偏移 (手抖)
            angle_jitter = random.uniform(-0.15, 0.15) # 弧度
            
            # 旋转向量
            rotated_dx = dx * math.cos(angle_jitter) - dy * math.sin(angle_jitter)
            rotated_dy = dx * math.sin(angle_jitter) + dy * math.cos(angle_jitter)

            aim_x = target_x + rotated_dx * overshoot_amount
            aim_y = target_y + rotated_dy * overshoot_amount
        else:
            aim_x, aim_y = target_x, target_y

        # === 3. 定义分段移动逻辑 (内部函数) ===
        async def perform_bezier_move(s_x, s_y, e_x, e_y, speed_factor=1.0):
            dist = math.hypot(e_x - s_x, e_y - s_y)
            # 速度计算：这里调高了基准速度 (4000-8000 px/s)，模拟快速甩动
            # speed_factor 用于控制修正阶段要慢一点
            base_speed = random.uniform(4000, 8000) * speed_factor
            duration = max(0.15, dist / base_speed) 
            
            steps = max(5, int(duration * 120)) # 120hz 采样保证平滑

            # 贝塞尔控制点 (制造弧线)
            mid_x = (s_x + e_x) / 2
            mid_y = (s_y + e_y) / 2
            offset = min(dist * 0.3, 150) * random.choice([1, -1])
            ctrl_x = mid_x + random.uniform(-offset, offset)
            ctrl_y = mid_y + random.uniform(-offset, offset)

            for i in range(steps + 1):
                t = i / steps
                eased_t = self._ease_out_quart(t) # 使用急停曲线

                u = 1 - eased_t
                tt = eased_t * eased_t
                uu = u * u

                curr_x = (uu * s_x) + (2 * u * eased_t * ctrl_x) + (tt * e_x)
                curr_y = (uu * s_y) + (2 * u * eased_t * ctrl_y) + (tt * e_y)
                
                await self.page.mouse.move(curr_x, curr_y)
                await self._set_cursor_pos(curr_x, curr_y)
                
                # 极短的 sleep
                await asyncio.sleep(duration / steps)
            
            return e_x, e_y

        # === 4. 执行移动 (包含停顿逻辑) ===
        
        curr_x, curr_y = start_x, start_y

        # [逻辑] 如果距离非常长 (>1200px)，模拟手指不够长了，中途停顿一次 (Clutching)
        if total_dist > 1200 and random.random() > 0.3:
            # 移动到总路程的 60%-80% 处
            stop_ratio = random.uniform(0.6, 0.8)
            stop_x = start_x + (aim_x - start_x) * stop_ratio + random.uniform(-50, 50)
            stop_y = start_y + (aim_y - start_y) * stop_ratio + random.uniform(-50, 50)
            
            # 第一段：快速甩过去
            curr_x, curr_y = await perform_bezier_move(curr_x, curr_y, stop_x, stop_y, speed_factor=1.2)
            
            # 中途停顿：模拟提起鼠标或手指重定位 (0.05s - 0.15s)
            await asyncio.sleep(random.uniform(0.05, 0.15))

        # 第二段 (或第一段)：移动到"瞄准点" (可能是过冲点)
        curr_x, curr_y = await perform_bezier_move(curr_x, curr_y, aim_x, aim_y, speed_factor=1.0)

        # === 5. 回正逻辑 (Correction) ===
        # 如果有过冲，或者瞄准点和真实目标有偏差，进行最后一次慢速修正
        dist_to_real = math.hypot(target_x - curr_x, target_y - curr_y)
        if dist_to_real > 3:
            # 视觉确认停顿：人眼确认"哎呀移过了"，然后修正 (0.02s - 0.1s)
            await asyncio.sleep(random.uniform(0.02, 0.1))
            
            # 慢速回正 (speed_factor=0.3)
            # 这里不用贝塞尔，直接线性或简单平滑过去即可，因为距离很短
            correction_steps = max(5, int(dist_to_real / 2))
            for i in range(correction_steps + 1):
                t = i / correction_steps
                # 简单的 ease out
                t = 1 - (1-t)*(1-t) 
                
                final_x = curr_x + (target_x - curr_x) * t
                final_y = curr_y + (target_y - curr_y) * t
                await self.page.mouse.move(final_x, final_y)
                await self._set_cursor_pos(final_x, final_y)
                await asyncio.sleep(0.005)

        # 强制归位
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
                    print("[CLICK] Cursor visually missing, resetting to viewport center...")
                    await self._inject_cursor_overlay()
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

                center = await self._smooth_scroll_into_view(elem)
                if not center:
                    print(f"[ENTER] Element not visible/bbox missing: {xpath}")
                    return False
                tx, ty = center

                # 移动鼠标并点击聚焦
                await self._move_mouse_natural(tx, ty)
                await self.page.mouse.click(tx, ty)

                # 数字处理逻辑
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

                # === ✅ 伪造逐字输入 + 绿色状态条 (智能中英文区分版) ===
                
                await elem.fill("") 
                
                # 1. 制定打字计划 (预计算延迟)
                # 我们先生成每个字的延迟时间，这样能精确计算绿条需要亮多久
                typing_plan = []
                total_duration_ms = 500 # 基础缓冲时间(ms)

                for char in fill_content:
                    # 判断是否为汉字 (Unicode 范围 \u4e00-\u9fff)
                    if '\u4e00' <= char <= '\u9fff':
                        # 中文：慢速，模拟拼音选词思考时间 (0.5s - 2.0s)
                        delay = random.uniform(0.5, 2.0)
                    else:
                        # 英文/数字/符号：快速 (0.03s - 0.3s)
                        delay = random.uniform(0.03, 0.3)
                    
                    typing_plan.append((char, delay))
                    # 累加时间用于设置绿条 (秒转毫秒，加一点额外渲染开销缓冲)
                    total_duration_ms += (delay * 1000) + 50

                # 2. 启动绿色长脉冲 (使用计算好的总时长)
                await self._debug_bar_pulse("TYPING", active_ms=total_duration_ms, color="#00cc00")

                # 3. 执行打字
                for char, delay in typing_plan:
                    await elem.type(char)
                    await asyncio.sleep(delay)

                # === ✅ 修改结束 ===

                await asyncio.sleep(0.1)
                
                await self._reload_tools_if_needed(start_url)

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

                await self._move_mouse_natural(tx, ty)
                await self._debug_bar_pulse("SELECTT", active_ms=160)
                time.sleep(1)
                await self.page.mouse.click(tx, ty)

                try:
                    await self.page.evaluate("() => window.__webenvClickFrame && window.__webenvClickFrame.promise")
                except Exception:
                    pass

                async def finish_action(action_key, val=None):
                    await asyncio.sleep(0.1)
                    await self._reload_tools_if_needed(start_url)

                    try:
                        self._last_url = self.page.url
                    except:
                        pass

                    rec = (action_key, unique_id) if val is None else (action_key, unique_id, val)
                    self.click_history.append(rec)
                    return True, t_start

                if tag == "select":
                    try:
                        result = await elem.select_option(str(value))
                        if result and result != []:
                            return await finish_action("select", value)

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
        async with self._page_lock:
            """
            开始录制（切换到启用 record_video 的新 context/page），后续 reset/click/enter/select/load_url 等都不会中断录制。
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

            cx, cy = await self._get_viewport_center()
            await self.page.mouse.move(cx, cy)
            await self._set_cursor_pos(cx, cy)
            self._mouse_x, self._mouse_y = cx, cy

            self.recording = True
            print(f"[VIDEO] Started. dir={self._video_tmpdir}, size={record_size}")
            return True

    async def end_video(self) -> Optional[Dict[str, str]]:
        async with self._page_lock:
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

            if output:
                self.video_segments.append(output)
            return output

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
            # 1. 优先清理可能残留的录制目录
            if self._video_tmpdir and os.path.isdir(self._video_tmpdir):
                print(f"[CLEANUP] Removing residual video dir: {self._video_tmpdir}")
                try:
                    shutil.rmtree(self._video_tmpdir, ignore_errors=True)
                except Exception as e:
                    print(f"[CLEANUP] Failed to remove dir: {e}")
            self._video_tmpdir = None
            self.recording = False

            # 2. 关闭浏览器资源
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
