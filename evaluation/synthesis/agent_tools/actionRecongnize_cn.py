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
import openai
from openai import OpenAI

from agent_tools import helpers

class ActionRecognizer:
    """通过大模型分析视频帧，识别交互动作并映射到 DOM 元素。"""

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _call_model(self, messages: list[dict]) -> str:
        """统一的模型调用接口，带重试。"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=8000,
                    timeout=600,
                )
                return str(response.choices[0].message.content)
            except Exception as e:
                print(f"[WARN] 模型调用失败 (attempt {attempt + 1}): {e}")
                time.sleep(2)
        return ""

    def analyze_frame_pair(
        self,
        frame_before_b64: str,
        frame_after_b64: str,
        domtree: str,
        page_screenshot_b64: str,
        step_index: int,
        history_actions: list[dict],
    ) -> tuple[list[dict], str]:
        """
        分析相邻两帧，识别发生的交互操作。

        参数:
            frame_before_b64: 交互前帧的 base64
            frame_after_b64: 交互后帧的 base64
            domtree: 复刻网页当前的 DOM Tree (字符串)
            page_screenshot_b64: 复刻网页当前截图 base64
            step_index: 当前步骤编号
            history_actions: 之前已识别执行的操作历史

        返回:
            (actions_list, model_raw_output)
        """
        history_desc = self._format_history(history_actions)

        prompt = f"""你是一个网页交互重放助手。你的任务是分析视频中的网页交互操作，并在一个复刻网页的 DOM Tree 中找到对应的元素，给出可执行的操作指令。

## 输入说明
你会收到以下内容:
1. **视频帧（交互前）**: 视频中某个交互操作发生之前的截图
2. **视频帧（交互后）**: 同一交互操作发生之后的截图
3. **复刻网页截图**: 当前复刻网页的实时截图
4. **复刻网页 DOM Tree**: 当前复刻网页的结构化 DOM 信息

## DOM Tree 关键字段说明
每个可交互元素包含:
- id: 元素唯一标识（用于操作指令中引用）
- tag: HTML 标签（BUTTON, INPUT, SELECT, A 等）
- attrs: HTML 属性字典
- visible_text: 元素上显示的文字
- address: 语义区域和在页面上的绝对坐标 "zone--[x,y,w,h]"
- can_interact: 是否当前可交互（可见且未被遮挡）
- input_value: 输入框当前值（仅 input/textarea 有）
- options: 下拉选项列表（仅 select 有，包含 value/text/selected）

## 你的任务
1. **对比两帧视频截图**，判断在这两帧之间发生了什么交互操作
2. **在复刻网页的 DOM Tree 中定位**到对应的元素（通过对比元素的 visible_text、位置 address、tag 类型等信息）
3. **输出操作指令**

## 输出格式
请用 LaTeX \\boxed{{}} 包裹你的操作指令。格式如下:
- 点击操作: click[元素id]
- 输入操作: enter[元素id][输入内容]
- 选择操作: select[元素id][选项文本]
- 多个连续操作用逗号分隔，放在同一个 \\boxed{{}} 内

如果两帧之间**没有发生可识别的交互操作**（如只是页面加载、动画、或滚动），请回复: \\boxed{{无操作}}

## 历史操作记录（已在复刻网页上执行的操作）
{history_desc if history_desc else "暂无历史操作"}

## 当前步骤: 第 {step_index + 1} 步

## DOM Tree
{domtree}

请仔细对比视频帧变化与 DOM Tree 元素，给出精确的操作指令。先简要说明你的分析过程，然后给出最终答案。"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "【视频帧 - 交互前】"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame_before_b64}"}},
            {"type": "text", "text": "【视频帧 - 交互后】"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame_after_b64}"}},
            {"type": "text", "text": "【复刻网页当前截图】"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{page_screenshot_b64}"}},
        ]

        messages = [{"role": "user", "content": content}]
        output = self._call_model(messages)

        if not output:
            return [], ""

        actions = self._parse_model_output(output)
        return actions, output

    def analyze_full_sequence(
        self,
        frames: list[dict],
        domtree: str,
        page_screenshot_b64: str,
    ) -> tuple[list[list[dict]], str]:
        """
        一次性发送所有帧，让模型识别完整的交互序列。
        适用于帧数较少（≤10帧）的场景。

        返回: (action_seqs, model_raw_output)
        """
        prompt = f"""你是一个网页交互重放助手。我会给你一组按时间顺序排列的视频截帧图片，它们记录了用户在网页上的一系列交互操作。
同时我会给你一个复刻网页的 DOM Tree 和当前截图。

## DOM Tree 关键字段
每个可交互元素包含: id（操作指令用的标识）, tag, attrs, visible_text, address（"zone--[x,y,w,h]"）, can_interact, input_value, options 等。

## 你的任务
1. 逐帧对比，识别每两帧之间发生的交互操作
2. 在 DOM Tree 中定位每个操作对应的元素（通过 visible_text、address、tag 等匹配）
3. 按时间顺序输出所有操作指令

## 输出格式
对于每一个识别出的操作，用 LaTeX \\boxed{{}} 包裹，格式:
- click[元素id]
- enter[元素id][输入内容]
- select[元素id][选项文本]
- 连续操作放在同一个 \\boxed{{}} 内，用逗号分隔

请按时间顺序排列所有 \\boxed{{}}。在每个操作前简述是在第几帧到第几帧之间发生的。

## DOM Tree
{domtree}

以下是视频帧序列和复刻网页截图:"""

        content = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            content.append({"type": "text", "text": f"【第 {i + 1} 帧】"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{frame['b64']}"},
            })
        content.append({"type": "text", "text": "【复刻网页当前截图】"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{page_screenshot_b64}"},
        })

        messages = [{"role": "user", "content": content}]
        output = self._call_model(messages)

        if not output:
            return [], ""

        action_seqs = self._parse_all_boxed_actions(output)
        return action_seqs, output

    def verify_action_result(
        self,
        frame_expected_b64: str,
        actual_screenshot_b64: str,
        action_desc: str,
    ) -> tuple[bool, str]:
        """
        验证执行操作后的实际截图是否与视频帧中期望的状态一致。
        """
        prompt = f"""请对比以下两张网页截图：
1. 第一张是视频中交互操作后的**期望状态**
2. 第二张是在复刻网页上执行操作后的**实际状态**

执行的操作是: {action_desc}

请判断两张截图的关键区域是否一致（忽略细微的样式差异，关注功能层面的变化是否匹配）。

回复格式:
- 如果一致，用 \\boxed{{一致}} 表示
- 如果不一致，用 \\boxed{{不一致}} 表示
然后简要说明理由。"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "【期望状态（视频帧）】"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame_expected_b64}"}},
            {"type": "text", "text": "【实际状态（复刻网页截图）】"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{actual_screenshot_b64}"}},
        ]

        messages = [{"role": "user", "content": content}]
        output = self._call_model(messages)

        boxed = helpers.extract_boxed_text(output) if output else ""
        is_match = "一致" in boxed and "不一致" not in boxed
        return is_match, output


    def verify_video_frames_sequence(
        self,
        expected_frames_b64: list[str],
        actual_frames_b64: list[str],
        sequence_desc: str = "",
    ) -> tuple[bool, str]:
        """
        验证复刻网页生成的视频抽帧序列是否与原始视频的抽帧序列在动态变化上保持一致。
        """
        prompt = f"""请对比以下两组连续的视频抽帧图像序列：
1. 第一组是**原始视频（期望状态）**的时间序列抽帧
2. 第二组是**复刻网页录屏（实际状态）**的时间序列抽帧

{"这两组序列展示的预期交互/动画过程是: " + sequence_desc if sequence_desc else ""}

请评估这两组序列所反映的**动态变化过程、UI 状态更迭（如弹窗、悬浮效果、页面滚动等）以及核心功能流**是否一致。

【评判标准】
- 允许两组视频在帧率、动画播放速度上存在差异（即状态改变可能发生在不同的帧数上）。
- 请忽略细微的像素级样式差异（如字体微调、颜色轻微偏差、抗锯齿边缘等）。
- 重点关注**起始状态、中间过渡逻辑以及最终状态**是否在语义和功能层面匹配。

回复格式:
- 如果动态变化流程整体一致，用 \\boxed{{一致}} 表示
- 如果存在关键流程缺失或错误导致不一致，用 \\boxed{{不一致}} 表示
然后简要说明你的对比推理过程。"""

        content = [{"type": "text", "text": prompt}]

        # 压入第一组：期望状态序列
        content.append({"type": "text", "text": f"======== 【第一组：期望状态（共 {len(expected_frames_b64)} 帧）】 ========"})
        for i, b64 in enumerate(expected_frames_b64):
            content.append({"type": "text", "text": f"期望状态 - 第 {i+1} 帧:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}"}})

        # 压入第二组：实际状态序列
        content.append({"type": "text", "text": f"======== 【第二组：实际状态（共 {len(actual_frames_b64)} 帧）】 ========"})
        for i, b64 in enumerate(actual_frames_b64):
            content.append({"type": "text", "text": f"实际状态 - 第 {i+1} 帧:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}"}})

        messages = [{"role": "user", "content": content}]
        
        # 调用模型获取输出
        output = self._call_model(messages)

        # 解析输出
        boxed = helpers.extract_boxed_text(output) if output else ""
        is_match = "一致" in boxed and "不一致" not in boxed
        
        return is_match, output
    # ---------- 辅助方法 ----------

    def _format_history(self, history_actions: list[dict]) -> str:
        if not history_actions:
            return ""
        lines = []
        for i, act in enumerate(history_actions):
            action_type = act.get("action", "unknown")
            elem_id = act.get("id", "?")
            value = act.get("value") or act.get("input_val") or ""
            dom_info = act.get("dom_info", {})
            text = ""
            if dom_info and isinstance(dom_info, dict):
                text = (dom_info.get("visible_text", "") or
                        dom_info.get("attrs", {}).get("text", "") or
                        dom_info.get("attrs", {}).get("value", ""))
            line = f"  步骤{i+1}: {action_type}[{elem_id}]"
            if value:
                line += f"[{value}]"
            if text:
                line += f"  (元素文本: {text})"
            lines.append(line)
        return "\n".join(lines)

    def _parse_model_output(self, output: str) -> list[dict]:
        """从模型输出中解析操作指令。"""
        boxed_raws = helpers.extract_boxed_actions(output)
        all_actions = []
        for bx in boxed_raws:
            if "无操作" in bx:
                continue
            parsed = helpers.parse_action_seq(bx)
            if parsed is not None:
                all_actions.extend(parsed)
        return all_actions

    def _parse_all_boxed_actions(self, output: str) -> list[list[dict]]:
        """从模型输出中解析所有 boxed 操作序列。"""
        boxed_raws = helpers.extract_boxed_actions(output)
        action_seqs = []
        for bx in boxed_raws:
            if "无操作" in bx:
                continue
            parsed = helpers.parse_action_seq(bx)
            if parsed is not None:
                action_seqs.append(parsed)
        return action_seqs
