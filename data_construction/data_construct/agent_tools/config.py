import os
import argparse
import json
import re
from pathlib import Path

def load_config():
    parser = argparse.ArgumentParser(description="WEBVIA Exploration Pipeline (HTML / URL)")
    parser.add_argument("--config", type=str, default=None, help="路径配置文件(config.json)")
    # 通用
    parser.add_argument("--image_dir", type=str, help="输出图像与日志目录")
    parser.add_argument("--bug_dir", type=str, help="异常HTML保存目录（仅HTML模式使用）")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--api_base", type=str, help="OpenAI Base URL")
    parser.add_argument("--model_name", type=str, help="使用的模型名称")
    parser.add_argument("--num_port", type=int, help="并行端口数")
    parser.add_argument("--port_base", type=int, help="基础端口号")
    parser.add_argument("--input_type", type=str, choices=["html", "url"], help="输入类型：html 或 url")

    # HTML 模式
    parser.add_argument("--input_dir", type=str, help="输入HTML文件目录")

    # URL 模式
    parser.add_argument("--url_file", type=str, help="包含URL列表的文本文件（每行一个）")

    args = parser.parse_args()

    config = {}
    config_path = os.path.abspath(os.path.expanduser(args.config)) if args.config else None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    config_dir = os.path.dirname(config_path) if config_path else os.getcwd()

    # 命令行优先级高于配置文件
    def get(key, default=None, legacy_key=None, env_key=None):
        val = getattr(args, key, None)
        if val is not None:
            return val
        if env_key and os.environ.get(env_key) is not None:
            return os.environ[env_key]
        if key in config:
            return config[key]
        if legacy_key and legacy_key in config:
            return config[legacy_key]
        return default

    # 统一路径格式
    def abs_path(p):
        if not p:
            return None
        expanded = os.path.expanduser(p)
        if not os.path.isabs(expanded):
            expanded = os.path.join(config_dir, expanded)
        return os.path.abspath(expanded)

    default_webenv = Path(__file__).resolve().parents[1] / "webenv-init" / "webenv.py"

    final_config = {
        "input_type": get("input_type", "html", env_key="VIDEO2CODE_INPUT_TYPE"),
        "input_dir": abs_path(get("input_dir", "./input_html", legacy_key="input_html_dir")),
        "url_file": abs_path(get("url_file", "./urls.txt", legacy_key="input_url_txt")),
        "image_dir": abs_path(get("image_dir", "./output_images")),
        "bug_dir": abs_path(get("bug_dir", "./trash_dir")),
        "api_key": get("api_key", "", env_key="OPENAI_API_KEY"),
        "api_base": get("api_base", "https://api.openai.com/v1", env_key="OPENAI_BASE_URL"),
        "model_name": get("model_name", "gpt-4.1", env_key="VIDEO2CODE_MODEL"),
        "webenv_py": abs_path(get("webenv_py", str(default_webenv), legacy_key="webenv_path")),
        "num_port": int(get("num_port", 4, env_key="VIDEO2CODE_NUM_WORKERS")),
        "port_base": int(get("port_base", 8000, env_key="VIDEO2CODE_PORT_BASE")),
    }
    # print(final_config)
    return final_config


class ConfigError(Exception):
    pass


def url_to_name(url: str) -> str:
    """把任意 URL 转成短而安全的文件名片段"""
    return re.sub(r'\W+', '_', url)[:40]

def _is_writable_dir(path: str) -> tuple[bool, str | None]:
    """
    返回 (是否可写, 失败原因)
    """
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, ".write_test___")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def validate_config(cfg: dict) -> dict:
    """
    校验必要项；若有问题，统一收集并一次性抛错（逐条指出）。
    返回经过规范化（绝对路径等）的 cfg。
    """
    errors = []
    # print(cfg)
    # 统一绝对路径
    def abs_path(p):
        if p is None:
            return None
        return os.path.abspath(os.path.expanduser(p))

    cfg["image_dir"] = abs_path(cfg.get("image_dir"))
    cfg["bug_dir"]   = abs_path(cfg.get("bug_dir"))  # 仅 HTML 用，但先校验可写
    cfg["input_dir"] = abs_path(cfg.get("input_dir"))
    cfg["url_file"]  = abs_path(cfg.get("url_file"))
    webenv_py = abs_path(cfg.get("webenv_py"))

    # 可选：webenv 脚本存在性（你的代码依赖这个文件）
    if not os.path.isfile(webenv_py):
        errors.append(f"[webenv_init] 缺少环境脚本：{webenv_py}（请确认路径或调整启动命令）")
    else:
        cfg["webenv_py"] = webenv_py  # 供后续使用

    # 2) 模型与 OpenAI 相关
    model_name = cfg.get("model_name", "")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append(f"[model_name] 模型名称不能为空（实得：{model_name!r}）")

    api_key = cfg.get("api_key", "")
    if not isinstance(api_key, str) or not api_key.strip():
        errors.append("[api_key] 不能为空（建议通过 --api_key 或 config.json 提供）")

    api_base = cfg.get("api_base", "")
    if not isinstance(api_base, str) or not api_base.strip():
        errors.append("[api_base] 不能为空（例如：https://api.openai.com/v1）")

    # 3) 端口设置
    try:
        num_port = int(cfg.get("num_port", 0))
        if num_port < 1:
            errors.append(f"[num_port] 必须为 >= 1 的整数（实得：{cfg.get('num_port')!r}）")
    except Exception:
        errors.append(f"[num_port] 非法整数：{cfg.get('num_port')!r}")

    try:
        port_base = int(cfg.get("port_base", -1))
        if not (1 <= port_base <= 65535):
            errors.append(f"[port_base] 必须位于 1..65535（实得：{cfg.get('port_base')!r}）")
        else:
            # 端口区间不应越界
            last_port = port_base + int(cfg.get("num_port", 0)) - 1
            if last_port > 65535:
                errors.append(
                    f"[port_base + num_port] 端口区间越界：{port_base}..{last_port}（最大65535）"
                )
    except Exception:
        errors.append(f"[port_base] 非法整数：{cfg.get('port_base')!r}")

    # 4) 按 input_type 做差异化校验
    input_type = (cfg.get("input_type") or "html").lower()
    if input_type not in ("html", "url"):
        errors.append(f"[input_type] 仅支持 'html' 或 'url'（实得：{cfg.get('input_type')!r}）")
    else:
        if input_type == "html":
            if not cfg["input_dir"] or not os.path.isdir(cfg["input_dir"]):
                errors.append(f"[input_dir] 目录不存在或不可访问：{cfg.get('input_dir')!r}")
            else:
                html_files = [fn for fn in os.listdir(cfg["input_dir"]) if fn.endswith(".html")]
                if len(html_files) == 0:
                    errors.append(f"[input_dir] 目录存在但没有任何 .html 文件：{cfg['input_dir']}")
            # bug_dir 在 HTML 模式下需要可写
            if not cfg["bug_dir"]:
                errors.append("[bug_dir] 未提供异常HTML保存目录路径")
            else:
                ok, reason = _is_writable_dir(cfg["bug_dir"])
                # if not ok:
                #     errors.append(f"[bug_dir] 无法创建/写入：{cfg['bug_dir']}（{reason}）")
        else:  # url
            if not cfg["url_file"] or not os.path.isfile(cfg["url_file"]):
                errors.append(f"[url_file] URL 列表文件不存在或不可访问：{cfg.get('url_file')!r}")
            else:
                with open(cfg["url_file"], "r", encoding="utf-8") as f:
                    urls = [ln.strip() for ln in f if ln.strip()]
                if len(urls) == 0:
                    errors.append(f"[url_file] 文件存在但未包含任何 URL：{cfg['url_file']}")

    # image_dir 通用校验
    if not cfg["image_dir"]:
        errors.append("[image_dir] 未提供输出目录路径")
    else:
        ok, reason = _is_writable_dir(cfg["image_dir"])
        # if not ok:
        #     errors.append(f"[image_dir] 无法创建/写入：{cfg['image_dir']}（{reason}）")

    # 汇总报错
    if errors:
        bullet = "\n  - "
        msg = "配置校验失败，需修正以下问题：" + bullet + bullet.join(errors)
        raise ConfigError(msg)

    return cfg
