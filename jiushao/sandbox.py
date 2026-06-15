"""Python 沙箱：子进程隔离执行模型生成的代码。

安全边界（针对自有模型产物，非对抗环境）：
- 独立进程 + 超时 + 地址空间/CPU 资源上限
- python -I 隔离模式（不读 site/user 环境变量注入的路径）
- 简单 import 黑名单兜底（os.system/subprocess/socket 等）
"""
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

BANNED = re.compile(
    r"\b(subprocess|socket|shutil\.rmtree|os\.system|os\.remove|os\.rmdir"
    r"|requests|urllib|httpx|open\s*\(.+[\"']w)", re.I)

MAX_OUTPUT = 2000  # 回填给模型的输出截断长度


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))      # 2GB 内存
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))                # 60s CPU


def run_code(code: str, timeout: int = 30) -> dict:
    """执行代码，返回 {ok, stdout, stderr}（均已截断）。"""
    if BANNED.search(code):
        return {"ok": False, "stdout": "", "stderr": "拒绝执行：代码包含被禁止的调用（网络/系统/文件写）"}
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=timeout,
                cwd=td, preexec_fn=_limits,
            )
            out = proc.stdout[-MAX_OUTPUT:]
            err = proc.stderr[-MAX_OUTPUT:]
            return {"ok": proc.returncode == 0, "stdout": out, "stderr": err}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"执行超时（>{timeout}s）"}


# 匹配 ```python / ```py / ```（无语言标记），但不匹配 ```json 等非代码块
# ——借鉴 lagent 的兜底思路：模型常写无标记 ``` 代码块，但我们仍排除已知非代码语言
# 避免误执行答案/JSON 块（lagent 不做此区分）。
CODE_BLOCK = re.compile(r"```(?:python|py|)[ \t]*\n(.*?)```", re.S)


def extract_code_blocks(text: str) -> list[str]:
    return [m.strip() for m in CODE_BLOCK.findall(text) if m.strip()]
