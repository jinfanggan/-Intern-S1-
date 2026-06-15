"""sandbox 单元测试：执行、超时、黑名单、代码块抽取。"""
from jiushao.sandbox import extract_code_blocks, run_code


class TestRunCode:
    def test_basic_exec(self):
        r = run_code("print(1 + 1)")
        assert r["ok"] and "2" in r["stdout"]

    def test_sympy_available(self):
        r = run_code("import sympy as sp\nx = sp.Symbol('x')\n"
                     "print(sp.integrate(x**2, (x, 0, 1)))")
        assert r["ok"] and "1/3" in r["stdout"]

    def test_error_returns_traceback(self):
        r = run_code("1 / 0")
        assert not r["ok"] and "ZeroDivisionError" in r["stderr"]

    def test_timeout(self):
        r = run_code("while True: pass", timeout=2)
        assert not r["ok"] and "超时" in r["stderr"]

    def test_banned_socket(self):
        r = run_code("import socket\nsocket.socket()")
        assert not r["ok"] and "禁止" in r["stderr"]

    def test_banned_subprocess(self):
        r = run_code("import subprocess")
        assert not r["ok"]

    def test_output_truncated(self):
        r = run_code("print('x' * 100000)")
        assert r["ok"] and len(r["stdout"]) <= 2000


class TestExtractCodeBlocks:
    def test_single(self):
        text = "思路……\n```python\nprint(1)\n```\n继续"
        assert extract_code_blocks(text) == ["print(1)"]

    def test_multiple(self):
        text = "```python\na = 1\n```\n```py\nb = 2\n```"
        assert extract_code_blocks(text) == ["a = 1", "b = 2"]

    def test_ignores_non_python(self):
        assert extract_code_blocks("```json\n{}\n```") == []

    def test_none(self):
        assert extract_code_blocks("没有代码") == []


class TestCodeBlockFallback:
    """借鉴 lagent：无语言标记的 ``` 也应提取，但 ```json 等仍排除。"""

    def test_unmarked_block_extracted(self):
        assert extract_code_blocks("```\nprint(1)\n```") == ["print(1)"]

    def test_python_marked_still_works(self):
        assert extract_code_blocks("```python\nx=1\n```") == ["x=1"]

    def test_json_block_still_excluded(self):
        assert extract_code_blocks('```json\n{"a":1}\n```') == []
