"""judge 单元测试：答案抽取 + 等价判定。"""
from jiushao.judge import equiv, extract_answer, extract_boxed


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"所以 \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{3}}") == r"\frac{1}{3}"

    def test_takes_last(self):
        assert extract_boxed(r"\boxed{1} ... \boxed{2}") == "2"

    def test_missing(self):
        assert extract_boxed("没有答案") is None

    def test_unclosed(self):
        assert extract_boxed(r"\boxed{1+2") is None


class TestExtractAnswer:
    def test_prefers_boxed(self):
        assert extract_answer(r"答案是 5。\boxed{6}") == "6"

    def test_answer_keyword(self):
        assert extract_answer("推理过程……\n最终答案：42") == "42"

    def test_fallback_last_line(self):
        assert extract_answer("第一行\n第二行结论") == "第二行结论"


class TestEquiv:
    def test_fraction_vs_decimal(self):
        assert equiv("1/2", "0.5")

    def test_latex_vs_plain(self):
        assert equiv(r"\frac{\pi}{4}", "pi/4")
        assert equiv("sqrt(2)/2", r"\frac{\sqrt{2}}{2}")

    def test_numeric_tolerance(self):
        assert equiv("3.14159", "3.1416")
        assert not equiv("3.14", "3.5")

    def test_bool(self):
        assert equiv("True", "true")
        assert equiv("yes", "True")
        assert not equiv("True", "False")

    def test_negative(self):
        assert not equiv("2", "3")
        assert not equiv(None, "1")
        assert not equiv("1", None)
        assert not equiv("", "")

    def test_latex_interval(self):
        assert equiv(r"\left(3, \frac{\pi}{2}\right)", r"(3, \frac{\pi}{2})")
