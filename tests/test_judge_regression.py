"""判分器回归测试：每条用例来自 debug 采样中的真实误判（修一个固化一个）。"""
from jiushao.judge import equiv


class TestSymbolicVsDecimal:
    """符号表达式 vs 小数 gold（TheoremQA 高频形态）。"""

    def test_pi_fraction(self):                       # theoremqa-7
        assert equiv(r"-\dfrac{\pi}{3}", "-1.047", rel_tol=1e-2)

    def test_exp_expression(self):                    # theoremqa-19
        assert equiv(r"e^{5/4} - e^{3/4}", "1.3733", rel_tol=1e-2)

    def test_not_close(self):
        assert not equiv(r"\frac{\pi}{2}", "-1.047", rel_tol=1e-2)


class TestUnits:
    def test_text_units(self):                        # theoremqa-8
        assert equiv(r"0.954 \text{ ft}^3", "0.955", rel_tol=1e-2)

    def test_plain_units(self):                       # theoremqa-28
        assert equiv(r"1.3038\ \text{m}", "1.3", rel_tol=1e-2)


class TestListForms:
    def test_fraction_list_vs_decimal(self):          # theoremqa-13
        assert equiv(r"[\frac{1}{3}, \frac{1}{4}]", "[0.333, 0.25]", rel_tol=1e-2)

    def test_pmatrix_vs_list(self):                   # theoremqa-15
        assert equiv(r"\begin{pmatrix}4\\2\end{pmatrix}", "[4, 2]")

    def test_set_braces_with_prefix(self):            # theoremqa-25
        assert equiv(r"c \in \{-1, 6\}", "[-1, 6]")

    def test_wrong_list_rejected(self):               # theoremqa-9 真错，不能放过
        assert not equiv("[0,4]", "[1, 1]")

    def test_length_mismatch(self):
        assert not equiv("[1, 2, 3]", "[1, 2]")


class TestBool:
    def test_yes_with_explanation(self):              # theoremqa-18
        assert equiv("Yes. For example, x = (0, 1, 2, ...) satisfies T(x)=...", "True")

    def test_chinese_negative(self):                  # theoremqa-12
        assert equiv(r"W 不是 \mathbb{R}^2 的子空间。原因如下…", "False")

    def test_chinese_negative_not_true(self):
        assert not equiv(r"W 不是子空间", "True")


class TestVarPrefixAndApprox:
    def test_var_eq_with_approx(self):                # theoremqa-20
        assert equiv(r"h_{\max} = \frac{1}{8}(3\log_2 3 + 4) \approx 1.094360", "1.094",
                     rel_tol=1e-2)


class TestGoldWrappers:
    def test_gold_with_boxed_wrapper(self):           # hardmath gold 形态
        assert equiv("1/3", "$$\\boxed{\\frac{1}{3}}$$")

    def test_gold_boxed_list(self):
        assert equiv(r"[\frac{1.0}{\epsilon^{0.93}}, \frac{0.67}{\epsilon^{0.75}}]",
                     "\n  $$\\boxed{[\\frac{1.0}{\\epsilon^{0.93}}, \\frac{0.67}{\\epsilon^{0.75}}]}$$")


class TestNoRegression:
    """v1 已覆盖的能力不能退化。"""

    def test_basic(self):
        assert equiv("1/2", "0.5")
        assert equiv(r"\frac{\pi}{4}", "pi/4")
        assert not equiv("2", "3")
        assert equiv("90", r"90^\circ")
        assert equiv("Evelyn", r"\text{Evelyn}")
        assert equiv(r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})")


class TestHardmathRegime:
    """HARDMath regime 数值点判分（judge.hardmath_equiv）。"""

    HM = {"small_eval": 0.001, "small_val": "177.83",
          "large_eval": 739072203.35, "large_val": "4.74e-09"}

    def test_gold_own_expr_passes(self):
        from jiushao.judge import hardmath_equiv
        # gold 自己的表达式（与 analytical 差 ~40%）必须在因子窗口内通过
        assert hardmath_equiv(
            r"[\frac{1.0}{\epsilon^{0.93}}, \frac{0.67}{\epsilon^{0.75}}]",
            "whatever", self.HM)

    def test_wrong_order_rejected(self):
        from jiushao.judge import hardmath_equiv
        # 数量级错误的表达式必须拒绝
        assert not hardmath_equiv(r"[\epsilon^{2}, \epsilon^{3}]", "x", self.HM)

    def test_none_rejected(self):
        from jiushao.judge import hardmath_equiv
        assert not hardmath_equiv(None, "x", self.HM)


class TestJudgeWithMeta:
    def test_dispatch_hardmath(self):
        from jiushao.judge import judge_with_meta
        meta = {"hardmath": {"small_eval": 1e-3, "small_val": "177.83",
                             "large_eval": None, "large_val": None}}
        assert judge_with_meta(r"\frac{1.0}{\epsilon^{0.75}}", "x", meta)

    def test_dispatch_default(self):
        from jiushao.judge import judge_with_meta
        assert judge_with_meta("0.5", "1/2", None)
        assert judge_with_meta("1.428", "1.42", {"rel_tol": 1e-2})


class TestAsymptoticPrefix:
    """渐近式/函数前缀提取（裁判过滤真实场景暴露）。"""

    def test_sim_prefix_stripped(self):
        # I(x) \sim π/(2√2) ε^{-3/4} 应能与纯表达式判等
        assert equiv(r"I(x) \sim \frac{\pi}{2\sqrt{2}} \epsilon^{-3/4}",
                     r"\frac{\pi}{2\sqrt{2}} \epsilon^{-3/4}")

    def test_func_paren_prefix(self):
        assert equiv(r"I(\epsilon) = \frac{1}{\epsilon}", r"\frac{1}{\epsilon}")
