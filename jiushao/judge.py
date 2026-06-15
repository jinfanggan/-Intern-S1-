"""答案抽取与等价判定 v2。

判定流程（逐级兜底，全部由 debug 采样的真实失败案例驱动）：
  0. 预处理生成候选串（去 $$/boxed/单位/变量前缀/approx 拆分/矩阵集合归一）
  1. 布尔题：gold 为 True/False 时在 pred 原文中找极性词（Yes/No/不是/存在…）
  2. math-verify（LaTeX/区间/集合）
  3. SymPy 表达式：先 simplify 判恒等，再 evalf 数值比较（符号 vs 小数）
  4. 列表逐元素比较（[1/3, 1/4] vs [0.333, 0.25]、pmatrix vs [4,2]）
  5. 纯数值容差
  6. 规范化字符串
"""
import re

from math_verify import parse as mv_parse, verify as mv_verify

# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """抽取最后一个 \\boxed{...}（支持嵌套花括号）。"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def extract_answer(text: str) -> str | None:
    """优先 boxed；否则取「最终答案/answer is」之后的内容；再否则最后一行。"""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"(?:最终答案|答案是|answer is|Answer)[:：\s]*(.+)", text)
    if m:
        return m.group(1).strip().rstrip("。.")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------
# 预处理：候选串生成
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    b = extract_boxed(s)                       # gold 自带 boxed 包装（HARDMath）
    if b is not None:
        s = b.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\d?frac", r"\\frac", s)      # \dfrac/\tfrac → \frac
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", s)  # \text{X} → X（解包）
    s = re.sub(r"\\[,;!\s]", " ", s)           # LaTeX 空白转义 \, \; \&nbsp
    # 矩阵 → 列表
    s = re.sub(r"\\begin\{[pbv]matrix\}(.*?)\\end\{[pbv]matrix\}",
               lambda m: "[" + m.group(1).replace("\\\\", ",") + "]", s, flags=re.S)
    # 集合花括号 → 列表
    if re.fullmatch(r"\\\{.*\\\}", s.strip()):
        s = "[" + s.strip()[2:-2] + "]"
    return s.strip().rstrip("。.,;")


def _candidates(s: str) -> list[str]:
    out, base = [], _clean(s)
    out.append(base)
    if "\\approx" in base:                      # h = expr \approx 1.094 → 两段都试
        parts = base.split("\\approx")
        out.extend(p.strip() for p in parts if p.strip())
    if "\\sim" in base:                         # 渐近式 I(x) \sim expr → 取右边
        out.append(_clean(base.split("\\sim")[-1]))
    # 函数名/变量前缀：c \in {...} / x = ... / h_{max} = ... / I(\epsilon) = ...
    m = re.match(r"^\\?[A-Za-z][A-Za-z_0-9{}\\()]*\s*(?:=|\\in)\s*(.+)$", base, re.S)
    if m:
        out.append(_clean(m.group(1)))
    # 数值 + 单位后缀（0.954 ft^3 / 1.3038 m / 2.98 Hz）→ 数字前缀单独作候选
    m = re.match(r"^([-+]?\d[\d.,]*(?:[eE][-+]?\d+)?)\s*\S", base)
    if m and m.group(1) != base:
        out.append(m.group(1))
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------------------
# 各级判定
# ---------------------------------------------------------------------------

_FALSE_WORDS = r"(?:false|no\b|not\b|incorrect|不是|不存在|不成立|不可以|不能|否|错误)"
_TRUE_WORDS = r"(?:true|yes\b|correct|是|存在|成立|可以|能|正确)"


def _bool_of(text: str) -> str | None:
    t = text.lower()
    if re.search(_FALSE_WORDS, t):
        return "false"
    if re.search(_TRUE_WORDS, t):
        return "true"
    return None


def _to_float(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip(".")
    s = re.sub(r"[a-zA-Z\$%\s°]+$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _expr_of(s: str):
    """答案串 → SymPy 表达式：纯文本走 sympify，LaTeX 走 math-verify。"""
    import sympy
    if "\\" not in s and "{" not in s:
        try:
            return sympy.sympify(s.replace("^", "**"), rational=True)
        except Exception:
            pass
    try:
        for cand in mv_parse(f"\\boxed{{{s}}}"):
            if not isinstance(cand, str):
                return cand
    except Exception:
        pass
    return None


def _num_close(a: float, b: float, rel_tol: float) -> bool:
    if b == 0:
        return abs(a) < max(rel_tol, 1e-6)
    return abs(a - b) / abs(b) < rel_tol


def _expr_equiv(p: str, g: str, rel_tol: float) -> bool:
    import sympy
    ep, eg = _expr_of(p), _expr_of(g)
    if ep is None or eg is None:
        return False
    try:
        if sympy.simplify(ep - eg) == 0:
            return True
    except Exception:
        pass
    try:  # 符号 vs 小数：数值求值比较（-pi/3 vs -1.047）
        fp, fg = float(ep.evalf()), float(eg.evalf())
        return _num_close(fp, fg, max(rel_tol, 5e-3))
    except Exception:
        return False


def _split_list(s: str) -> list[str] | None:
    s = s.strip()
    if not (len(s) >= 2 and s[0] in "[(" and s[-1] in "])"):
        return None
    inner, parts, depth, cur = s[1:-1], [], 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()] or None


def _scalar_equiv(p: str, g: str, rel_tol: float) -> bool:
    try:
        if mv_verify(mv_parse(f"\\boxed{{{g}}}"), mv_parse(f"\\boxed{{{p}}}")):
            return True
    except Exception:
        pass
    if _expr_equiv(p, g, rel_tol):
        return True
    fp, fg = _to_float(p), _to_float(g)
    if fp is not None and fg is not None:
        return _num_close(fp, fg, rel_tol)
    return _norm_str(p) == _norm_str(g)


def _norm_str(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\$\\,]+", "", s)
    s = s.replace("left", "").replace("right", "").replace("{", "").replace("}", "")
    return s


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def equiv(pred: str | None, gold: str | None, rel_tol: float = 1e-4) -> bool:
    """pred 与 gold 是否数学等价。"""
    if pred is None or gold is None:
        return False
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred or not gold or pred.lower() == "none":
        return False

    # 布尔题：gold 是 True/False 时，在 pred 全文找极性（Yes, For example... / W 不是子空间）
    if gold.lower() in ("true", "false"):
        bp = _bool_of(pred)
        if bp is not None:
            return bp == gold.lower()
        return False

    for p in _candidates(pred):
        for g in _candidates(gold):
            # 列表/向量/集合：逐元素比较
            lp, lg = _split_list(p), _split_list(g)
            if lp is not None and lg is not None:
                if len(lp) == len(lg) and all(
                        _scalar_equiv(a, b, rel_tol) for a, b in zip(lp, lg)):
                    return True
                continue
            if _scalar_equiv(p, g, rel_tol):
                return True
    return False


# ---------------------------------------------------------------------------
# HARDMath 专用：渐近表达式按 regime 数值点判分
# 注意：gold 自身的渐近式在评测点上与真值可差 ~40%（首阶近似的本性），
# 因此用「因子窗口」而非严格容差；阈值待与官方 eval 脚本对齐后收紧。
# ---------------------------------------------------------------------------

def _eval_at(expr_str: str, point: float) -> complex | None:
    """表达式在数值点求值（单自由变量自动替换）。"""
    e = _expr_of(expr_str)
    if e is None:
        return None
    try:
        syms = list(e.free_symbols)
        if len(syms) == 1:
            e = e.subs(syms[0], point)
        elif len(syms) > 1:
            return None
        return complex(e.evalf())
    except Exception:
        return None


def _within_factor(a: complex, b: complex, factor: float = 3.0) -> bool:
    ma, mb = abs(a), abs(b)
    if mb == 0:
        return ma < 1e-9
    if ma == 0:
        return False
    return (1 / factor) < (ma / mb) < factor and abs(a - b) / mb < factor


def hardmath_equiv(pred: str | None, gold: str, hm: dict) -> bool:
    """先试符号等价（严格），再按 regime 数值点因子窗口判（宽松）。"""
    if pred is None or not str(pred).strip():
        return False
    pred = str(pred)
    if equiv(pred, gold, rel_tol=1e-2):          # 严格路径：符号等价
        return True
    # 收集 pred 候选元素（列表拆开 + 整体）
    elements: list[str] = []
    for c in _candidates(pred):
        elements.append(c)
        elements.extend(_split_list(c) or [])
    regimes = []
    for key in ("small", "large"):
        pt, val = hm.get(f"{key}_eval"), hm.get(f"{key}_val")
        if pt is not None and val not in (None, ""):
            try:
                regimes.append((float(pt), complex(float(val))))
            except (TypeError, ValueError):
                try:
                    import sympy
                    regimes.append((float(pt), complex(sympy.sympify(str(val)).evalf())))
                except Exception:
                    continue
    if not regimes:
        return False
    for pt, target in regimes:
        vals = [_eval_at(el, pt) for el in elements]
        if not any(v is not None and _within_factor(v, target) for v in vals):
            return False
    return True


def judge_with_meta(pred: str | None, gold: str, meta: dict | None) -> bool:
    """统一判分入口：pipeline 与 rejudge 共用。"""
    meta = meta or {}
    if "hardmath" in meta:
        return hardmath_equiv(pred, gold, meta["hardmath"])
    return equiv(pred, gold, rel_tol=meta.get("rel_tol", 1e-4))
