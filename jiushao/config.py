"""配置中心：模型档案、价格表、路径。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
RUNS_ROOT = PROJECT_ROOT / "runs"


def load_env():
    """从项目根目录 .env 读取密钥（不依赖 python-dotenv）。"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

# ---------------------------------------------------------------------------
# 模型档案：屏蔽不同后端的参数差异
#   max_tokens_param : gpt-5/o 系用 max_completion_tokens，其余用 max_tokens
#   supports_temperature : gpt-5 reasoning 系不支持自定温度（采样多样性靠 prompt 扰动）
#   extra : 每次请求附带的额外 body（如书生 thinking_mode）
# ---------------------------------------------------------------------------
MODEL_PROFILES = {
    "gpt-5-nano": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False,
        "extra": {"reasoning_effort": "medium"},
        "price_in": 0.05, "price_out": 0.40,  # $/1M tokens
        "rpm": 480,
    },
    "gpt-5-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False,
        "extra": {"reasoning_effort": "medium"},
        "price_in": 0.25, "price_out": 2.00,
        "rpm": 480,
    },
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
        "extra": {},
        "price_in": 0.15, "price_out": 0.60,
        "rpm": 480,
    },
    # 比赛目标模型：拿到 key 后只需填 INTERN_API_KEY，无需改代码
    "intern-s1": {
        "base_url": "https://chat.intern-ai.org.cn/api/v1",
        "api_key_env": "INTERN_API_KEY",
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
        # 官方推荐采样参数（user_guide / HF card 核实）：temp=0.7, top_p=1.0, top_k=50
        # thinking 开关：自部署用 enable_thinking；书生云 API 字段待拿到 key 实测
        # （两种候选都列出，实测后删掉错的那个）
        "default_temperature": 0.7,
        "extra": {"thinking_mode": True, "top_p": 1.0},
        "supports_n": True,        # OpenAI 兼容，实测 gpt-5 系支持 n；Intern-S1 待确认
        "price_in": 0.0, "price_out": 0.0,
        "rpm": 28,  # 官方约 30 次/分，留余量
    },
    # 官方推荐主力模型（数学能力最强，已实测可用）。thinking 默认开，
    # 单次约 5000 token / 53s（实测），max_tokens 须给足否则思考链被截断。
    "intern-s2-preview": {
        "base_url": "https://chat.intern-ai.org.cn/api/v1",
        "api_key_env": "INTERN_API_KEY",
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
        "default_temperature": 0.7,
        "extra": {"top_p": 1.0},
        "supports_n": True,
        "price_in": 0.0, "price_out": 0.0,
        "rpm": 90,  # 申请 RPM 100，留余量；实际吞吐受单次 53s 墙钟限制
    },
}

DEFAULT_MAX_TOKENS = 16000

# ---------------------------------------------------------------------------
# 评测档案（消融矩阵）。每个 profile 声明：
#   solver        → solver.SOLVERS 注册表中的求解器名
#   n_chains      → 并行采样链数（>1 触发投票聚合）
#   solver_kwargs → 透传给该求解器的参数
#   temperature   → 可选，覆盖默认（默认多链 0.7 / 单链 None）
# 新增对比实验：注册一个 solver + 在此加一行，即可 `--profile 名` 跑。
# ---------------------------------------------------------------------------
PROFILES = {
    "b0": {"solver": "cot", "n_chains": 1, "solver_kwargs": {}},
    "b1": {"solver": "tir", "n_chains": 1, "solver_kwargs": {"max_rounds": 4}},
    "b2": {"solver": "tir", "n_chains": 8, "solver_kwargs": {"max_rounds": 4}},
    "b2-16": {"solver": "tir", "n_chains": 16, "solver_kwargs": {"max_rounds": 4}},
    # 创新点一：数值裁判反馈闭环（单链 + 内部至多 2 轮修正）
    "b3-oracle": {"solver": "oracle-tir", "n_chains": 1,
                  "solver_kwargs": {"max_revisions": 2, "tir_rounds": 4}},
}
