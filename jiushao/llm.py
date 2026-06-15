"""LLM 客户端：令牌桶限流 + 指数退避重试 + 磁盘缓存 + 参数差异屏蔽。"""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from .config import MODEL_PROFILES, DEFAULT_MAX_TOKENS, RUNS_ROOT


class TokenBucket:
    """按每分钟请求数限流。"""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)


class LLM:
    """单模型客户端。cache_tag 让"同一题同一链"的重跑命中缓存，不同链彼此独立。"""

    def __init__(self, model: str, cache_dir: Path | None = None):
        if model not in MODEL_PROFILES:
            raise ValueError(f"未知模型 {model}，请先在 config.MODEL_PROFILES 注册")
        self.model = model
        self.profile = MODEL_PROFILES[model]
        api_key = os.environ.get(self.profile["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"环境变量 {self.profile['api_key_env']} 未设置（检查 .env）")
        # timeout: 防悬挂连接（nano 长推理单次可达数分钟，给足余量但必须有界）；
        # max_retries=0: SDK 内置重试与我们的指数退避重试叠加会放大最坏等待，关掉
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.profile["base_url"],
                                  timeout=300.0, max_retries=0)
        self.bucket = TokenBucket(self.profile["rpm"])
        self.cache_dir = cache_dir or (RUNS_ROOT / "cache" / model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"in": 0, "out": 0, "calls": 0, "cache_hits": 0}

    # -- 缓存 ---------------------------------------------------------------
    def _cache_key(self, messages, kwargs, cache_tag) -> Path:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "kwargs": kwargs, "tag": cache_tag},
            ensure_ascii=False, sort_keys=True,
        )
        return self.cache_dir / (hashlib.sha256(payload.encode()).hexdigest()[:32] + ".json")

    # -- 调用 ---------------------------------------------------------------
    async def chat(self, messages, *, temperature=None, max_tokens=DEFAULT_MAX_TOKENS,
                   cache_tag: str = "", retries: int = 5) -> dict:
        """返回 {content, usage_in, usage_out, cached}。"""
        kwargs: dict = dict(self.profile["extra"])
        kwargs[self.profile["max_tokens_param"]] = max_tokens
        if temperature is not None and self.profile["supports_temperature"]:
            kwargs["temperature"] = temperature

        cpath = self._cache_key(messages, kwargs, cache_tag)
        if cpath.exists():
            data = json.loads(cpath.read_text())
            self.usage["cache_hits"] += 1
            return {**data, "cached": True}

        delay = 2.0
        for attempt in range(retries):
            await self.bucket.acquire()
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs)
                content = resp.choices[0].message.content or ""
                u_in = resp.usage.prompt_tokens if resp.usage else 0
                u_out = resp.usage.completion_tokens if resp.usage else 0
                self.usage["in"] += u_in
                self.usage["out"] += u_out
                self.usage["calls"] += 1
                data = {"content": content, "usage_in": u_in, "usage_out": u_out}
                cpath.write_text(json.dumps(data, ensure_ascii=False))
                return {**data, "cached": False}
            except (RateLimitError, APIConnectionError) as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
            except APIError as e:
                # 4xx 参数错误不重试，直接抛出暴露问题
                if getattr(e, "status_code", 500) < 500 or attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def cost_usd(self) -> float:
        return (self.usage["in"] * self.profile["price_in"]
                + self.usage["out"] * self.profile["price_out"]) / 1e6
