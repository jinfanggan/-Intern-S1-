# 官方信息核实记录（用于校准 harness 与方案）

来源：Intern-S1 GitHub / user_guide.md / HuggingFace card；OpenAI API 实测。
更新日期：2026-06。标 ⚠️ 的需拿到书生 API key 后二次确认。

## 1. Intern-S1 模型规格（GitHub 核实）

- **Intern-S1**：235B MoE 语言主干（Qwen3）+ 6B InternViT 视觉编码器
- **Intern-S1-mini**：8B 语言 + 0.3B 视觉
- **Intern-S1-Pro**：⭐ 1T 总参 / 512 专家 / 每 token 激活 8 专家（22B 激活）
  → 比赛若可选 Pro，数学能力上限更高，值得在拿到 key 后对比
- benchmark：AIME2025 **86.0**，GPQA **77.3**，MMLU-Pro 83.5，MathVista 81.5

## 2. 官方推荐采样参数（user_guide + HF card 核实）→ 已写入 config.py

```
temperature = 0.7   top_p = 1.0   top_k = 50   min_p = 0.0
```
- 我们 b2 多链采样用的 temp=0.7 **正好对齐官方推荐** ✓
- b0/b1 单链原用 temp=0（None），切到 Intern-S1 后应改用 0.7（已设 default_temperature）
- ⚠️ top_k / min_p 是否被书生云 API 暴露（OpenAI 兼容接口通常只认 top_p）待实测

## 3. thinking 模式开关（核实，但云 API 字段存歧义）

- 自部署（tokenizer / vLLM / LMDeploy）：`enable_thinking`（vLLM 嵌在 `chat_template_kwargs`）
- 书生云 API（chat.intern-ai.org.cn）：之前调研记为 `thinking_mode` 顶层字段
- ⚠️ **两者哪个对线上 API 生效必须实测**。config.py 暂留 `thinking_mode`，注释已标。
  实测方法：拿到 key 后发一个简单请求，对比带/不带字段时响应里是否含思考过程。

## 4. tool calling（核实）

- OpenAI 兼容，自部署需 `--tool-call-parser intern-s1 --reasoning-parser intern-s1`
- ⚠️ 书生云 API 是否暴露 `tools` 待实测
- **不阻塞我们**：TIR 不依赖原生 tool calling，我们自己解析 ```python 代码块

## 5. ⭐ n 参数（多候选单请求）—— OpenAI 实测确认，对 Intern-S1 限流是质变

实测 gpt-5-nano：一次请求 `n=4` 返回 4 个候选，**prompt token 只计 1 次**（31 vs 4×31）。

| 影响 | 说明 |
|---|---|
| **限流** | Intern-S1 仅 ~30 rpm。八链投票若用 n=8，从 **8 请求压成 1 请求**，限流压力 ÷8 |
| token 成本 | prompt 省（共享），completion 仍 N 份，总 token 不变 |
| ⚠️ Intern-S1 是否支持 n | OpenAI 兼容大概率支持，待实测；config 已加 `supports_n` 标志 |

## 6. ⭐ 推理模型多候选多样性低（OpenAI 实测，重要洞察）

实测 gpt-5-nano 在 **temp=1.0** 下对同一积分题采 4 候选，末位数值**高度一致**（均收敛同值）。

→ **解释了我们 b2 八链投票相比 b1 仅 +1.4pt 的根因**：推理模型采样多样性天然低，
  纯靠温度的 SC 投票增益有限。

**对策（已纳入方案演进方向）**：
- 多样性靠 **prompt 切入角扰动**（我们 solver.py 已实现 8 种角度）而非仅靠温度
- 更进一步走**创新点二「歧义分叉」**：在解读层制造差异，比路径层温度扰动更有效
- → 为"为什么需要创新点而非堆采样"提供了实证支撑
