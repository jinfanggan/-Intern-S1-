# 九韶（JiuShao）· 基于 Intern-S 的数学推理智能体

> 挑战杯·书生赛道 **XH-202627「基于 Intern-S 系列大模型的数学智能体设计与推理创新」** 参赛系统。

代号取秦九韶《数书九章》（中国古代算法化巅峰）+「箫韶九成」（多声部协奏 = 多智能体协作）。

---

## 这是什么

赛题底模固定为 Intern-S 系列 API（**不训练、不微调**），分差只来自智能体工程。本项目在「多路采样 + 工具集成推理」的实战配方之上，叠加**两个原创机制**攻坚剩余失败模式：

| | 机制 | 解决的失败模式 | 状态 |
|---|---|---|---|
| 主干 | **SC-TIR**：多链「推理 + 真实代码执行」+ 等价归一投票 | 机械计算出错 | ✅ |
| 创新点一 | **数值裁判（Evaluator-First）**：先让智能体写代码数值求解原问题作「可执行裁判」，再对着裁判过滤候选 | "算错了但不自知"——LLM 自验不可靠 | ✅ |
| 创新点二 | **歧义分叉（Interpretation Forking）**：枚举题意解读→分叉求解→仲裁，把自洽性从路径层提到解读层 | "审题错了"——所有链共享同一误读，采样救不回 | ✅ |

> 设计理念：**别人优化"怎么解题"，我们优化"怎么知道解对了、怎么知道读对了"。**

完整方案见 [`docs/proposal/方案_九韶.pdf`](docs/proposal/方案_九韶.pdf)（LaTeX 源同目录）；设计与实验细节见 [`docs/项目维护/`](docs/项目维护/)。

---

## 快速开始

```bash
cp .env.example .env          # 填 OPENAI_API_KEY（开发）/ INTERN_API_KEY（比赛）
pip install -r requirements.txt
python -m pytest -q           # 120 用例，FakeLLM 离线不打网络，应全绿

# 在官方样例题上跑（最接近正式评测分布）
python scripts/run_eval.py --dataset official --profile b2
# baseline(纯投票) vs +数值裁判 对比
python scripts/compare_oracle.py official 3
```

> 数据集与第三方代码不在仓库里，获取方式见 [数据与资源清单](docs/项目维护/数据与资源清单.md)。

---

## 提交到官方评测

正式评测只调 `user_agent.py` 的契约接口，本项目已适配：

```python
# submission/user_agent.py
agent = ReasoningAgent(client=official_client)   # client 平台给
result = agent.solve(problem, metadata)          # → {"final_response": str, "trace": [...]}
```

提交时 fork 官方 baseline（`github.com/InternLM/Challenge-Cup-2026`），用 `submission/user_agent.py` + `jiushao/` 包替换其入口。适配层 `jiushao/adapter.py` 把官方同步 client 包成引擎所需接口，完整复用求解逻辑。

---

## 架构（四层流水线）

```
题目 → ① 求解：N 链 TIR（推理 + 真实代码执行）
            ↓ N 个候选
       ② 数值裁判过滤（构造可执行裁判 → 剔除被数值否决的候选，带适用性自检）
            ↓ 幸存候选
       ③ 等价归一投票 → 选最大簇
       ④ → final_response + trace（全程事件流落盘）
   （歧义题走分叉通道：枚举解读 → 各解读分叉求解 → 仲裁）
```

```
jiushao/
├── config.py      模型档案（参数差异屏蔽/价格/限流）+ 消融 profile
├── llm.py         API 客户端：令牌桶限流 + 磁盘缓存 + 退避重试 + 超时
├── sandbox.py     代码执行：子进程隔离 + 超时/内存上限 + 黑名单
├── judge.py       六级等价判定 + HARDMath regime 数值判分（开发期自测）
├── dsets.py       数据集加载（official/theoremqa/hardmath/arb/math500）
├── solver.py      CoT/TIR 求解链 + 求解器注册表
├── aggregate.py   等价归簇 + 多数投票
├── oracle.py      【创新点一】数值裁判构造 + 候选打分
├── oracle_loop.py 【创新点一】裁判反馈闭环 + 防博弈（留出点/溯源/账本）
├── forking.py     【创新点二】枚举解读 + 分叉求解 + 仲裁
├── adapter.py     官方契约适配 + 两条创新点入口
├── pipeline.py    单题编排（开发期评测）
├── runner.py      批量评测：并发 + 断点续跑 + 分领域报告
└── runlog.py      结构化事件流日志
```

## 消融 profile（一键对比）

| profile | 求解 | 链数 | 说明 |
|---|---|---|---|
| `b0` | CoT | 1 | 裸基线摸底 |
| `b1` | TIR | 1 | +工具集成推理 |
| `b2` / `b2-16` | TIR | 8 / 16 | +多链投票 |
| `b3-oracle` | TIR+裁判 | 1 | +数值裁判反馈闭环 |

> 新增对比实验三步：写 solver → `register_solver` → `config.PROFILES` 加一行。详见 [CONTRIBUTING](CONTRIBUTING.md)。

## 关键实验结论（gpt-5-nano 代理模型）

| 实验 | 结果 |
|---|---|
| TheoremQA ×200 单链 TIR | 80.0%（超文献 GPT-4 PoT 的 51% 口径） |
| 八链投票 vs 单链 | 净 +0 —— 推理模型多候选多样性低，**实证"需创新机制而非堆采样"** |
| 数值裁判构造准确率 | HARDMath 积分 7/7，与真值误差 ≤2.6% |
| 数值裁判过滤 | 正确式 score 1.0 / 错误式 0.0，符号题自动退回投票（零副作用） |

完整数据见 [开发进展记录](docs/项目维护/开发进展记录.md)。

## 运行日志（前端对接约定）

每次运行产出 `runs/<run_name>/`：`meta.json`（配置）+ `results.jsonl`（每题摘要，断点续跑依据）+ `events/<题id>.jsonl`（完整事件流）。

事件 = `{ts, stage, type, **data}`，前端按 ts 回放即重建推理树（链=分支、沙箱=工具节点、vote=汇聚、oracle/fork=创新机制节点）。回放：`python scripts/view_log.py <run_name> <题id>`。

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `run_eval.py` | 批量评测 |
| `compare_oracle.py` | baseline vs +数值裁判 对比 |
| `triage.py` | 失败分诊（管线/抽取/判分误判/真错） |
| `rejudge.py` | 判分器升级后离线重判（零 API 成本） |
| `view_log.py` | 终端事件流回放 |

## 切换到 Intern-S 模型

`.env` 填 `INTERN_API_KEY` 后 `--model intern-s2-preview`（官方推荐，数学更强）即可，代码零改动（`config.py` 已配 base_url / 限流 / thinking）。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/项目维护/项目结构文档_v0.1.md](docs/项目维护/项目结构文档_v0.1.md) | 架构 / 模块 / 关键函数 / 流程 / 运维 |
| [docs/项目维护/数据与资源清单.md](docs/项目维护/数据与资源清单.md) | 数据在哪 / 怎么获取 / 缓存结构 |
| [docs/项目维护/开发进展记录.md](docs/项目维护/开发进展记录.md) | 各轮进展 + 实验数据 + 待办 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作规范（分支 / 提交 / 密钥纪律） |
| [OFFICIAL_FINDINGS.md](OFFICIAL_FINDINGS.md) | 官方文档核实记录 |

## 待办

- [ ] 仲裁可靠性提升（歧义分叉瓶颈）
- [ ] HARDMath 数值题大样本：数值裁判真实增益数字
- [ ] 路由智能体 + 动态预算（按题型分流 oracle / forking / 投票）
- [ ] 切到 Intern-S2 实测（key 到位后）
- [ ] 决赛 Web Demo（推理树可视化）

## 版本

v0.1（研发中）。升号需负责人授权，见 [CONTRIBUTING](CONTRIBUTING.md) 版本纪律。
