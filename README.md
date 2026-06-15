# 九韶（JiuShao）— 数学智能体评测 Harness

挑战杯 XH-202627「基于 Intern-S1 的数学智能体设计与推理创新」工程底座。
技术方案见 `../xh202627_proposal/proposal.pdf`。

## 快速开始

```bash
# 1. 密钥放 .env（OPENAI_API_KEY 开发用 / INTERN_API_KEY 比赛用）
# 2. 单元测试（不打网络）
python3 -m pytest tests/ -q
# 3. 评测
python3 scripts/run_eval.py --dataset math500 --model gpt-5-nano --profile b0 --limit 10
python3 scripts/run_eval.py --dataset theoremqa --model gpt-5-mini --profile b2 --limit 50
```

## 架构（对应方案 §3 四层流水线）

```
jiushao/
├── config.py      模型档案（参数差异屏蔽/价格/限流）+ 消融 profile（b0/b1/b2）
├── llm.py         LLM 客户端：令牌桶限流 + 指数退避 + 磁盘缓存（断点续跑基础）
├── sandbox.py     Python 沙箱：子进程隔离 + 超时/内存上限 + 黑名单
├── judge.py       判分：boxed 抽取 → math-verify → SymPy 跨格式 → 数值容差 → 串归一
├── dsets.py       数据集加载（math500/theoremqa/hardmath/arb → 统一 Problem）
├── solver.py      求解链：CoT（B0）/ TIR 工具循环（B1+），多链切入角扰动
├── aggregate.py   等价归簇 + 多数投票（B2+）
├── pipeline.py    单题编排：N 链并行 → 聚合 → 判分
├── runner.py      批量评测：并发 + 断点续跑 + 分领域报告 + 成本统计
└── runlog.py      结构化事件日志（debug + 前端推理树回放 + 赛规留痕）
```

## 运行日志（前端对接约定）

每次运行产出 `runs/<run_name>/`：

- `meta.json` — 运行配置
- `results.jsonl` — 每题一行摘要（id/pred/gold/correct/elapsed），断点续跑依据
- `events/<题id>.jsonl` — **每题完整事件流**，事件 = `{ts, stage, type, **data}`

| stage | type | 含义 / 关键字段 |
|---|---|---|
| input | problem | 题面、gold、profile |
| solve | chain_round | 某链某轮模型输出（content/usage/cached）|
| sandbox | exec | 代码执行（code/ok/stdout/stderr）|
| solve | chain_done / chain_error | 单链结束（answer/rounds）/ 异常 |
| aggregate | vote / single | 聚簇结果（clusters=[(答案,票数)...]）|
| verdict | judge | pred/gold/correct/elapsed |

前端按 ts 顺序回放事件流即可重建推理树（链=分支、沙箱=工具节点、vote=汇聚）。

## 消融 profile

| profile | 求解 | 链数 N | TIR 轮数 M | 对应方案 |
|---|---|---|---|---|
| b0 | CoT | 1 | — | 基线摸底 |
| b1 | TIR | 1 | 4 | +工具 |
| b2 / b2-16 | TIR | 8 / 16 | 4 | +SC 投票 |

## 切换到 Intern-S1

`.env` 填 `INTERN_API_KEY` 后 `--model intern-s1` 即可（config.py 已配好
base_url / thinking_mode / 限流 28 rpm），代码零改动。

## 待办（按方案 §6 节奏）

- [ ] 路由智能体 + 动态预算（方案 §4.1）
- [ ] 数值代回校验 + GenSelect 仲裁（§4.3）
- [ ] 进化求解器（OpenEvolve，§4.4）
- [ ] HARDMath 按 `precision` 字段定制判分；ARB 符号题 LLM-judge
- [ ] 官方 112 题 loader + 官方 JSON schema 输出（§4.5）
