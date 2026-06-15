# 协作规范（九韶 / Intern-S1 数学智能体）

仓库：`git@github.com:jinfanggan/-Intern-S1-.git`

多人协作 + 多实验分支，请所有人遵守本规范。**核心三条**：
1. **绝不提交密钥与大文件**（`.env` / `runs/` / `datasets/` 已在 `.gitignore`，别强行 `git add -f`）
2. **不直接推 `main`**，一律走分支 + Pull Request
3. **提交前 `pytest` 必须全绿**

---

## 一、首次配置（每人每机一次）

### 1. 生成并登记 SSH key

```bash
ssh-keygen -t ed25519 -C "你的邮箱" -f ~/.ssh/github_ed25519 -N ""
cat ~/.ssh/github_ed25519.pub        # 复制整行
```
到 GitHub → Settings → SSH and GPG keys → New SSH key，粘贴公钥。

### 2. 配置 ssh（非默认文件名需指定）

把以下写入 `~/.ssh/config`：
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_ed25519
  IdentitiesOnly yes
```
验证：`ssh -T git@github.com` → 看到 `Hi <用户名>!` 即成功。

### 3. 配置 git 身份

```bash
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
```

### 4. 克隆与环境

```bash
git clone git@github.com:jinfanggan/-Intern-S1-.git
cd -Intern-S1-
cp .env.example .env            # 填入自己的 API key
pip install -r requirements.txt
python -m pytest -q             # 确认环境 OK（应全绿）
```

> 数据集（`datasets/`）和第三方参考代码（`reference/`）不在仓库里，需各自获取，
> 路径见 `README.md`。

---

## 二、分支模型

| 分支 | 用途 | 规则 |
|---|---|---|
| `main` | 稳定可运行版本 | **受保护**，只能通过 PR 合入；任何时刻 `pytest` 全绿 |
| `feat/<功能>` | 新功能 | 如 `feat/router-agent`、`feat/genselect` |
| `fix/<问题>` | 修 bug | 如 `fix/judge-latex-interval` |
| `exp/<实验名>` | 实验/消融（可能不合并）| 如 `exp/oracle-bestof-n`、`exp/interns1-baseline` |
| `report/<版本>` | 技术报告/方案文档 | 如 `report/v1` |

**命名一律小写 + 连字符**。实验分支即使最后不并入 `main` 也要推上去（留痕 + 队友可复现）。

---

## 三、日常工作流

```bash
# 1. 永远从最新 main 开分支
git checkout main && git pull
git checkout -b feat/my-feature

# 2. 改代码 → 跑测试（必须绿）
python -m pytest -q

# 3. 提交（信息见下节规范）
git add <具体文件>            # 避免 git add -A 误加产物
git commit -m "feat(solver): 增加 GenSelect 仲裁求解器"

# 4. 推送并开 PR
git push -u origin feat/my-feature
#   到 GitHub 点 Compare & pull request，指派 reviewer
```

合入后删分支：`git branch -d feat/my-feature && git push origin --delete feat/my-feature`

---

## 四、提交信息规范（Conventional Commits）

格式：`<类型>(<范围>): <简述>`

| 类型 | 用于 |
|---|---|
| `feat` | 新功能 / 新求解器 / 新实验脚本 |
| `fix` | 修 bug |
| `exp` | 实验配置或结果 |
| `docs` | 文档 |
| `test` | 测试 |
| `refactor` | 重构（不改行为） |
| `chore` | 杂项（依赖、配置） |

范围举例：`solver` / `judge` / `oracle` / `pipeline` / `router` / `dsets`。

示例：
```
feat(router): 题型路由智能体，按 answer_type 分流求解通道
fix(judge): 修复 LaTeX 区间答案 vs 小数 gold 的等价判定
exp(oracle): HARDMath 积分题 best-of-N 裁判过滤，救回率 3/7
```

---

## 五、绝不提交清单（重要）

| 内容 | 原因 | 已在 .gitignore |
|---|---|---|
| `.env` / 任何 API key、SSH 私钥 | **泄露即被盗刷/盗号** | ✅ |
| `runs/` | 运行产物+缓存，体积大可复现 | ✅ |
| `datasets/` | 第三方数据，各自获取 | ✅ |
| `reference/` | clone 的第三方源码（lagent 等） | ✅ |
| `__pycache__/` `.pytest_cache/` | Python 缓存 | ✅ |

提交前自查：`git status` 看清楚加了什么；可疑就 `git diff --cached`。

**万一密钥已被提交**：立即到对应平台吊销该 key（OpenAI/GitHub）重新生成——
git 历史里删除很麻烦且不彻底，**换 key 才是根本**。

---

## 六、代码质量门槛

- 新增/修改逻辑**必须配单元测试**（本项目惯例：每个组件一个 `test_*.py`，
  用 `tests/conftest.py` 的 `FakeLLM` 离线测试，不打网络）
- 新增对比实验：注册一个 solver + `config.PROFILES` 加一行 + 写测试（见 `README.md`）
- 判分器每修一个误判，在 `tests/test_judge_regression.py` 固化一条回归用例
- PR 合入前确保 `python -m pytest -q` 全绿

---

## 七、实验复现约定

- 实验结果（正确率、消融数）写进 PR 描述或 `report/` 文档，**不要把 `runs/` 推上去**
- 跑实验用 `--run-name` 显式命名（如 `abl-tqa200-b1`），便于队友按名复现
- 关键结论附复现命令，例：
  ```bash
  python scripts/run_eval.py --dataset theoremqa --profile b1 --limit 200
  ```
