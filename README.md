# 🎵 LLM-based Music Recommendation System / 基于大语言模型的音乐推荐系统

> **English** (below) · **中文**（见后半部分）  
> A course project: conversational music recommendation with RAG, LLM re-ranking, user profiling, collaborative filtering, and explainable recommendations (TempChain-ExRec).  
> 课程项目：融合 RAG、LLM 重排序、用户画像、协同过滤与可解释推荐（TempChain-ExRec）的对话式音乐推荐系统。

---

# English

## Overview

A music recommendation system that understands natural-language requests and returns personalized, explainable song recommendations. It combines **retrieval-augmented generation (RAG)** with **LLM re-ranking**, **user profiling**, **collaborative filtering**, and an **explainable recommendation module**.

The UI and the music database are fully in **English**; the song library contains **9,793 songs**.

## Features

- 🤖 **Conversational recommendation** — ask in natural language (scenario / mood / genre / vague).
- 📚 **RAG pipeline** — ChromaDB semantic retrieval (top-30 candidates) → LLM re-ranking → final top-k.
- 🎯 **User profiling** — preferences are extracted from the conversation and persisted (confidence ≥ 0.6) for long-term personalization.
- 👥 **Multi-user + collaborative filtering** — User-CF and Item-CF blended with content ranking (activated after ≥ 3 feedbacks).
- 🧾 **Explainable recommendations** — TempChain-ExRec generates readable reasons (6 explanation paths, chain-of-explanation + validation).
- 🔐 **Login / register / guest** — isolated profiles per user.
- 🧪 **Experiment suite** — fixed test set, 3 baselines + full model, and quantitative metrics (NDCG@5 / Precision@5 / Recall@5).

## Architecture

```
User input → LLM intent & preference extraction → vector retrieval (ChromaDB)
             └───────────── user profile ────────────────┘
                             ↓
                    LLM re-ranking + CF fusion
                             ↓
              Explainable recommendation (TempChain-ExRec)
```

| Layer | Module | Role |
|---|---|---|
| Data | `data_loader.py` | Load/manage the song database (`MusicDatabase`, `MusicItem`) |
| Retrieval | `vector_engine.py` | ChromaDB vector store + semantic search (`all-MiniLM-L6-v2` ONNX) |
| LLM | `llm_engine.py` | Preference extraction, intent detection, re-ranking, explanation |
| Profile | `user_profile.py` | Persistent user preferences + `UserManager` (multi-user) |
| CF | `collaborative_engine.py` | User-CF + Item-CF, blended with content ranking |
| Explainability | `tempchain_exrec.py` | TempChain-ExRec: template chain + chain-of-explanation + validation |
| UI | `streamlit_app.py` / `app.py` | Streamlit UI (primary) / Gradio backend |

## Quick start

### 1. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root:

```bash
DEEPSEEK_API_KEY=your_api_key_here
MUSIC_DATA_PATH=./data/music_database_en.json
VECTOR_DB_PATH=./data/vector_db_en
USER_PROFILE_DIR=./data/user_profiles
```

Get an API key at <https://platform.deepseek.com/>. The LLM is accessed via the OpenAI-compatible endpoint (`deepseek-chat`).  
> ⚠️ Keep your API key private — do not commit `.env`.

### 3. Run

```bash
# Primary UI (Streamlit, English)
streamlit run streamlit_app.py

# Alternative UI (Gradio backend)
python app.py
```

Open the URL printed in the terminal. On first launch the vector index is built from `MUSIC_DATA_PATH` (or the existing index under `VECTOR_DB_PATH` is reused).

## Usage

1. **Register / log in**, or continue as a guest.
2. Type a request, e.g.:
   - *"recommend calm instrumental music for studying"*
   - *"sad songs to heal a broken heart"*
   - *"classical piano pieces"*
3. The system extracts your preferences, retrieves candidates, re-ranks them, and returns the top songs **with explanations**.
4. 👍 / 👎 feedback on individual songs updates your profile and (after enough feedback) activates collaborative filtering.

## Project structure

```
.
├── streamlit_app.py         # Streamlit UI (primary)
├── app.py                   # Backend + Gradio UI (MusicRecommendationApp)
├── config.py                # Settings (pydantic-settings, reads .env)
├── data_loader.py           # Song database loader
├── vector_engine.py         # ChromaDB vector retrieval
├── llm_engine.py            # LLM engine (extract / rerank / explain)
├── user_profile.py          # User profiles + multi-user manager
├── collaborative_engine.py  # User-CF + Item-CF
├── tempchain_exrec.py       # TempChain-ExRec explainable recommendation
├── eval_metrics.py          # NDCG@k / Precision@k / Recall@k
├── run_experiments.py       # Experiment runner
├── data/
│   ├── music_database_en.json   # English song library (9,793 songs)
│   ├── music_database.json      # Original (Chinese) library
│   ├── test_queries.json        # Fixed evaluation query set (40 queries)
│   ├── vector_db_en/            # Vector index for the English DB
│   └── user_profiles/           # Persisted user profiles
└── experiments/             # Results, predictions, report draft, human-eval template
```

## Experiments & evaluation

The experiment suite compares **4 model variants** on a fixed test set of **40 English queries** (scenario / mood / genre / ambiguous):

| Variant | Description |
|---|---|
| `baseline1_vector` | ChromaDB semantic retrieval only (no LLM re-ranking) |
| `baseline2_llm` | Pure LLM, no retrieval (RAG removed) |
| `full` | Retrieval + LLM re-ranking + user profile (our system) |
| `ablation_noprofile` | Full model with the profile disabled |

Ground truth is rule-based (per-query genre/tag relevance rules, graded 0–3). Metrics: **NDCG@5 / Precision@5 / Recall@5**.

**Overall results (top-5, averaged over 40 queries):**

| Model | NDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Baseline-1 (vector only) | 0.590 | 0.655 | 0.005 |
| Baseline-2 (LLM only) | 0.083 | 0.105 | 0.000 |
| **Full** | **0.693** | **0.765** | **0.005** |
| Ablation (no profile) | 0.653 | 0.720 | 0.006 |

**Headline findings**

1. **Full > vector-only** on NDCG@5 (+0.103) and Precision@5 (+0.110) — LLM re-ranking fixes candidates that matched literally but not semantically.
2. **Full ≫ LLM-only** — without RAG the LLM generates songs that don't exist in the catalog (**81.5% ungrounded rate**). RAG grounds generation in the real song library.
3. **Profile adds a small, positive gain** (+0.04 NDCG), concentrated in *mood* queries; its benefit depends on whether the extracted preference matches the query's true semantics.
4. Recall@5 is structurally near zero because the rule-based relevant sets are very large (top-5 covers a tiny fraction); **NDCG@5 and Precision@5** are the primary metrics.

### Run the experiments

```bash
python run_experiments.py
```

LLM outputs are cached under `experiments/cache/` (re-runs are idempotent and don't re-spend API calls). Outputs: `experiments/results_metrics.json`, `results_table.md`, `predictions.json`, plus `report_draft.md` (report draft) and `human_eval_template.md` (qualitative human-evaluation template).

## Tech stack

- **LLM**: DeepSeek `deepseek-chat` (OpenAI-compatible API)
- **Vector DB**: ChromaDB, `all-MiniLM-L6-v2` (ONNX) embeddings
- **UI**: Streamlit (primary), Gradio (alternative)
- **Data**: pandas, numpy, pydantic / pydantic-settings
- **Logging**: loguru

## Configuration

Edit `config.py` or `.env`. Key settings: `TOP_K_RETRIEVE` (candidates, default 30), `TOP_K_RECOMMEND` (default 10), `CF_WEIGHT` (0.3), `CF_MIN_FEEDBACK` (3), and the TempChain-ExRec options (`N_EXPLANATION_PATHS`, `EXPLANATION_DETAIL_LEVEL`, validation toggle). Presets (`development` / `production` / `experiment` / `fast`) are available via `config.apply_preset(...)`.

## FAQ

- **How do I switch to another LLM?** Change `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` in `.env` — any OpenAI-compatible endpoint works.
- **The library is too small / dirty?** Replace `data/music_database_en.json` and rebuild the index (delete `data/vector_db_en/`).
- **Cross-language queries?** The current embedding model is English-only; switch to a multilingual embedding model for other languages.

## License

MIT License.

---

# 中文

## 项目简介

一个理解自然语言请求、返回个性化且**可解释**音乐推荐的系统。它融合了 **检索增强生成（RAG）**、**LLM 重排序**、**用户画像**、**协同过滤**和**可解释推荐模块**。

界面与音乐数据库已完全英文化，曲库共 **9,793 首**。

## 核心特性

- 🤖 **对话式推荐**：用自然语言描述（场景 / 情绪 / 风格 / 模糊需求）。
- 📚 **RAG 流程**：ChromaDB 语义检索（top-30 候选）→ LLM 重排序 → 最终 top-k。
- 🎯 **用户画像**：从对话中抽取偏好并持久化（置信度 ≥ 0.6），实现长期个性化。
- 👥 **多用户 + 协同过滤**：User-CF 与 Item-CF 混合（累计 ≥ 3 次反馈后启用）。
- 🧾 **可解释推荐**：TempChain-ExRec 生成可读推荐理由（6 条解释路径、解释链 + 验证）。
- 🔐 **登录 / 注册 / 游客**：用户间画像隔离。
- 🧪 **实验套件**：固定测试集、3 个基线 + 完整模型、定量指标（NDCG@5 / Precision@5 / Recall@5）。

## 系统架构

```
用户输入 → LLM 意图识别与偏好抽取 → 向量检索（ChromaDB）
             └───────────── 用户画像 ──────────────┘
                             ↓
                    LLM 重排序 + 协同过滤融合
                             ↓
                  可解释推荐（TempChain-ExRec）
```

| 层 | 模块 | 职责 |
|---|---|---|
| 数据 | `data_loader.py` | 加载 / 管理曲库（`MusicDatabase`、`MusicItem`） |
| 检索 | `vector_engine.py` | ChromaDB 向量存储 + 语义检索（`all-MiniLM-L6-v2` ONNX） |
| LLM | `llm_engine.py` | 偏好抽取、意图识别、重排序、解释生成 |
| 画像 | `user_profile.py` | 用户偏好持久化 + `UserManager` 多用户隔离 |
| 协同过滤 | `collaborative_engine.py` | User-CF + Item-CF，与内容排序加权融合 |
| 可解释性 | `tempchain_exrec.py` | TempChain-ExRec：模板链 + 解释链 + 验证 |
| 界面 | `streamlit_app.py` / `app.py` | Streamlit 界面（主）/ Gradio 后端 |

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置

在项目根目录新建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=你的_API_Key
MUSIC_DATA_PATH=./data/music_database_en.json
VECTOR_DB_PATH=./data/vector_db_en
USER_PROFILE_DIR=./data/user_profiles
```

在 <https://platform.deepseek.com/> 注册并获取 API Key，模型通过 OpenAI 兼容接口调用（`deepseek-chat`）。  
> ⚠️ API Key 请勿泄露、不要提交到 git。

### 3. 运行

```bash
# 主界面（Streamlit，英文）
streamlit run streamlit_app.py

# 备用界面（Gradio 后端）
python app.py
```

打开终端输出的地址即可。首次启动会根据 `MUSIC_DATA_PATH` 构建向量索引（或复用 `VECTOR_DB_PATH` 下已有索引）。

## 使用说明

1. **注册 / 登录**，或以游客身份进入。
2. 输入请求，例如：
   - *"recommend calm instrumental music for studying"*
   - *"sad songs to heal a broken heart"*
   - *"classical piano pieces"*
3. 系统抽取你的偏好 → 检索候选 → 重排序 → 返回 top 歌曲**并附带解释理由**。
4. 对单首歌曲 👍 / 👎 反馈会更新画像，并在反馈足够后启用协同过滤。

## 项目结构

```
.
├── streamlit_app.py         # Streamlit 界面（主）
├── app.py                   # 后端 + Gradio 界面（MusicRecommendationApp）
├── config.py                # 配置（pydantic-settings，读取 .env）
├── data_loader.py           # 曲库加载
├── vector_engine.py         # ChromaDB 向量检索
├── llm_engine.py            # LLM 引擎（抽取 / 重排序 / 解释）
├── user_profile.py          # 用户画像 + 多用户管理
├── collaborative_engine.py  # User-CF + Item-CF
├── tempchain_exrec.py       # TempChain-ExRec 可解释推荐
├── eval_metrics.py          # NDCG@k / Precision@k / Recall@k
├── run_experiments.py       # 实验运行器
├── data/
│   ├── music_database_en.json   # 英文曲库（9,793 首）
│   ├── music_database.json      # 原始（中文）曲库
│   ├── test_queries.json        # 固定评测 query 集（40 条）
│   ├── vector_db_en/            # 英文库向量索引
│   └── user_profiles/           # 持久化用户画像
└── experiments/             # 结果、预测、报告草稿、人工评估模板
```

## 实验与评测

实验套件在固定测试集（**40 条英文 query**，分场景 / 情绪 / 风格 / 模糊四类）上对比 **4 个模型变体**：

| 变体 | 说明 |
|---|---|
| `baseline1_vector` | 仅 ChromaDB 语义检索（无 LLM 重排序） |
| `baseline2_llm` | 纯 LLM、无检索（去掉 RAG） |
| `full` | 检索 + LLM 重排序 + 用户画像（本系统） |
| `ablation_noprofile` | 完整模型但关闭画像 |

Ground-truth 采用规则法（每条 query 配 genre/tag 相关性规则，0–3 分级）。指标：**NDCG@5 / Precision@5 / Recall@5**。

**总体结果（top-5，40 条 query 平均）：**

| 模型 | NDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Baseline-1（仅向量） | 0.590 | 0.655 | 0.005 |
| Baseline-2（仅 LLM） | 0.083 | 0.105 | 0.000 |
| **Full** | **0.693** | **0.765** | **0.005** |
| Ablation（去画像） | 0.653 | 0.720 | 0.006 |

**主要结论**

1. **Full 优于纯向量检索**：NDCG@5（+0.103）、Precision@5（+0.110）——LLM 重排序纠正了「字面命中但语义不符」的候选。
2. **Full 远优于纯 LLM**：无 RAG 时 LLM 会生成曲库中不存在的歌（**未接地率 81.5%**），RAG 将生成约束在真实曲库内。
3. **画像带来小幅正向增益**（+0.04 NDCG），集中在 *mood* 类；其效果取决于抽取的偏好是否贴合 query 真实语义。
4. Recall@5 结构性趋近 0（规则法相关集很大，top-5 覆盖极小部分），故以 **NDCG@5 与 Precision@5** 为主要指标。

### 运行实验

```bash
python run_experiments.py
```

LLM 输出会缓存到 `experiments/cache/`（重跑幂等、不重复计费）。产出：`experiments/results_metrics.json`、`results_table.md`、`predictions.json`，另有 `report_draft.md`（报告草稿）与 `human_eval_template.md`（人工定性评估模板）。

## 技术栈

- **LLM**：DeepSeek `deepseek-chat`（OpenAI 兼容接口）
- **向量数据库**：ChromaDB，`all-MiniLM-L6-v2`（ONNX）嵌入
- **界面**：Streamlit（主）、Gradio（备用）
- **数据处理**：pandas、numpy、pydantic / pydantic-settings
- **日志**：loguru

## 配置说明

修改 `config.py` 或 `.env`。关键参数：`TOP_K_RETRIEVE`（候选数，默认 30）、`TOP_K_RECOMMEND`（默认 10）、`CF_WEIGHT`（0.3）、`CF_MIN_FEEDBACK`（3），以及 TempChain-ExRec 相关选项（`N_EXPLANATION_PATHS`、`EXPLANATION_DETAIL_LEVEL`、验证开关）。可通过 `config.apply_preset(...)` 一键切换 `development` / `production` / `experiment` / `fast` 预设。

## 常见问题

- **如何换其他 LLM？** 修改 `.env` 里的 `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`，任意 OpenAI 兼容接口均可。
- **曲库太小 / 数据不干净？** 替换 `data/music_database_en.json` 后删除 `data/vector_db_en/` 重建索引。
- **跨语言查询？** 当前嵌入模型仅支持英文；换多语言嵌入模型即可支持其他语言。

## 许可证

MIT License。
