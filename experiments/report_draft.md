# 实验报告配套内容（草稿）

> 本文件是课程报告 **Experiments / Analysis / Related Work / Approach / Conclusion / 组员贡献** 的配套草稿。`Results` 表格与 `Analysis` 案例已用 `run_experiments.py` 的实际结果回填（数据见 `experiments/results_table.md`、`results_metrics.json`、`predictions.json`）。

---

## 1. Related Work（相关工作）

- **RAG（Retrieval-Augmented Generation）**：Lewis et al. 提出用外部知识库增强 LLM 生成，缓解 LLM 的幻觉与知识截止问题。本系统用 ChromaDB 向量库存储歌曲语料，检索后再交给 LLM 重排序，即典型 RAG 范式。
- **对话式推荐（Conversational Recommendation）**：通过自然语言交互理解用户偏好，动态更新画像，再给出可解释的推荐。本系统以 LLM 抽取用户偏好、维护多轮画像。
- **向量检索音乐推荐**：用语义向量（`all-MiniLM-L6-v2`）把歌曲的文本描述（歌名/歌手/流派/标签/摘要）与用户查询映射到同一向量空间，计算相似度召回候选。
- **可解释推荐（Explainable Recommendation）**：TempChain-ExRec 用模板链 + 解释链（Chain-of-Explanation）+ 验证，为每首推荐生成可读理由。

## 2. Approach（系统架构，分模块）

1. **数据层** `data_loader.py`：`MusicDatabase` 加载 JSON 歌曲库（9793 首），`MusicItem` 组合各字段生成嵌入文本。
2. **检索引擎** `vector_engine.py`：ChromaDB 持久化向量库，`search()` 做语义检索，返回 top-k 候选。
3. **LLM 推荐引擎** `llm_engine.py`：`extract_user_preferences()` 抽取偏好、`recommend_songs()` 做「检索 + LLM 重排序」、`chat_with_context()` 自由对话、`is_recommendation_request()` 意图识别。
4. **用户画像** `user_profile.py`：`UserProfile` 持久化偏好/喜欢/不喜欢，`UserManager` 多用户隔离（注册/登录/游客）。
5. **协同过滤** `collaborative_engine.py`：User-CF + Item-CF 混合，与内容推荐加权融合。
6. **可解释性** `tempchain_exrec.py`：6 条解释路径 + 多路径 CoE 生成 + 验证。
7. **前端** `streamlit_app.py`：登录/注册/聊天推荐/反馈/画像/统计。

（架构图可直接复用 README 里的流程图，贴到此处。）

## 3. Experiments —— Model settings（模型配置）

| 项目 | 取值 |
|---|---|
| LLM | DeepSeek `deepseek-chat`（OpenAI 兼容，`https://api.deepseek.com/v1`） |
| 偏好抽取 temperature | 0.3 |
| LLM 重排序 temperature | 0.7 |
| 意图识别 temperature | 0.0 |
| 嵌入模型 | ChromaDB 默认 `all-MiniLM-L6-v2`（ONNX） |
| 检索候选数 `TOP_K_RETRIEVE` | 30 |
| 最终 top-k（本实验） | 5 |
| 协同过滤权重 `CF_WEIGHT` | 0.3（实验未启用 CF，因需 ≥3 次反馈） |
| 用户画像 | Full 启用（LLM 抽取偏好，置信度 ≥0.6 才持久化）；Ablation 关闭 |
| 数据库 | `music_database_en.json`（英文版，9793 首） |

## 4. Experiments —— Data（数据集）

- 歌曲库：公开音乐数据整理为 `{song_id, song, artist, album, genre, year, tags, semantic_summary, description}`，共 9793 首（英文版）。
- 测试集：固定 40 条英文自然语言 query，分 4 类（场景 / 情绪 / 风格流派 / 模糊歧义），见 `data/test_queries.json`。
- 任务定义：输入 = 用户自然语言查询；输出 = Top-5 推荐歌曲列表。
- Ground-truth：规则法（每条 query 配 genre/tag 相关性规则，`eval_metrics.py` 判定 0–3 级相关）。

## 5. Experiments —— Results

### 5.1 总体指标（40 条 query 平均，top-k=5）

| 模型 | NDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Baseline-1 纯向量检索 | 0.5903 | 0.6550 | 0.0048 |
| Baseline-2 纯 LLM（无 RAG） | 0.0830 | 0.1050 | 0.0002 |
| **Full（检索 + LLM 重排序 + 画像）** | **0.6932** | **0.7650** | **0.0048** |
| Ablation（去掉画像） | 0.6527 | 0.7200 | 0.0056 |

### 5.2 分类型 Precision@5 / NDCG@5

| 模型 | scene | mood | genre | ambiguous |
|---|---|---|---|---|
| Baseline-1 纯向量检索 | 0.72 / 0.591 | 0.60 / 0.485 | 0.88 / 0.878 | 0.42 / 0.407 |
| Baseline-2 纯 LLM | 0.08 / 0.058 | 0.14 / 0.107 | 0.06 / 0.057 | 0.14 / 0.110 |
| **Full** | 0.78 / 0.621 | **0.92 / 0.811** | 0.94 / 0.928 | 0.42 / 0.413 |
| Ablation（去掉画像） | 0.82 / 0.650 | 0.68 / 0.573 | 0.94 / 0.943 | 0.44 / 0.446 |

完整分项见 `experiments/results_table.md` / `results_metrics.json`。

### 5.3 三组对比结论

1. **Full 优于 Baseline-1（纯向量检索）**：NDCG@5 +0.103（相对 +17.4%），P@5 +0.110。说明在向量召回的候选上再做 LLM 重排序，能显著纠正「字面关键词命中但语义不符」的候选（典型如 `dreamy atmospheric music` 被向量检索成摇滚乐队 Dream Theater，见 §6.1 样例 B）。
2. **Full 远优于 Baseline-2（纯 LLM）**：NDCG@5 +0.610（8.4×），P@5 +0.660。纯 LLM 无检索时，5 首里平均 4 首以上是曲库中不存在、凭空生成的歌（**幻觉率 81.5%，163/200**），只有个别知名曲目（如 Prince《Purple Rain》）能命中。RAG 把生成约束到真实曲库内，是相关性提升的根本原因。
3. **画像（Ablation）带来小幅正向增益**：Full 相比 Ablation，NDCG@5 +0.040、P@5 +0.045。但增益集中在 **mood 类**（P@5 0.92 vs 0.68，+0.24），scene/ambiguous 类几乎持平甚至略降（scene 0.78 vs 0.82）——画像是否受益，取决于 LLM 抽取的偏好是否与 query 真实语义一致（见 §5.2 分类型表 mood 行 / §6.2 失败类型 1）。

### 5.4 关于 Recall@5 的说明

Recall@5 整体趋近于 0（~0.005），原因是规则法下相关集很大（每条 query 命中几十到上千首歌），top-5 只能覆盖其中极小一部分，属结构性必然，而非系统缺陷。因此本实验以 **NDCG@5 与 Precision@5** 为主要指标，Recall@5 仅作参考。

## 6. Analysis（定性案例）

### 6.1 ✅ 成功样例

**样例 A：明确风格 query，Full 拿到满分。**
- query：`classical piano pieces`（genre_02）
- Full 输出：`Ultimate Classical Piano`、`Easy-Listening Piano Classics: J.S.Bach`、`Best 100 Piano Classics`、`Piano Book`(Lang Lang)、`The piano's 101` —— 全部为 Classical + Piano，rel 全 3，P@5=1.0、NDCG@5=1.0。
- 原因：query 语义清晰，向量召回即命中古典钢琴合集，LLM 重排序在同类候选中再筛出更贴合条目，画像未产生干扰。

**样例 B：LLM 重排序救回「字面命中但语义不符」的召回。**
- query：`dreamy atmospheric music`（mood_08）
- Baseline-1 输出：`Octavarium`(Dream Theater)、`Dreamspace`(Stratovarius)、`Moonrise Kingdom` —— 全是字面含 "dream/moon" 的摇滚/原声，P@5 仅 0.2。
- Full 输出：`Five Songs`(Endless Melancholy, Electronic)、`Moon Safari`(Air, Electronic)、`Dreaming.`(Easy Listening)、`vol.3/Dreaming`(Easy Listening)、S.E.N.S.(Easy Listening) —— 全部符合「梦幻氛围」语义，P@5=1.0、NDCG@5=0.79。
- 原因：向量检索被 "dream" 字面误导成摇滚乐队，LLM 重新理解「dreamy atmospheric = 氛围/电子/轻音乐」后重排序修正了结果。这正是 RAG「检索 + 重排序」优于纯检索的关键证据。

### 6.2 ❌ 失败样例

**失败类型 1：画像偏好与真实语义不一致，干扰排序。**
- query：`songs for a rainy day`（scene_10），ground-truth 相关集 = {Easy Listening, Jazz, Folk}。
- Ablation（无画像）输出：`Sentiment in rain`(Folk)、`Rain Stories`(Easy Listening)、… → 两个相关排前，P@5=0.4、NDCG@5=0.55。
- Full（有画像）输出：`Singin' in the Rain`(Soundtrack)、`Rain Stories`(Easy Listening)、`Ref:rain`(Pop)、`Kisses in the Rain`(Pop)、… → 只剩 1 个相关，P@5=0.2、NDCG@5=0.21。
- 根源：LLM 从 query 抽取的画像偏好（Pop/Soundtrack 方向的「雨天」联想）与规则定义的 ground-truth 流派不一致，注入增强查询后把不相关候选顶到前面。属**画像累积错误偏好**的典型案例。

**失败类型 2：模糊/歧义 query 全线失败。**
- query：`something different to listen to`（vague_02），ground-truth = {World Music, Electronic, Jazz}。
- 四个模型 P@5 全部为 0。向量检索被字面词 "listen" 误导（`Are You Listening?`、`Hear You` 等摇滚），LLM 重排序也无从纠正，因为 query 本身不携带可检索的流派/情绪信号。
- 额外暴露**流派词表歧义**：系统据此召回了一些「Easy Listening」流派歌曲，但规则认定相关的是 World Music/Electronic/Jazz，故计 0 —— 「Easy Listening」流派名与字面词 "listen" 撞车。

**失败类型 3：纯 LLM 幻觉（Baseline-2）。**
- query：`classical piano pieces`（genre_02），纯 LLM 无检索，5/5 全是曲库中不存在的曲目（如真实的《月光奏鸣曲》《致爱丽丝》等，但这些不在本 9793 首曲库内）。
- query：`songs for sleep and bedtime`（scene_03），5/5 全部幻觉。
- 全库统计：40 条 query × 5 首 = 200 个槽位，其中 163 个无法匹配到数据库（**81.5% 幻觉率**）。说明不接 RAG 的 LLM 会「一本正经地推荐不存在的歌」。

### 6.3 失败根源归类

| 根源 | 代表 query | 表现 |
|---|---|---|
| 检索问题（字面 vs 语义） | mood_08、vague_02 | 向量按字面词召回错误候选 |
| LLM 理解问题（歧义/模糊） | vague_02、vague_06 | query 无明确信号，重排序无法纠正 |
| 画像干扰（偏好错位） | scene_10 | 画像注入的错误偏好顶掉正确候选 |
| 知识库/词表问题 | vague_02、genre 词表 | 「Easy Listening」流派名与字面词歧义、曲库覆盖有限 |

## 7. Conclusion（结论）

- 总结：RAG + LLM 重排序 + 用户画像的完整系统在相关性上优于「纯向量检索」和「纯 LLM」两个基线，并具备可解释性。
- 局限性：
  1. 依赖外部 LLM API（成本、延迟、离线不可用）；
  2. 知识库覆盖有限（仅 9793 首，且英文摘要为机翻、质量一般）；
  3. 模糊/歧义 query 处理差；
  4. 嵌入模型为英文模型，跨语言检索受限。
- 未来改进：换多语言嵌入模型、扩大并清洗知识库、引入更细粒度相关性标注、加入消融更彻底的画像更新策略。

## 8. 组员贡献（占位，需 3 人填写）

| 成员 | 贡献（1–2 句） | 百分比 |
|---|---|---|
| 成员 A | （占位） | % |
| 成员 B | （占位） | % |
| 成员 C | （占位） | % |
