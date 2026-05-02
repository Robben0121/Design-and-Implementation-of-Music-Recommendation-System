# 🎵 基于LLM的音乐推荐系统

## 项目简介

这是一个基于大语言模型（LLM）和检索增强生成（RAG）技术的智能音乐推荐系统。系统通过自然语言对话理解用户的音乐偏好，结合向量检索和LLM推理，提供个性化的音乐推荐。

### 核心特点

- 🤖 **对话式交互**：通过自然语言对话收集用户偏好
- 🎯 **个性化推荐**：基于用户画像和对话历史提供精准推荐
- 📚 **RAG架构**：结合向量检索和LLM生成，保证推荐质量
- 💾 **用户画像**：持久化存储用户偏好，实现长期个性化
- 🚀 **易于扩展**：模块化设计，方便集成其他功能

## 系统架构

```
用户输入 → LLM理解意图 → 向量检索 → LLM重排序 → 推荐结果
                ↓                              ↑
            提取偏好 → 更新用户画像 ──────────┘
```

### 核心模块

1. **数据层** (`data_loader.py`)
   - 音乐数据库加载和管理
   - 支持多维度筛选和查询

2. **向量检索层** (`vector_engine.py`)
   - 基于ChromaDB的向量存储
   - 语义相似度检索

3. **用户画像层** (`user_profile.py`)
   - 用户偏好存储和管理
   - 对话历史记录

4. **LLM引擎** (`llm_engine.py`)
   - 意图理解和偏好提取
   - 推荐生成和解释

5. **应用层** (`app.py`)
   - Gradio Web界面
   - 完整交互流程

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API Key

```bash
# 复制配置文件
config.py
DEEPSEEK_API_KEY: str = "自己的API Key"

# 在 https://platform.deepseek.com/ 注册并获取
```

### 3. 准备数据

将你的音乐数据库JSON文件放到 `data/music_database.json`

数据格式要求：
```json
[
  {
    "song_id": "sid_0",
    "song": "歌曲名",
    "artist": "歌手",
    "album": "专辑",
    "genre": "流派",
    "year": 2023,
    "release_date": "2023-09-22",
    "album_type": "专辑",
    "media": "CD",
    "description": "描述",
    "tags": ["标签1", "标签2"],
    "semantic_summary": "语义摘要"
  }
]
```

### 4. 运行系统

```bash
# 启动Web界面
python app.py

# 访问 http://localhost:7860
```

## 使用指南

### 对话示例

**场景1：基础推荐**
```
用户：推荐一些适合学习的音乐
系统：[提取偏好：场景=学习]
      为您推荐以下轻柔的纯音乐...
```

**场景2：情绪推荐**
```
用户：我心情不好，想听点治愈的歌
系统：[提取偏好：情绪=治愈，心情=低落]
      为您推荐这些温暖治愈的歌曲...
```

**场景3：精确查找**
```
用户：有没有Chappell Roan的歌？
系统：[检索：artist=Chappell Roan]
      为您找到了以下歌曲...
```

**场景4：风格偏好**
```
用户：我喜欢节奏快的流行歌
系统：[更新画像：genre=流行，tempo=快]
      根据您的偏好推荐...
```

### 系统功能

1. **智能意图识别**
   - 自动判断用户是否需要推荐
   - 提取多维度偏好信息

2. **用户画像管理**
   - 自动从对话中提取偏好
   - 持久化存储用户信息
   - 支持多用户隔离

3. **推荐策略**
   - 向量相似度检索
   - LLM智能重排序
   - 考虑推荐多样性

4. **反馈机制**
   - 支持点赞/不喜欢
   - 动态调整推荐策略

## 项目结构

```
music_recommend_system/
├── app.py                 # 主应用
├── config.py              # 配置管理
├── data_loader.py         # 数据加载
├── vector_engine.py       # 向量检索
├── user_profile.py        # 用户画像
├── llm_engine.py          # LLM引擎
├── requirements.txt       # 依赖列表
├── .env.example          # 配置示例
├── data/                 # 数据目录
│   ├── music_database.json
│   ├── chromadb/         # 向量数据库
│   └── user_profiles/    # 用户画像
└── logs/                 # 日志目录
```

## 技术栈

- **LLM**: DeepSeek API（兼容OpenAI格式）
- **向量数据库**: ChromaDB
- **Web框架**: Gradio
- **数据处理**: Pandas, Pydantic
- **日志**: Loguru

## 高级功能

### 1. 扩展音乐知识库

```python
# 添加更多音乐数据
from data_loader import MusicDatabase

db = MusicDatabase("data/music_database.json")
# 添加新歌曲...
```

### 2. 自定义推荐策略

编辑 `llm_engine.py` 中的 `_llm_rerank_and_explain` 方法

### 3. 集成音乐播放

可以集成Spotify API或其他音乐平台：

```python
# 在推荐结果中添加播放链接
def get_spotify_link(song_name, artist):
    # 调用Spotify API
    pass
```

### 4. 评估系统效果

```python
# 收集用户反馈
def evaluate_recommendation(recommended_songs, user_feedback):
    # 计算准确率、多样性等指标
    pass
```

  
## 常见问题

### Q: 如何使用其他LLM？

修改 `config.py` 中的配置，支持任何兼容OpenAI API格式的模型。

### Q: 数据库很大怎么办？

- 使用更强大的向量数据库（如Milvus）
- 分批构建索引
- 添加缓存机制

### Q: 如何部署到生产环境？

```bash
# 使用Docker
docker build -t music-recommend .
docker run -p 7860:7860 music-recommend

# 或使用云服务
# 部署到HuggingFace Spaces, Render等
```

## 参考资料

- [LangChain文档](https://python.langchain.com/)
- [ChromaDB文档](https://docs.trychroma.com/)
- [DeepSeek API文档](https://platform.deepseek.com/docs)
- [Gradio文档](https://www.gradio.app/docs/)

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提Issue或PR。

---

