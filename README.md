# AI Research Agent（阶段四）

基于 LangChain 和 ECNU API 的状态驱动 Agent 系统，支持原版 Agent 和 LangGraph 两种实现。

## 项目特点

- ✅ 模块化设计，职责清晰
- ✅ 状态驱动架构（AgentState）
- ✅ 支持工具调用（Tool Calling）
- ✅ 集成 RAG 系统（检索增强生成）
- ✅ 向量数据库（Chroma + HuggingFace Embeddings）
- ✅ 两种 Agent 实现（原版 + LangGraph）
- ✅ 标准 ReAct 模式（多轮推理）
- ✅ 易于扩展和维护

## 技术栈

- **Python**: 3.9+（推荐 3.10 或 3.11）
- **LLM 框架**: LangChain, LangGraph
- **向量数据库**: Chroma
- **Embedding**: HuggingFace sentence-transformers
- **API**: ECNU API（OpenAI 兼容）

## 项目结构

```
AI Research Agent/
├── src/                       # 源代码目录
│   ├── __init__.py           # 包初始化
│   ├── config.py             # 配置管理
│   ├── llm.py                # LLM 初始化
│   ├── state.py              # 状态定义
│   ├── tools.py              # 工具定义
│   ├── agent.py              # Agent 核心逻辑（原版）
│   ├── langgraph_agent.py    # LangGraph Agent 实现
│   ├── embeddings.py         # Embedding 模型
│   ├── vector_store.py       # 向量数据库
│   ├── knowledge_base.py     # 知识库数据
│   └── rag.py                # RAG 系统
├── docs/                      # 文档目录
│   ├── RAG_IMPLEMENTATION.md           # RAG 实现详解
│   └── LANGGRAPH_IMPLEMENTATION.md     # LangGraph 实现详解
├── data/                      # 数据目录
│   └── chroma_db/            # 向量数据库持久化
├── main.py                    # 原版 Agent 演示
├── main_langgraph.py          # LangGraph Agent 演示
├── view_knowledge_base.py     # 查看知识库工具
├── ALL_CODE.py                # 完整代码合集（方便分享给 AI）
├── ALL_CODE_README.md         # 代码合集使用说明
├── QUICKSTART.md              # 快速上手指南
├── requirements.txt           # 依赖列表
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

### 核心模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | 管理环境变量和配置项 |
| `state.py` | 定义 Agent 的状态结构（AgentState） |
| `llm.py` | 初始化和配置语言模型 |
| `tools.py` | 定义 Agent 可用的工具（calculator, knowledge_search） |
| `agent.py` | 原版 Agent 核心逻辑（支持多轮推理） |
| `langgraph_agent.py` | LangGraph Agent 实现（标准 ReAct 模式） |
| `embeddings.py` | 创建 Embedding 模型（HuggingFace） |
| `vector_store.py` | 管理向量数据库（Chroma） |
| `knowledge_base.py` | 定义知识库数据（8个文档） |
| `rag.py` | RAG 系统实现 |

### 演示程序

| 文件 | 说明 |
|------|------|
| `main.py` | 原版 Agent 演示（while 循环实现） |
| `main_langgraph.py` | LangGraph Agent 演示（图结构实现） |
| `view_knowledge_base.py` | 查看知识库内容 |

## 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境（如果使用）
conda activate AIResearch

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 ECNU API Key：

```
ECNU_API_KEY=你的实际API密钥
```

### 3. 运行程序

**原版 Agent**（while 循环实现）：
```bash
python main.py
```

**LangGraph Agent**（图结构实现）：
```bash
python main_langgraph.py
```

**查看知识库**：
```bash
python view_knowledge_base.py
```

> 💡 更多详细说明请查看 [QUICKSTART.md](QUICKSTART.md)

## 核心功能

### 1. 两种 Agent 实现

| 特性 | 原版 Agent | LangGraph Agent |
|------|-----------|----------------|
| 实现方式 | while 循环 | 图结构（StateGraph） |
| 复杂度 | 简单 | 中等 |
| 可视化 | 无 | 支持 |
| 扩展性 | 中等 | 高 |
| 多轮推理 | ✅ 支持 | ✅ 支持 |
| 功能 | 完全相同 | 完全相同 |

### 2. 工具系统

Agent 可以自动判断何时需要使用工具：

- **calculator**: 数学计算工具（加法）
- **knowledge_search**: 知识库检索工具（RAG）

### 3. RAG 系统

检索增强生成（Retrieval-Augmented Generation）：

1. 用户提问
2. 从向量数据库检索相关文档（语义搜索）
3. 将检索结果作为上下文
4. LLM 基于上下文生成答案

**知识库内容**（8个文档）：
- 华东师范大学相关信息（5个文档）
- 技术知识（Python、LangChain、RAG）

### 4. 向量数据库

使用 Chroma + HuggingFace Embeddings：

- 自动持久化到 `./data/chroma_db`
- 支持语义搜索（不只是关键词匹配）
- 首次运行自动构建索引
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 模型

## 核心概念

### 1. 状态驱动架构

所有函数都遵循 `(state, *args) -> state` 的签名：

```python
# 状态定义
class AgentState(TypedDict):
    messages: List[Any]                  # 消息历史
    tool_calls: Optional[List[dict]]     # 待执行的工具调用
    final_answer: Optional[str]          # 最终答案
    error: Optional[str]                 # 错误信息

# 状态转换函数
def call_model(state: AgentState, llm) -> AgentState:
    """调用模型，更新状态"""
    # ...
    return state

def execute_tools(state: AgentState, tools_map) -> AgentState:
    """执行工具，更新状态"""
    # ...
    return state
```

### 2. ReAct 模式

Reasoning（推理）+ Acting（行动）循环：

```
用户输入
   ↓
创建初始状态
   ↓
call_model() → LLM 推理，决定是否需要工具
   ↓
should_continue() → 判断
   ↓ (需要工具)
execute_tools() → 执行工具
   ↓
call_model() → 基于工具结果继续推理
   ↓ (不需要工具)
返回最终答案
```

### 3. LLM 驱动流程

通过 `tool_calls` 决定路由，不使用关键词匹配：

```python
# ✅ 正确：LLM 驱动
if state.get("tool_calls"):
    return "tools"

# ❌ 错误：关键词匹配
if "计算" in user_message:
    return "tool"
```

### 4. 完全复用逻辑

LangGraph Agent 100% 复用原版 Agent 的逻辑：

```python
# LangGraph 节点
def agent_node(state, llm, tools_map):
    # 完全复用 agent.py 的 call_model
    state = call_model(state, llm_with_tools)
    return state

def tool_node(state, tools_map):
    # 完全复用 agent.py 的 execute_tools
    state = execute_tools(state, tools_map)
    return state
```

## 扩展指南

### 添加新工具

在 `src/tools.py` 中定义新工具：

```python
@tool
def web_search(query: str) -> str:
    """网络搜索工具"""
    # 实现搜索逻辑
    return f"搜索结果: {query}"

def get_all_tools():
    return [calculator, knowledge_search, web_search]  # 添加到列表
```

### 扩展知识库

在 `src/knowledge_base.py` 中添加文档：

```python
KNOWLEDGE_BASE.append({
    "id": "doc9",
    "content": "新的知识内容...",
    "metadata": {"source": "来源", "category": "分类"}
})
```

然后删除 `./data/chroma_db` 目录，重新运行程序重建索引。

### 添加 LangGraph 节点

在 `src/langgraph_agent.py` 中添加新节点：

```python
def planning_node(state, llm):
    """规划节点"""
    # 实现规划逻辑
    return state

# 在 create_langgraph_agent 中添加
workflow.add_node("planning", planning_node)
workflow.add_edge("start", "planning")
```

## 预期输出

```
============================================================
AI Research Agent - 启动中...
============================================================

[1/5] 初始化 LLM...
      LLM 初始化完成

[2/5] 初始化 RAG 系统...
   📚 初始化知识库...
   📂 加载已有向量数据库...
   ✅ 已加载 8 条文档
   ✅ 知识库初始化完成

[3/5] 加载工具...
      已加载 2 个工具: calculator, knowledge_search

[4/5] 绑定工具到 LLM...
      工具绑定完成

[5/5] Agent 准备就绪!
============================================================


============================================================
测试 1/5: 请帮我计算 25 加 17 等于多少？
============================================================
🤖 调用模型...
🤔 模型决定: 需要调用 1 个工具
🔧 执行工具...
🔧 工具调用: calculator(25.0, 17.0) = 42.0
🤖 调用模型...
🤔 模型决定: 不需要工具，直接回答

[结果] 最终答案: 25 加 17 等于 42


============================================================
测试 2/5: 华东师范大学在哪里？有几个校区？
============================================================
🤖 调用模型...
🤔 模型决定: 需要调用 1 个工具
🔧 执行工具...
🔧 工具调用: knowledge_search('华东师范大学校区')
   🔍 从知识库检索...
   📄 找到 3 条相关文档
      [1] 校区信息 (相似度: 0.7234)
      [2] 学校简介 (相似度: 0.6891)
      [3] 学科设置 (相似度: 0.5432)
🤖 调用模型...
🤔 模型决定: 不需要工具，直接回答

[结果] 最终答案: 华东师范大学有两个校区：闵行校区和中山北路校区...
```

## 为什么选择这个架构？

1. **模块化**：每个模块职责单一，易于维护
2. **状态驱动**：为 LangGraph 做好准备，易于迁移
3. **可测试**：每个函数可独立测试
4. **可扩展**：添加新功能无需修改核心逻辑
5. **清晰的依赖关系**：模块之间依赖关系明确

## 常见问题

### Q: 原版 Agent 和 LangGraph Agent 有什么区别？
A: 功能完全相同，都支持多轮推理。原版使用 while 循环实现，代码简单；LangGraph 使用图结构，更易于可视化和扩展复杂流程。

### Q: 如何修改模型参数？
A: 在 `src/config.py` 中修改 `Config` 类的配置项。

### Q: 如何添加更多工具？
A: 在 `src/tools.py` 中使用 `@tool` 装饰器定义新工具，并添加到 `get_all_tools()` 返回列表中。

### Q: 如何更新知识库？
A: 修改 `src/knowledge_base.py` 后，删除 `./data/chroma_db` 目录，重新运行程序会自动重建索引。

### Q: 为什么不使用关键词匹配？
A: 关键词匹配会替代 LLM 的推理能力，无法处理同义词和复杂表达。应该让 LLM 通过 `tool_calls` 自主决定。

### Q: 如何分享代码给 AI 学习？
A: 使用 `ALL_CODE.py` 文件，包含了所有代码，方便复制粘贴给 AI。详见 `ALL_CODE_README.md`。

## 文档

- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手指南
- **[docs/RAG_IMPLEMENTATION.md](docs/RAG_IMPLEMENTATION.md)** - RAG 系统实现详解
- **[docs/LANGGRAPH_IMPLEMENTATION.md](docs/LANGGRAPH_IMPLEMENTATION.md)** - LangGraph Agent 实现详解
- **[ALL_CODE_README.md](ALL_CODE_README.md)** - 代码合集使用说明

## 为什么选择这个架构？

1. **模块化**：每个模块职责单一，易于维护
2. **状态驱动**：统一的状态管理，易于调试和扩展
3. **LLM 驱动**：让 LLM 通过 tool_calls 决定流程，不使用硬编码规则
4. **完全复用**：LangGraph 100% 复用原版逻辑，无重复代码
5. **支持多轮推理**：标准 ReAct 模式，可处理复杂任务
6. **可测试**：每个函数可独立测试
7. **可扩展**：添加新功能无需修改核心逻辑
8. **清晰的依赖关系**：模块之间依赖关系明确

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [RAG 论文](https://arxiv.org/abs/2005.11401)

## License

MIT License
