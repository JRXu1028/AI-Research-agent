# AI Research Agent

基于 LangChain 和 ECNU API 的状态驱动 Agent 系统。

## 项目特点

- ✅ 模块化设计，职责清晰
- ✅ 状态驱动架构，为 LangGraph 做准备
- ✅ 支持工具调用（Tool Calling）
- ✅ 集成 RAG 系统（检索增强生成）
- ✅ 向量数据库（Chroma）
- ✅ 易于扩展和维护

## 技术栈

- Python 3.9+（推荐 3.10 或 3.11）
- LangChain
- LangChain-OpenAI
- ECNU API（OpenAI 兼容）

## 项目结构

```
AI Research Agent/
├── src/                    # 源代码目录
│   ├── __init__.py        # 包初始化
│   ├── config.py          # 配置管理
│   ├── llm.py             # LLM 初始化
│   ├── tools.py           # 工具定义
│   ├── state.py           # 状态定义
│   ├── agent.py           # Agent 核心逻辑
│   ├── embeddings.py      # Embedding 模型
│   ├── vector_store.py    # 向量数据库
│   ├── knowledge_base.py  # 知识库数据
│   └── rag.py             # RAG 系统
├── data/                  # 数据目录
│   └── chroma_db/        # 向量数据库持久化
├── main.py                # 程序入口
├── requirements.txt       # 依赖列表
├── .env.example          # 环境变量模板
└── README.md             # 项目说明
```

### 模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | 管理环境变量和配置项 |
| `llm.py` | 初始化和配置语言模型 |
| `tools.py` | 定义 Agent 可用的工具（calculator, knowledge_search） |
| `state.py` | 定义 Agent 的状态结构 |
| `agent.py` | 实现 Agent 核心逻辑（状态驱动） |
| `embeddings.py` | 创建 Embedding 模型 |
| `vector_store.py` | 管理向量数据库（Chroma） |
| `knowledge_base.py` | 定义知识库数据 |
| `rag.py` | RAG 系统实现 |
| `main.py` | 程序入口，编排整体流程 |

## 快速开始

### 1. 安装依赖

```bash
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

```bash
python main.py
```

## 核心功能

### 1. 工具调用

Agent 可以自动判断何时需要使用工具：

- **calculator**: 数学计算工具
- **knowledge_search**: 知识库检索工具（RAG）

### 2. RAG 系统

检索增强生成（Retrieval-Augmented Generation）：

1. 用户提问
2. 从向量数据库检索相关文档
3. 将检索结果作为上下文
4. LLM 基于上下文生成答案

### 3. 向量数据库

使用 Chroma 存储和检索文档：

- 自动持久化到 `./data/chroma_db`
- 支持语义搜索
- 首次运行自动构建索引

## 核心概念

### 状态驱动架构

所有函数都遵循 `(state, *args) -> state` 的签名：

```python
# 状态定义
class AgentState(TypedDict):
    messages: List[Any]           # 消息历史
    tool_calls: Optional[List[dict]]  # 待执行的工具调用
    final_answer: Optional[str]   # 最终答案
    error: Optional[str]          # 错误信息

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

### 工作流程

```
用户输入
   ↓
创建初始状态
   ↓
call_model() → 模型决定是否需要工具
   ↓
should_continue() → 判断
   ↓ (需要工具)
execute_tools() → 执行工具
   ↓
call_model() → 生成最终答案
   ↓
返回最终状态
```

## 扩展指南

### 添加新工具

在 `src/tools.py` 中定义新工具：

```python
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

def get_all_tools():
    return [calculator, search]  # 添加到列表
```

### 扩展状态

在 `src/state.py` 中扩展状态定义：

```python
class AgentState(TypedDict):
    messages: List[Any]
    tool_calls: Optional[List[dict]]
    final_answer: Optional[str]
    error: Optional[str]
    user_feedback: Optional[str]  # 新增字段
```

### 添加新的状态转换函数

在 `src/agent.py` 中添加新函数：

```python
def collect_feedback(state: AgentState) -> AgentState:
    """收集用户反馈"""
    state["user_feedback"] = input("请提供反馈: ")
    return state
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

### Q: 如何修改模型参数？
A: 在 `src/config.py` 中修改 `Config` 类的配置项。

### Q: 如何添加更多工具？
A: 在 `src/tools.py` 中使用 `@tool` 装饰器定义新工具，并添加到 `get_all_tools()` 返回列表中。

### Q: 如何调试状态变化？
A: 在 `main.py` 中使用 `print_state_info()` 函数查看状态。

## 下一步

- ⏭️ 集成 LangGraph 实现复杂工作流
- ⏭️ 添加对话历史管理
- ⏭️ 实现流式输出
- ⏭️ 集成 RAG 系统

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 文档](https://platform.openai.com/docs/)
