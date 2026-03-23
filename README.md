
# AI-Research-agent
正在学习agent搭建...这是我的第一个agent项目，我将不断优化

# AI Research Agent

基于 LangChain 和 ECNU API 的状态驱动 Agent 系统。

## 项目特点

- ✅ 模块化设计，职责清晰
- ✅ 状态驱动架构，为 LangGraph 做准备
- ✅ 支持工具调用（Tool Calling）
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
│   └── agent.py           # Agent 核心逻辑
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
| `tools.py` | 定义 Agent 可用的工具 |
| `state.py` | 定义 Agent 的状态结构 |
| `agent.py` | 实现 Agent 核心逻辑（状态驱动） |
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
✅ Agent 已启动！
📦 支持的工具: calculator

测试 1: 请帮我计算 25 加 17 等于多少？
🤖 调用模型...
🤔 模型决定: 需要调用 1 个工具
🔧 执行工具...
🔧 工具调用: calculator(25.0, 17.0) = 42.0
🤖 调用模型...
🤔 模型决定: 不需要工具，直接回答
✅ 最终答案: 25 加 17 等于 42
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
