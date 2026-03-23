# 项目结构说明

## 目录结构

```
AI Research Agent/
├── src/                    # 源代码目录
│   ├── __init__.py        # 包初始化文件
│   ├── config.py          # 配置管理模块
│   ├── llm.py             # LLM 初始化模块
│   ├── tools.py           # 工具定义模块
│   ├── state.py           # 状态定义模块
│   └── agent.py           # Agent 核心逻辑模块
├── main.py                # 程序入口
├── requirements.txt       # Python 依赖列表
├── .env.example          # 环境变量模板
├── .env                  # 环境变量配置（需自行创建）
├── .python-version       # Python 版本标识
├── README.md             # 项目说明文档
└── PROJECT_STRUCTURE.md  # 本文件
```

## 模块详解

### 1. src/config.py - 配置管理

**职责**：
- 加载环境变量
- 管理 API 配置
- 验证配置完整性

**核心类**：
```python
class Config:
    ECNU_API_KEY = os.getenv("ECNU_API_KEY")
    ECNU_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
    ECNU_MODEL = "ecnu-plus"
    TEMPERATURE = 0
```

**使用场景**：
- 所有需要配置的地方都从这里读取
- 启动时验证配置是否完整

---

### 2. src/llm.py - LLM 初始化

**职责**：
- 创建 LLM 实例
- 配置模型参数
- 绑定工具到 LLM

**核心函数**：
```python
def create_llm(temperature: float = None) -> ChatOpenAI
def bind_tools_to_llm(llm, tools: list)
```

**依赖**：
- `config.py`：读取配置
- `langchain_openai.ChatOpenAI`：LLM 实现

**使用场景**：
- 程序启动时初始化 LLM
- 需要修改模型参数时

---

### 3. src/tools.py - 工具定义

**职责**：
- 定义所有可用工具
- 管理工具列表
- 创建工具映射

**核心函数**：
```python
@tool
def calculator(a: float, b: float) -> float

def get_all_tools() -> list
def create_tools_map(tools: list) -> dict
```

**扩展方式**：
```python
# 添加新工具只需 2 步：
# 1. 定义工具
@tool
def new_tool(param: str) -> str:
    """工具描述"""
    return result

# 2. 添加到列表
def get_all_tools():
    return [calculator, new_tool]
```

---

### 4. src/state.py - 状态定义

**职责**：
- 定义 Agent 状态结构
- 创建初始状态
- 提供状态类型提示

**核心类型**：
```python
class AgentState(TypedDict):
    messages: List[Any]           # 消息历史
    tool_calls: Optional[List[dict]]  # 待执行的工具调用
    final_answer: Optional[str]   # 最终答案
    error: Optional[str]          # 错误信息
```

**核心函数**：
```python
def create_initial_state(user_input: str) -> AgentState
```

**扩展方式**：
```python
# 添加新字段
class AgentState(TypedDict):
    messages: List[Any]
    tool_calls: Optional[List[dict]]
    final_answer: Optional[str]
    error: Optional[str]
    user_feedback: Optional[str]  # 新增
```

---

### 5. src/agent.py - Agent 核心逻辑

**职责**：
- 实现状态驱动的 Agent 逻辑
- 定义状态转换函数
- 编排执行流程

**核心函数**：
```python
def call_model(state, llm_with_tools) -> state
def execute_tools(state, tools_map) -> state
def should_continue(state) -> bool
def run_agent(state, llm_with_tools, tools_map) -> state
```

**设计模式**：
- 所有函数签名：`(state, *args) -> state`
- 纯函数：不修改外部状态
- 可组合：函数可以任意组合

**执行流程**：
```
run_agent()
  ├─> call_model()      # 第一次调用模型
  ├─> should_continue() # 判断是否需要工具
  ├─> execute_tools()   # 执行工具（如果需要）
  └─> call_model()      # 第二次调用模型（生成最终答案）
```

---

### 6. main.py - 程序入口

**职责**：
- 初始化所有组件
- 编排整体流程
- 运行测试用例
- 打印结果

**执行流程**：
```python
1. create_llm()           # 初始化 LLM
2. get_all_tools()        # 获取工具
3. create_tools_map()     # 创建工具映射
4. bind_tools_to_llm()    # 绑定工具
5. create_initial_state() # 创建初始状态
6. run_agent()            # 运行 Agent
7. 打印结果
```

---

## 模块依赖关系

```
main.py
  ├─> llm.py
  │    └─> config.py
  ├─> tools.py
  ├─> state.py
  └─> agent.py
       └─> state.py
```

**依赖层级**：
1. 底层：`config.py`, `state.py`（无依赖）
2. 中层：`llm.py`, `tools.py`（依赖底层）
3. 高层：`agent.py`（依赖中层）
4. 入口：`main.py`（依赖所有）

---

## 数据流

```
用户输入
   ↓
main.py: create_initial_state()
   ↓
state = {
    "messages": [HumanMessage(...)],
    "tool_calls": None,
    "final_answer": None,
    "error": None
}
   ↓
agent.py: run_agent(state, llm, tools_map)
   ↓
agent.py: call_model(state, llm)
   ↓
state = {
    "messages": [..., AIMessage(...)],
    "tool_calls": [...],  # 如果需要工具
    ...
}
   ↓
agent.py: execute_tools(state, tools_map)
   ↓
state = {
    "messages": [..., ToolMessage(...)],
    "tool_calls": None,
    ...
}
   ↓
agent.py: call_model(state, llm)
   ↓
state = {
    "messages": [...],
    "tool_calls": None,
    "final_answer": "...",  # 最终答案
    ...
}
   ↓
main.py: 打印结果
```

---

## 设计原则

### 1. 单一职责原则（SRP）
每个模块只负责一件事：
- `config.py`：只管配置
- `llm.py`：只管 LLM
- `tools.py`：只管工具
- `state.py`：只管状态
- `agent.py`：只管 Agent 逻辑

### 2. 开闭原则（OCP）
对扩展开放，对修改封闭：
- 添加新工具：只需修改 `tools.py`
- 扩展状态：只需修改 `state.py`
- 添加新功能：只需添加新函数

### 3. 依赖倒置原则（DIP）
高层模块不依赖低层模块，都依赖抽象：
- `agent.py` 不直接依赖具体工具，而是依赖 `tools_map`
- `main.py` 通过接口调用各模块

### 4. 状态驱动设计
所有函数都是 `(state, *args) -> state`：
- 易于测试
- 易于调试
- 易于组合
- 为 LangGraph 做准备

---

## 扩展指南

### 添加新工具

1. 在 `src/tools.py` 中定义：
```python
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"
```

2. 添加到工具列表：
```python
def get_all_tools():
    return [calculator, search]
```

### 扩展状态

在 `src/state.py` 中：
```python
class AgentState(TypedDict):
    messages: List[Any]
    tool_calls: Optional[List[dict]]
    final_answer: Optional[str]
    error: Optional[str]
    metadata: Optional[dict]  # 新增
```

### 添加新的状态转换函数

在 `src/agent.py` 中：
```python
def validate_answer(state: AgentState) -> AgentState:
    """验证答案"""
    # 验证逻辑
    return state
```

然后在 `run_agent()` 中使用：
```python
def run_agent(state, llm_with_tools, tools_map):
    state = call_model(state, llm_with_tools)
    state = validate_answer(state)  # 插入新步骤
    # ...
    return state
```

---

## 测试建议

### 单元测试

每个模块都可以独立测试：

```python
# 测试 tools.py
def test_calculator():
    result = calculator.invoke({"a": 2, "b": 3})
    assert result == 5

# 测试 state.py
def test_create_initial_state():
    state = create_initial_state("test")
    assert len(state["messages"]) == 1
    assert state["tool_calls"] is None

# 测试 agent.py
def test_should_continue():
    state = {"tool_calls": [{"name": "test"}]}
    assert should_continue(state) == True
```

### 集成测试

测试整个流程：

```python
def test_agent_with_tool():
    llm = create_llm()
    tools = get_all_tools()
    tools_map = create_tools_map(tools)
    llm_with_tools = bind_tools_to_llm(llm, tools)
    
    state = create_initial_state("1 + 2 = ?")
    final_state = run_agent(state, llm_with_tools, tools_map)
    
    assert final_state["final_answer"] is not None
```

---

## 常见问题

### Q: 为什么要模块化？
A: 
- 职责清晰，易于维护
- 代码复用性高
- 易于测试和调试
- 团队协作更方便

### Q: 如何调试状态变化？
A: 在 `main.py` 中使用 `print_state_info()` 函数，或在每个状态转换函数中添加日志。

### Q: 如何迁移到 LangGraph？
A: 当前架构已经为 LangGraph 做好准备，只需：
1. 将状态转换函数映射为节点
2. 使用 `should_continue()` 定义条件边
3. 用 LangGraph 的 API 构建图

---

## 总结

这个模块化结构的优势：

1. ✅ 清晰的职责划分
2. ✅ 易于扩展和维护
3. ✅ 状态驱动设计
4. ✅ 为 LangGraph 做好准备
5. ✅ 易于测试和调试
6. ✅ 代码复用性高

这是一个生产级的项目结构，可以直接用于实际开发！
