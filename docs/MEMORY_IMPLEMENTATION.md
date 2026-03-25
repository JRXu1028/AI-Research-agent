# Memory 功能实现详解

## 概述

本文档说明如何在 LangGraph Agent 中添加 Memory 功能，实现多轮对话和上下文记忆。

---

## 一、改动总览

### 1. 修改文件（1个）

```
src/
└── langgraph_agent.py    # 添加 Memory 支持
```

### 2. 新增文件（2个）

```
main_langgraph_memory.py           # 多轮对话演示程序
docs/MEMORY_IMPLEMENTATION.md      # 本文档
```

---

## 二、详细改动说明

### 改动 1：添加 Memory 支持（src/langgraph_agent.py）

#### 新增导入

```python
from langgraph.checkpoint.memory import MemorySaver
```

**作用**：
- `MemorySaver` 是 LangGraph 提供的内存检查点保存器
- 用于在内存中保存对话历史状态

#### 修改 `create_langgraph_agent` 函数

**修改前**：
```python
def create_langgraph_agent(llm, tools_map):
    # ...
    app = workflow.compile()
    return app
```

**修改后**：
```python
def create_langgraph_agent(llm, tools_map, with_memory=False):
    # ...
    if with_memory:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        print("   ✅ Memory 已启用（支持多轮对话）")
    else:
        app = workflow.compile()
    return app
```

**关键点**：
- 添加 `with_memory` 参数控制是否启用 Memory
- 使用 `checkpointer=memory` 参数编译图
- Memory 会自动保存每一步的状态

#### 新增 `run_langgraph_agent_with_memory` 函数

```python
def run_langgraph_agent_with_memory(state, llm, tools_map, thread_id):
    """运行 LangGraph Agent（带 Memory）"""
    # 创建 Agent（启用 Memory）
    agent = create_langgraph_agent(llm, tools_map, with_memory=True)
    
    # 配置：指定 thread_id 来标识对话会话
    config = {"configurable": {"thread_id": thread_id}}
    
    # 运行（会自动保存和加载历史状态）
    final_state = agent.invoke(state, config)
    
    return final_state
```

**关键点**：
- `thread_id`：对话线程 ID，用于区分不同的对话会话
- `config`：配置对象，传递给 `invoke` 方法
- 相同 `thread_id` 的调用会共享对话历史

---

## 三、Memory 工作原理

### 1. Checkpointer 机制

**什么是 Checkpointer？**
- Checkpointer 是 LangGraph 的状态持久化机制
- 在每个节点执行后自动保存状态
- 在下次调用时自动加载历史状态

**工作流程**：
```
第 1 轮对话：
  用户输入 → Agent 处理 → 保存状态（checkpoint_1）

第 2 轮对话：
  加载状态（checkpoint_1） → 用户输入 → Agent 处理 → 保存状态（checkpoint_2）

第 3 轮对话：
  加载状态（checkpoint_2） → 用户输入 → Agent 处理 → 保存状态（checkpoint_3）
```

### 2. Thread ID 的作用

**Thread ID 是什么？**
- 对话线程的唯一标识符
- 用于区分不同的对话会话
- 相同 thread_id 的调用共享历史

**示例**：
```python
# 对话 A（thread_id="user_001"）
run_langgraph_agent_with_memory(state1, llm, tools_map, "user_001")
run_langgraph_agent_with_memory(state2, llm, tools_map, "user_001")  # 能看到 state1 的历史

# 对话 B（thread_id="user_002"）
run_langgraph_agent_with_memory(state3, llm, tools_map, "user_002")  # 看不到 user_001 的历史
```

### 3. MemorySaver vs 其他 Checkpointer

| Checkpointer | 存储位置 | 持久化 | 适用场景 |
|--------------|---------|--------|---------|
| MemorySaver | 内存 | ❌ 否 | 开发、测试、短期会话 |
| SqliteSaver | SQLite 数据库 | ✅ 是 | 生产环境、长期会话 |
| RedisSaver | Redis | ✅ 是 | 分布式系统、高并发 |

**MemorySaver 的特点**：
- ✅ 简单易用，无需配置
- ✅ 性能好（内存访问）
- ❌ 程序重启后丢失
- ❌ 不适合生产环境

---

## 四、使用示例

### 示例 1：基本多轮对话

```python
from src.langgraph_agent import run_langgraph_agent_with_memory

# 初始化
llm = create_llm()
tools_map = create_tools_map(get_all_tools())
thread_id = "conversation_001"

# 第 1 轮
state1 = create_initial_state("请帮我计算 25 加 17")
result1 = run_langgraph_agent_with_memory(state1, llm, tools_map, thread_id)
# Agent: 25 + 17 = 42

# 第 2 轮（引用上一轮结果）
state2 = create_initial_state("再加上 10 呢？")
result2 = run_langgraph_agent_with_memory(state2, llm, tools_map, thread_id)
# Agent: 42 + 10 = 52（Agent 记得上一轮的 42）
```

### 示例 2：多个独立对话

```python
# 对话 A
thread_a = "user_alice"
state_a1 = create_initial_state("华东师范大学在哪里？")
result_a1 = run_langgraph_agent_with_memory(state_a1, llm, tools_map, thread_a)

state_a2 = create_initial_state("它有几个校区？")
result_a2 = run_langgraph_agent_with_memory(state_a2, llm, tools_map, thread_a)
# Agent 知道"它"指的是华东师范大学

# 对话 B（独立的会话）
thread_b = "user_bob"
state_b1 = create_initial_state("它有几个校区？")
result_b1 = run_langgraph_agent_with_memory(state_b1, llm, tools_map, thread_b)
# Agent 不知道"它"指什么（因为是新会话）
```

### 示例 3：清空历史（开始新对话）

```python
import time

# 方法 1：更换 thread_id
old_thread = "conversation_001"
new_thread = f"conversation_{int(time.time())}"  # 使用时间戳生成新 ID

# 方法 2：使用新的 Agent 实例
# 重新调用 create_langgraph_agent 会创建新的 Memory
```

---

## 五、运行演示程序

### 1. 多轮对话演示

```bash
python main_langgraph_memory.py
```

**演示内容**：
1. 第 1 轮：计算 25 + 17
2. 第 2 轮：再加上 10（测试 Memory）
3. 第 3 轮：查询华东师范大学
4. 第 4 轮：它有几个校区（测试 Memory）
5. 第 5 轮：回顾第 1 轮的答案（测试长期 Memory）

**预期输出**：
```
第 1 轮对话：简单计算
👤 用户: 请帮我计算 25 加 17 等于多少？
🤖 Agent: 25 + 17 = 42

第 2 轮对话：引用上一轮的结果（测试 Memory）
👤 用户: 再加上 10 呢？
🤖 Agent: 42 + 10 = 52

第 3 轮对话：切换话题，查询知识库
👤 用户: 华东师范大学在哪里？
🤖 Agent: 华东师范大学位于上海市...

第 4 轮对话：继续上一个话题（测试 Memory）
👤 用户: 它有几个校区？
🤖 Agent: 华东师范大学有两个校区...

第 5 轮对话：回顾更早的对话（测试长期 Memory）
👤 用户: 我刚才问的第一个问题的答案是多少？
🤖 Agent: 第一个问题的答案是 42
```

### 2. 交互式对话

程序会自动进入交互式模式：

```
🎮 进入交互式对话模式...
   输入 'quit' 或 'exit' 退出
   输入 'new' 开始新对话（清空历史）

👤 你: 你好
🤖 Agent: 你好！有什么我可以帮助你的吗？

👤 你: 请计算 100 + 200
🤖 Agent: 100 + 200 = 300

👤 你: 再乘以 2
🤖 Agent: 300 × 2 = 600

👤 你: new
🆕 已开始新对话（历史已清空）

👤 你: 再乘以 2
🤖 Agent: 抱歉，我不知道你要对什么数字乘以 2
```

---

## 六、Memory 的优势

### 1. 自然的对话体验

**无 Memory**：
```
用户: 华东师范大学在哪里？
Agent: 在上海市

用户: 它有几个校区？
Agent: 抱歉，我不知道"它"指什么
```

**有 Memory**：
```
用户: 华东师范大学在哪里？
Agent: 在上海市

用户: 它有几个校区？
Agent: 华东师范大学有两个校区
```

### 2. 支持复杂任务

**示例：多步骤任务**
```
用户: 请帮我计算 25 + 17
Agent: 42

用户: 再加上 30
Agent: 72

用户: 再减去 10
Agent: 62

用户: 最终结果是多少？
Agent: 62
```

### 3. 减少重复输入

**无 Memory**：
```
用户: 华东师范大学的校训是什么？
Agent: 求实创造，为人师表

用户: 华东师范大学有几个校区？
Agent: 两个校区
```

**有 Memory**：
```
用户: 华东师范大学的校训是什么？
Agent: 求实创造，为人师表

用户: 有几个校区？  # 不需要重复"华东师范大学"
Agent: 两个校区
```

---

## 七、注意事项

### 1. Memory 的生命周期

**MemorySaver**：
- 只在程序运行期间有效
- 程序重启后丢失
- 适合开发和测试

**生产环境建议**：
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 使用 SQLite 持久化
memory = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=memory)
```

### 2. Thread ID 管理

**建议**：
- 使用用户 ID 作为 thread_id（如 `user_12345`）
- 或使用会话 ID（如 `session_abc123`）
- 避免使用随机 ID（会导致无法恢复历史）

**示例**：
```python
# ✅ 好的做法
thread_id = f"user_{user_id}"

# ❌ 不好的做法
import uuid
thread_id = str(uuid.uuid4())  # 每次都不同，无法恢复历史
```

### 3. Memory 大小控制

**问题**：对话历史会不断增长，可能导致：
- Token 超限
- 响应变慢
- 内存占用过大

**解决方案**：
```python
# 方法 1：限制历史长度
def trim_messages(messages, max_length=10):
    """只保留最近的 N 条消息"""
    if len(messages) > max_length:
        return messages[-max_length:]
    return messages

# 方法 2：定期清空历史
if message_count > 50:
    # 开始新对话
    thread_id = f"user_{user_id}_{int(time.time())}"
```

### 4. 隐私和安全

**注意**：
- Memory 中包含完整的对话历史
- 可能包含敏感信息
- 生产环境需要考虑数据加密和访问控制

---

## 八、扩展方向

### 1. 使用 SQLite 持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def create_langgraph_agent_with_sqlite(llm, tools_map):
    workflow = StateGraph(AgentState)
    # ... 添加节点和边 ...
    
    # 使用 SQLite 持久化
    memory = SqliteSaver.from_conn_string("./data/checkpoints.db")
    app = workflow.compile(checkpointer=memory)
    
    return app
```

### 2. 添加对话摘要

```python
def summarize_conversation(messages):
    """对长对话进行摘要"""
    if len(messages) > 20:
        # 使用 LLM 生成摘要
        summary = llm.invoke(f"请总结以下对话：{messages}")
        # 用摘要替换旧消息
        return [summary] + messages[-10:]
    return messages
```

### 3. 多模态 Memory

```python
# 支持图片、文件等
class MultimodalMemory:
    def __init__(self):
        self.text_memory = MemorySaver()
        self.image_storage = {}
        self.file_storage = {}
    
    def save_image(self, thread_id, image):
        self.image_storage[thread_id] = image
    
    def load_image(self, thread_id):
        return self.image_storage.get(thread_id)
```

---

## 九、常见问题

### Q1: Memory 会占用多少内存？
A: 取决于对话长度。每条消息约 1-2KB，100 条消息约 100-200KB。

### Q2: 如何清空某个用户的历史？
A: 
```python
# 方法 1：使用新的 thread_id
new_thread_id = f"user_{user_id}_{int(time.time())}"

# 方法 2：如果使用 SQLite，可以删除数据库记录
```

### Q3: 可以跨程序共享 Memory 吗？
A: 
- MemorySaver：不可以（内存中）
- SqliteSaver：可以（数据库文件）
- RedisSaver：可以（Redis 服务器）

### Q4: Memory 会影响性能吗？
A: 
- MemorySaver：几乎无影响
- SqliteSaver：轻微影响（磁盘 I/O）
- 对话越长，LLM 处理越慢（Token 增加）

---

## 十、总结

### 核心改动

1. ✅ 添加 `MemorySaver` 支持
2. ✅ 修改 `create_langgraph_agent` 函数
3. ✅ 新增 `run_langgraph_agent_with_memory` 函数
4. ✅ 创建多轮对话演示程序

### 技术要点

- **Checkpointer**：状态持久化机制
- **Thread ID**：对话会话标识
- **MemorySaver**：内存中的 Checkpointer
- **Config**：传递配置参数

### 使用场景

- ✅ 多轮对话
- ✅ 上下文引用
- ✅ 复杂任务分解
- ✅ 个性化对话

### 下一步

- 使用 SQLite 实现持久化
- 添加对话摘要功能
- 实现对话历史管理
- 添加隐私保护机制

---

**恭喜你完成了 Memory 功能的学习！** 🎉

现在你已经掌握了：
- LangGraph Memory 的工作原理
- 如何实现多轮对话
- Thread ID 的使用方法
- Memory 的最佳实践

继续探索和实践吧！
