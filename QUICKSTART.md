# 快速开始指南

## 🚀 5分钟快速上手

### 1. 环境准备

```bash
# 激活虚拟环境
conda activate AIResearch

# 安装依赖（如果还没安装）
pip install -r requirements.txt
```

### 2. 配置 API 密钥

编辑 `.env` 文件：
```bash
ECNU_API_KEY=your_api_key_here
ECNU_BASE_URL=https://api.ecnu.edu.cn/v1
ECNU_MODEL_NAME=gpt-4
```

### 3. 运行测试

#### 方式 1：原版 Agent
```bash
python main.py
```

#### 方式 2：LangGraph Agent
```bash
python main_langgraph.py
```

### 4. 查看知识库
```bash
python view_knowledge_base.py
```

---

## 📖 测试用例

程序会自动运行以下测试：

1. **计算问题**：`请帮我计算 25 加 17 等于多少？`
   - 预期：调用 calculator 工具

2. **知识查询**：`华东师范大学在哪里？有几个校区？`
   - 预期：调用 knowledge_search 工具

3. **普通对话**：`你好，今天天气怎么样？`
   - 预期：直接回答

4. **简单计算**：`123 + 456 = ?`
   - 预期：调用 calculator 工具

5. **技术问题**：`什么是 RAG 技术？`
   - 预期：调用 knowledge_search 工具

---

## 🎯 核心概念

### 1. 两种实现方式

| 特性 | 原版 Agent | LangGraph Agent |
|------|-----------|----------------|
| 实现方式 | while 循环 | 图结构 |
| 复杂度 | 简单 | 中等 |
| 可视化 | 无 | 支持 |
| 扩展性 | 中等 | 高 |
| 功能 | 完全相同 | 完全相同 |

### 2. 工作流程（ReAct 模式）

```
用户提问
  ↓
LLM 推理
  ↓
需要工具？
  ├─ 是 → 执行工具 → 回到 LLM 推理
  └─ 否 → 返回答案
```

### 3. 可用工具

- `calculator(a, b)` - 计算加法
- `knowledge_search(query)` - 搜索知识库

---

## 📁 项目结构

```
AI Research Agent/
├── src/                    # 核心代码
│   ├── agent.py           # 原版 Agent
│   ├── langgraph_agent.py # LangGraph Agent
│   ├── tools.py           # 工具定义
│   ├── rag.py             # RAG 系统
│   └── ...
├── docs/                   # 文档
│   ├── PROJECT_STATUS.md  # 项目状态
│   ├── CODE_REVIEW_SUMMARY.md
│   └── ...
├── main.py                # 原版演示
├── main_langgraph.py      # LangGraph 演示
└── requirements.txt       # 依赖
```

---

## 🔧 常用命令

### 运行程序
```bash
python main.py              # 原版 Agent
python main_langgraph.py    # LangGraph Agent
```

### 查看知识库
```bash
python view_knowledge_base.py
```

### 语法检查
```bash
python -m py_compile src/*.py
```

### 安装依赖
```bash
pip install -r requirements.txt
```

---

## 💡 使用技巧

### 1. 修改测试问题

编辑 `main.py` 或 `main_langgraph.py`：

```python
test_cases = [
    "你的问题1",
    "你的问题2",
    # ...
]
```

### 2. 添加新工具

在 `src/tools.py` 中：

```python
@tool
def my_tool(param: str) -> str:
    """工具描述"""
    # 实现逻辑
    return result

def get_all_tools():
    return [calculator, knowledge_search, my_tool]
```

### 3. 修改知识库

编辑 `src/knowledge_base.py`：

```python
KNOWLEDGE_BASE = [
    {
        "title": "新文档标题",
        "content": "文档内容..."
    },
    # ...
]
```

然后重新运行程序，会自动重建向量数据库。

### 4. 调整 RAG 检索数量

在 `src/tools.py` 的 `knowledge_search` 中：

```python
results = rag_system.retrieve(query, k=3)  # 改为 k=5 检索更多
```

---

## 🐛 常见问题

### Q: 运行时提示 "No module named 'xxx'"

**A**: 安装缺失的依赖
```bash
pip install xxx
```

### Q: API 调用失败

**A**: 检查 `.env` 文件中的 API 密钥是否正确

### Q: 知识库为空

**A**: 删除 `data/chroma_db` 文件夹，重新运行程序

### Q: 程序卡住不动

**A**: 检查网络连接，确保可以访问 API

---

## 📚 进阶阅读

- `docs/PROJECT_STATUS.md` - 项目完整状态
- `docs/CODE_REVIEW_SUMMARY.md` - 代码审查总结
- `docs/LANGGRAPH_MIGRATION.md` - LangGraph 迁移指南
- `docs/RAG_IMPLEMENTATION.md` - RAG 实现文档

---

## ✅ 验证安装

运行以下命令验证环境：

```bash
# 1. 检查 Python 版本
python --version  # 应该是 3.8+

# 2. 检查依赖
pip list | grep langchain
pip list | grep chromadb

# 3. 运行测试
python main.py
```

如果看到类似输出，说明安装成功：

```
============================================================
AI Research Agent - 启动中...
============================================================
[1/4] 初始化 LLM...
      LLM 初始化完成
[2/5] 初始化 RAG 系统...
      RAG 系统初始化完成
...
```

---

## 🎉 开始使用

现在你可以：

1. ✅ 运行演示程序
2. ✅ 修改测试问题
3. ✅ 添加新工具
4. ✅ 扩展知识库
5. ✅ 部署到生产环境

**祝使用愉快！** 🚀

---

## 📞 获取帮助

- 查看文档：`docs/` 目录
- 查看代码：`src/` 目录
- 运行示例：`main.py` 或 `main_langgraph.py`
