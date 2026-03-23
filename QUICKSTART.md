# 快速开始指南

## 5 分钟上手

### 1. 安装依赖（1 分钟）

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key（1 分钟）

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

编辑 `.env` 文件：
```
ECNU_API_KEY=你的实际API密钥
```

### 3. 运行程序（3 分钟）

```bash
python main.py
```

就这么简单！🎉

---

## 预期输出

```
🚀 初始化 LLM...
🔧 加载工具...
============================================================
✅ Agent 已启动！
📦 支持的工具: calculator
============================================================


============================================================
测试 1: 请帮我计算 25 加 17 等于多少？
============================================================

📊 初始状态:
   - messages: 1 条
   - tool_calls: None
   - final_answer: None

🤖 调用模型...
🤔 模型决定: 需要调用 1 个工具

🔧 执行工具...
🔧 工具调用: calculator(25.0, 17.0) = 42.0

🤖 调用模型...
🤔 模型决定: 不需要工具，直接回答

📊 最终状态:
   - messages: 4 条
   - tool_calls: None
   - final_answer: 25 加 17 等于 42

✅ 最终答案: 25 加 17 等于 42
============================================================
```

---

## 项目结构一览

```
AI Research Agent/
├── src/                # 源代码
│   ├── config.py      # 配置管理
│   ├── llm.py         # LLM 初始化
│   ├── tools.py       # 工具定义
│   ├── state.py       # 状态定义
│   └── agent.py       # Agent 逻辑
└── main.py            # 程序入口
```

---

## 下一步

### 添加新工具

编辑 `src/tools.py`：

```python
@tool
def multiply(a: float, b: float) -> float:
    """计算两个数的乘法"""
    return a * b

def get_all_tools():
    return [calculator, multiply]  # 添加到这里
```

### 修改测试用例

编辑 `main.py`：

```python
test_cases = [
    "你的问题 1",
    "你的问题 2",
    # 添加更多...
]
```

### 调整模型参数

编辑 `src/config.py`：

```python
class Config:
    TEMPERATURE = 0.7  # 修改温度参数
```

---

## 常见问题

### Q: 提示 "请设置环境变量 ECNU_API_KEY"
A: 确保 `.env` 文件存在且包含正确的 API Key

### Q: 如何查看详细日志？
A: 程序已经包含详细的状态打印，查看控制台输出即可

### Q: 如何只运行一个测试？
A: 修改 `main.py` 中的 `test_cases` 列表

---

## 学习路径

1. ✅ 运行程序，理解基本流程
2. 📖 阅读 `PROJECT_STRUCTURE.md` 了解架构
3. 🔧 尝试添加新工具
4. 🎨 修改状态结构
5. 🚀 准备迁移到 LangGraph

---

## 获取帮助

- 查看 `README.md` - 完整项目说明
- 查看 `PROJECT_STRUCTURE.md` - 详细架构文档
- 查看代码注释 - 每个函数都有详细说明

祝你使用愉快！🎉
