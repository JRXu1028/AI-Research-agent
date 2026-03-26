"""
=============================================================================
AI Research Agent - 完整代码合集
=============================================================================

本文件包含项目的所有核心代码，方便分享给 AI 学习。
每个代码块都标注了来源文件。

项目结构：
- src/config.py          # 配置管理
- src/state.py           # 状态定义
- src/llm.py             # LLM 初始化
- src/embeddings.py      # Embedding 模型
- src/vector_store.py    # 向量数据库
- src/knowledge_base.py  # 知识库数据
- src/rag.py             # RAG 系统
- src/tools.py           # 工具定义
- src/agent.py           # Agent 核心逻辑
- src/langgraph_agent.py # LangGraph Agent
- main.py                # 原版 Agent 演示
- main_langgraph.py      # LangGraph Agent 演示
- main_langgraph_memory.py # LangGraph Agent with Memory 演示
- app.py                 # FastAPI Web 服务
- static/index.html      # Web 聊天界面
- view_knowledge_base.py # 查看知识库工具

=============================================================================
"""

# =============================================================================
# 文件: src/config.py
# 说明: 配置管理模块
# =============================================================================

"""
配置模块
管理环境变量和配置项
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置类"""
    
    # ECNU API 配置
    ECNU_API_KEY = os.getenv("ECNU_API_KEY")
    ECNU_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
    ECNU_MODEL = "ecnu-plus"
    
    # LLM 配置
    TEMPERATURE = 0  # 0 = 更确定的输出
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.ECNU_API_KEY:
            raise ValueError(
                "请设置环境变量 ECNU_API_KEY\n"
                "1. 复制 .env.example 为 .env\n"
                "2. 在 .env 中填入你的 API Key"
            )


# =============================================================================
# 文件: src/state.py
# 说明: 状态定义模块
# =============================================================================

"""
状态模块
定义 Agent 的状态结构
"""

from typing import TypedDict, List, Optional, Any


class AgentState(TypedDict):
    """
    Agent 的状态定义
    所有函数都基于这个状态进行输入输出
    """
    messages: List[Any]  # 消息历史
    tool_calls: Optional[List[dict]]  # 待执行的工具调用
    final_answer: Optional[str]  # 最终答案
    error: Optional[str]  # 错误信息


def create_initial_state(user_input: str) -> AgentState:
    """
    创建初始状态
    
    Args:
        user_input: 用户输入的问题
    
    Returns:
        初始化的 AgentState
    """
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [HumanMessage(content=user_input)],
        "tool_calls": None,
        "final_answer": None,
        "error": None,
    }


# =============================================================================
# 文件: src/llm.py
# 说明: LLM 初始化模块
# =============================================================================

"""
LLM 模块
负责初始化和配置语言模型
"""

from langchain_openai import ChatOpenAI
# from .config import Config  # 实际使用时需要导入


def create_llm(temperature: float = None):
    """
    创建 LLM 实例
    
    Args:
        temperature: 温度参数，控制输出的随机性（0-1）
                    None 则使用配置文件中的默认值
    
    Returns:
        ChatOpenAI 实例
    """
    Config.validate()
    
    return ChatOpenAI(
        model=Config.ECNU_MODEL,
        openai_api_key=Config.ECNU_API_KEY,
        openai_api_base=Config.ECNU_BASE_URL,
        temperature=temperature if temperature is not None else Config.TEMPERATURE,
    )


def bind_tools_to_llm(llm, tools: list):
    """
    将工具绑定到 LLM
    
    Args:
        llm: LLM 实例
        tools: 工具列表
    
    Returns:
        绑定了工具的 LLM
    """
    return llm.bind_tools(tools)



# =============================================================================
# 文件: src/embeddings.py
# 说明: Embedding 模型模块
# =============================================================================

"""
Embedding 模块
提供文本向量化功能
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


def create_embeddings():
    """
    创建 Embedding 模型
    
    使用 HuggingFace 的 sentence-transformers 模型
    这是一个本地模型，不需要 API 调用
    
    Returns:
        Embeddings 实例
    """
    # 使用轻量级的 embedding 模型
    # 首次运行会自动下载模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # 使用 CPU
        encode_kwargs={'normalize_embeddings': True}  # 归一化向量
    )
    
    return embeddings


# =============================================================================
# 文件: src/knowledge_base.py
# 说明: 知识库数据模块
# =============================================================================

"""
知识库模块
定义知识库数据
"""

# 简单的知识库数据
KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "content": "华东师范大学（East China Normal University，简称华东师大或ECNU）创建于1951年10月16日，是新中国成立后组建的第一所社会主义师范大学。学校位于上海市，是教育部直属的全国重点大学，是国家'985工程'和'211工程'重点建设高校。",
        "metadata": {"source": "学校简介", "category": "基本信息"}
    },
    {
        "id": "doc2",
        "content": "华东师范大学有两个校区：闵行校区和中山北路校区。闵行校区位于上海市闵行区东川路500号，占地面积约207公顷。中山北路校区位于上海市普陀区中山北路3663号，占地面积约33公顷。",
        "metadata": {"source": "校区信息", "category": "地理位置"}
    },
    {
        "id": "doc3",
        "content": "华东师范大学设有教育学部、人文社会科学学院、理工学院等多个学院。学校拥有教育学、心理学、地理学、生态学等多个国家重点学科。学校现有全日制本科生约15000人，研究生约20000人。",
        "metadata": {"source": "学科设置", "category": "教学科研"}
    },
    {
        "id": "doc4",
        "content": "华东师范大学的校训是'求实创造，为人师表'。学校秉承'智慧的创获，品性的陶熔，民族和社会的发展'的大学理想，致力于培养具有创新精神和实践能力的高素质人才。",
        "metadata": {"source": "校园文化", "category": "文化理念"}
    },
    {
        "id": "doc5",
        "content": "华东师范大学图书馆是全国重点大学图书馆之一，馆藏纸质图书约400万册，电子图书约300万册。图书馆提供24小时自助服务，为师生提供良好的学习环境。",
        "metadata": {"source": "图书馆", "category": "设施服务"}
    },
    {
        "id": "doc6",
        "content": "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python以其简洁的语法和强大的功能而闻名，广泛应用于Web开发、数据科学、人工智能、自动化等领域。",
        "metadata": {"source": "编程知识", "category": "技术"}
    },
    {
        "id": "doc7",
        "content": "LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，可以轻松构建复杂的AI应用，包括聊天机器人、问答系统、文档分析等。LangChain支持多种LLM提供商。",
        "metadata": {"source": "AI框架", "category": "技术"}
    },
    {
        "id": "doc8",
        "content": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。它首先从知识库中检索相关文档，然后将检索到的信息作为上下文提供给语言模型，从而生成更准确、更有依据的回答。",
        "metadata": {"source": "AI技术", "category": "技术"}
    }
]


def get_knowledge_base():
    """
    获取知识库数据
    
    Returns:
        知识库文档列表
    """
    return KNOWLEDGE_BASE


def add_document(doc_id: str, content: str, source: str, category: str):
    """
    添加新文档到知识库
    
    Args:
        doc_id: 文档ID
        content: 文档内容
        source: 来源
        category: 分类
    """
    new_doc = {
        "id": doc_id,
        "content": content,
        "metadata": {"source": source, "category": category}
    }
    KNOWLEDGE_BASE.append(new_doc)
    return new_doc


def get_knowledge_summary():
    """
    获取知识库摘要信息
    
    Returns:
        知识库统计信息
    """
    categories = {}
    for doc in KNOWLEDGE_BASE:
        category = doc["metadata"]["category"]
        categories[category] = categories.get(category, 0) + 1
    
    return {
        "total_docs": len(KNOWLEDGE_BASE),
        "categories": categories,
        "sources": [doc["metadata"]["source"] for doc in KNOWLEDGE_BASE]
    }



# =============================================================================
# 文件: src/vector_store.py
# 说明: 向量数据库模块
# =============================================================================

"""
向量数据库模块
使用 Chroma 管理文档向量
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# from .embeddings import create_embeddings
# from .knowledge_base import get_knowledge_base
import os


class VectorStore:
    """向量数据库管理类"""
    
    def __init__(self, persist_directory="./data/chroma_db"):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 数据库持久化目录
        """
        self.persist_directory = persist_directory
        self.embeddings = create_embeddings()
        self.vectorstore = None
        
    def initialize(self, force_reload=False):
        """
        初始化或加载向量数据库
        
        Args:
            force_reload: 是否强制重新加载数据
        """
        # 检查是否已存在数据库
        if os.path.exists(self.persist_directory) and not force_reload:
            print("      📂 加载已有向量数据库...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"      ✅ 已加载 {self.vectorstore._collection.count()} 条文档")
        else:
            print("      🔨 创建新的向量数据库...")
            # 加载知识库数据
            knowledge_base = get_knowledge_base()
            
            # 转换为 LangChain Document 格式
            documents = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
                for doc in knowledge_base
            ]
            
            # 创建向量数据库
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"      ✅ 已创建向量数据库，包含 {len(documents)} 条文档")
    
    def search(self, query: str, k: int = 3):
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        # 相似度搜索
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_score(self, query: str, k: int = 3):
        """
        检索相关文档（带相似度分数）
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            (文档, 分数) 的列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        # 相似度搜索（带分数）
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def add_documents(self, documents):
        """
        添加新文档到向量数据库
        
        Args:
            documents: Document 对象列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        self.vectorstore.add_documents(documents)
        print(f"      ✅ 已添加 {len(documents)} 条新文档")


def create_vector_store(force_reload=False):
    """
    创建并初始化向量数据库
    
    Args:
        force_reload: 是否强制重新加载数据
    
    Returns:
        VectorStore 实例
    """
    store = VectorStore()
    store.initialize(force_reload=force_reload)
    return store


# =============================================================================
# 文件: src/rag.py
# 说明: RAG 系统模块
# =============================================================================

"""
RAG 模块
实现检索增强生成
"""

# from .vector_store import VectorStore

# 全局 RAG 系统实例
_rag_system_instance = None


class RAGSystem:
    """RAG 系统类"""
    
    def __init__(self, vector_store):
        """
        初始化 RAG 系统
        
        Args:
            vector_store: 向量数据库
        """
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 3):
        """
        检索相关文档
        
        Args:
            query: 用户查询
            k: 返回文档数量
        
        Returns:
            相关文档列表 [(doc, score), ...]
        """
        print(f"   🔍 从知识库检索...")
        results = self.vector_store.search_with_score(query, k=k)
        
        print(f"   📄 找到 {len(results)} 条相关文档")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"      [{i}] {source} (相似度: {score:.4f})")
        
        return results


def initialize_rag_system(force_reload=False):
    """
    初始化全局 RAG 系统
    
    Args:
        force_reload: 是否强制重新加载向量数据库
    
    Returns:
        RAGSystem 实例
    """
    global _rag_system_instance
    
    if _rag_system_instance is None or force_reload:
        # from .vector_store import create_vector_store
        
        print("   📚 初始化知识库...")
        vector_store = create_vector_store(force_reload=force_reload)
        _rag_system_instance = RAGSystem(vector_store)
        print("   ✅ 知识库初始化完成")
    
    return _rag_system_instance


def get_rag_system():
    """
    获取全局 RAG 系统实例
    
    Returns:
        RAGSystem 实例，如果未初始化则返回 None
    """
    return _rag_system_instance



# =============================================================================
# 文件: src/tools.py
# 说明: 工具定义模块
# =============================================================================

"""
工具模块
定义 Agent 可以使用的所有工具
"""

from langchain_core.tools import tool


@tool
def calculator(a: float, b: float) -> float:
    """
    计算两个数的加法
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两个数字的和
    """
    result = a + b
    print(f"🔧 工具调用: calculator({a}, {b}) = {result}")
    return result


@tool
def knowledge_search(query: str) -> str:
    """
    从知识库中搜索相关信息
    
    当用户询问关于华东师范大学、Python、LangChain、RAG等知识库中的内容时使用此工具。
    
    Args:
        query: 搜索查询
    
    Returns:
        相关的知识库内容
    """
    # from .rag import get_rag_system
    
    print(f"🔧 工具调用: knowledge_search('{query}')")
    
    # 获取 RAG 系统实例
    rag_system = get_rag_system()
    
    if rag_system is None:
        return "知识库未初始化，无法搜索"
    
    # 检索相关文档
    results = rag_system.retrieve(query, k=3)
    
    if not results:
        return "未找到相关信息"
    
    # 格式化返回结果
    context = "\n\n".join([
        f"【文档{i+1}】(相似度: {score:.4f})\n{doc.page_content}"
        for i, (doc, score) in enumerate(results)
    ])
    
    return context


def get_all_tools():
    """
    获取所有可用工具
    
    Returns:
        工具列表
    """
    return [calculator, knowledge_search]


def create_tools_map(tools: list) -> dict:
    """
    创建工具名称到工具对象的映射
    
    Args:
        tools: 工具列表
    
    Returns:
        {工具名称: 工具对象} 的字典
    """
    return {tool.name: tool for tool in tools}


# =============================================================================
# 文件: src/agent.py
# 说明: Agent 核心逻辑模块
# =============================================================================

"""
Agent 模块
实现状态驱动的 Agent 核心逻辑
"""

from langchain_core.messages import ToolMessage
# from .state import AgentState


def call_model(state, llm_with_tools):
    """
    调用 LLM，基于当前状态生成响应
    
    Args:
        state: 当前状态
        llm_with_tools: 绑定了工具的 LLM
    
    Returns:
        更新后的状态
    """
    print(f"\n🤖 调用模型...")
    
    # 从状态中获取消息
    messages = state["messages"]
    
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    
    # 更新状态：添加 AI 响应
    state["messages"].append(response)
    
    # 检查是否有工具调用
    if response.tool_calls:
        print(f"🤔 模型决定: 需要调用 {len(response.tool_calls)} 个工具")
        state["tool_calls"] = response.tool_calls
    else:
        print(f"🤔 模型决定: 不需要工具，直接回答")
        state["tool_calls"] = None
        state["final_answer"] = response.content
    
    return state


def execute_tools(state, tools_map: dict):
    """
    执行工具调用，基于状态中的 tool_calls
    
    Args:
        state: 当前状态
        tools_map: 工具名称到工具对象的映射
    
    Returns:
        更新后的状态
    """
    tool_calls = state.get("tool_calls")
    
    if not tool_calls:
        print("⏭️  无需执行工具")
        return state
    
    print(f"\n🔧 执行工具...")
    
    # 执行所有工具调用
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 从工具映射中获取工具并调用
        tool = tools_map.get(tool_name)
        if tool:
            tool_result = tool.invoke(tool_args)
        else:
            tool_result = f"错误: 未找到工具 '{tool_name}'"
            print(f"⚠️  {tool_result}")
        
        # 将工具结果添加到状态的消息历史
        state["messages"].append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )
    
    # 清空 tool_calls（已执行完毕）
    state["tool_calls"] = None
    
    return state


def should_continue(state) -> bool:
    """
    判断是否需要继续执行（路由函数）
    
    Args:
        state: 当前状态
    
    Returns:
        True（继续）或 False（结束）
    """
    # 如果有待执行的工具调用，继续
    if state.get("tool_calls"):
        return True
    
    # 如果已有最终答案，结束
    if state.get("final_answer"):
        return False
    
    # 默认继续
    return True


def run_agent(state, llm_with_tools, tools_map: dict, max_iterations: int = 10):
    """
    运行 Agent 的主函数（状态驱动，支持多轮推理）
    
    Args:
        state: 初始状态
        llm_with_tools: 绑定了工具的 LLM
        tools_map: 工具映射字典
        max_iterations: 最大迭代次数（防止无限循环）
    
    Returns:
        最终状态
    
    工作流程（ReAct 循环）：
    1. 调用模型 -> 更新 state
    2. 如果需要工具 -> 执行工具 -> 回到步骤 1
    3. 如果不需要工具 -> 返回最终答案
    """
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Step 1: 调用模型
        state = call_model(state, llm_with_tools)
        
        # Step 2: 检查是否需要继续
        if not should_continue(state):
            # 已有最终答案，结束
            break
        
        # Step 3: 执行工具
        state = execute_tools(state, tools_map)
        
        # 继续下一轮循环
    
    if iteration >= max_iterations:
        print(f"⚠️  达到最大迭代次数 ({max_iterations})，强制结束")
        if not state.get("final_answer"):
            state["final_answer"] = "抱歉，推理过程超时"
    
    return state



# =============================================================================
# =============================================================================
# 文件: src/langgraph_agent.py
# 说明: LangGraph Agent 实现模块（支持 Memory）
# =============================================================================

"""
LangGraph Agent 模块
基于 LangGraph 实现的状态驱动 Agent（标准 ReAct 模式）
支持 Memory（对话历史持久化）
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from functools import partial
# from .state import AgentState
# from .agent import call_model, execute_tools


# ============================================================
# 1. 节点函数（完全复用现有逻辑）
# ============================================================

def agent_node(state, llm, tools_map):
    """
    Agent 节点：调用 LLM 并让其决定是否使用工具
    
    完全复用 agent.py 的 call_model 逻辑，保留 tool_calls 机制
    """
    print(f"\n🤖 [Agent节点] 调用模型...")
    
    # 绑定工具到 LLM
    tools = list(tools_map.values())
    llm_with_tools = llm.bind_tools(tools)
    
    # 复用原有逻辑
    state = call_model(state, llm_with_tools)
    
    return state


def tool_node(state, tools_map):
    """
    工具节点：执行 LLM 决定的工具调用
    
    完全复用 agent.py 的 execute_tools 逻辑
    """
    print(f"\n🔧 [工具节点] 执行工具...")
    
    # 复用原有逻辑
    state = execute_tools(state, tools_map)
    
    return state


# ============================================================
# 2. 路由函数（基于 tool_calls，不是关键词）
# ============================================================

def should_continue_langgraph(state) -> str:
    """
    路由函数：基于 LLM 的 tool_calls 决定下一步
    
    让 LLM 通过 tool_calls 驱动流程，不使用关键词匹配
    
    Returns:
        "tools" - LLM 决定调用工具
        "end" - LLM 给出最终答案
    """
    # 检查是否有待执行的工具调用
    if state.get("tool_calls"):
        print(f"🔀 [Router] LLM 决定调用工具")
        return "tools"
    
    # 检查是否有最终答案
    if state.get("final_answer"):
        print(f"🔀 [Router] LLM 给出最终答案")
        return "end"
    
    # 默认继续（此项目理论上不应该到这里）
    # 因为逻辑为：
    # if response.tool_calls:
    #     state["tool_calls"] = response.tool_calls
    # else:
    #     state["final_answer"] = response.content
    print(f"🔀 [Router] 继续推理")
    return "continue"


# ============================================================
# 3. LangGraph 构建（标准 ReAct 模式 + Memory 支持）
# ============================================================

def create_langgraph_agent(llm, tools_map, with_memory=False):
    """
    创建 LangGraph Agent（标准 ReAct 模式 + Memory 支持）
    
    Args:
        llm: 语言模型
        tools_map: 工具映射字典
        with_memory: 是否启用 Memory（对话历史持久化）
    
    流程：
    START → agent (LLM 推理)
             ↓ (有 tool_calls)
           tools (执行工具)
             ↓
           agent (基于工具结果继续推理)
             ↓ (无 tool_calls)
            END
    
    特点：
    - 支持多轮推理
    - LLM 驱动流程
    - 完全复用现有逻辑
    - 可选的 Memory 支持（记住对话历史）
    """
    # 创建状态图
    workflow = StateGraph(AgentState)  # AgentState 定义了这个图中"状态"的数据结构
    
    # 使用 partial 绑定上下文，避免污染状态
    # 把函数需要的部分参数提前固定，生成一个新的函数，从而在调用时自动携带这些参数（即上下文）
    
    # 使用 functools.partial 对 agent_node 进行"参数预绑定"
    # 相当于创建一个新的函数 agent_with_context
    # 这个函数已经默认带上了：
    # - llm
    # - tools_map
    #
    # 后续调用 agent_with_context(state) 时：
    # 实际等价于 agent_node(state, llm=llm, tools_map=tools_map)
    agent_with_context = partial(agent_node, llm=llm, tools_map=tools_map)
    tool_with_context = partial(tool_node, tools_map=tools_map)
    
    # 添加节点
    workflow.add_node("agent", agent_with_context)
    workflow.add_node("tools", tool_with_context)
    
    # 设置入口点：指定流程从哪个节点开始执行
    workflow.set_entry_point("agent")
    
    # 添加条件边：agent → tools 或 END
    # 含义：从 "agent" 节点出来之后，调用 should_continue(state)，根据返回值决定下一步走向
    workflow.add_conditional_edges(
        "agent",
        should_continue_langgraph,
        {
            "tools": "tools",
            "end": END,
            "continue": "agent"  # 支持继续推理（此项目理论不会到这）
        }
    )
    
    # 添加边：tools → agent（形成 ReAct 循环）
    # 含义：从 "tools" 节点出来之后，进入 "agent" 节点
    workflow.add_edge("tools", "agent")
    
    # 编译图：把定义的流程图 "编译" 成一个可执行对象
    # 之后就可以：app.invoke(initial_state) 来运行整个 Agent
    if with_memory:
        # 使用 MemorySaver 作为 checkpointer
        # 这会自动保存每一步的状态，支持对话历史
        # 👉 启用 Memory 后：LangGraph 会自动保存每一步 state
        # Memory 的作用：
        # ✔ 记住对话历史
        # ✔ 支持多轮对话
        # ✔ 每个 thread_id 是一个会话 config = {"configurable": {"thread_id": thread_id}}
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        print("   ✅ Memory 已启用（支持多轮对话）")
    else:
        app = workflow.compile()
    
    return app


# ============================================================
# 4. 便捷函数
# ============================================================

def run_langgraph_agent(state, llm, tools_map: dict):
    """
    运行 LangGraph Agent（无 Memory）
    
    Args:
        state: 初始状态
        llm: 语言模型
        tools_map: 工具映射字典
    
    Returns:
        最终状态
    """
    # 创建 LangGraph Agent
    agent = create_langgraph_agent(llm, tools_map, with_memory=False)
    
    # 运行
    # 执行这个 Agent，从 state 开始，按照图的流程自动运行：
    # 实际发生：
    # state
    #   ↓
    # agent_node（LLM）
    #   ↓
    # 是否调用工具？
    #   ↓
    # tool_node（如果需要）
    #   ↓
    # 再回 agent_node
    #   ↓
    # 直到 should_continue 返回 "end"
    final_state = agent.invoke(state)
    
    # 返回最终状态（里面包含 final_answer 等信息）
    return final_state


def run_langgraph_agent_with_memory(state, llm, tools_map: dict, thread_id: str):
    """
    运行 LangGraph Agent（带 Memory）
    
    Args:
        state: 初始状态
        llm: 语言模型
        tools_map: 工具映射字典
        thread_id: 对话线程 ID（用于区分不同的对话会话）
    
    Returns:
        最终状态
    """
    # 创建 LangGraph Agent（启用 Memory）
    agent = create_langgraph_agent(llm, tools_map, with_memory=True)
    
    # 配置：指定 thread_id 来标识对话会话
    config = {"configurable": {"thread_id": thread_id}}
    
    # 运行（会自动保存和加载历史状态）
    # 👉 和普通 invoke 的区别：
    # 会自动：读取历史、保存新状态、恢复上下文
    final_state = agent.invoke(state, config)
    
    return final_state


# =============================================================================
# 文件: main.py
# 说明: 原版 Agent 演示程序
# =============================================================================

"""
主程序入口
运行 AI Research Agent
"""

# import sys
# import io
# 
# # 设置标准输出为 UTF-8
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 
# from src.llm import create_llm, bind_tools_to_llm
# from src.tools import get_all_tools, create_tools_map
# from src.state import create_initial_state
# from src.agent import run_agent
# from src.rag import initialize_rag_system


def print_state_info(state, label):
    """打印状态信息（用于调试）"""
    print(f"\n[状态] {label}:")
    print(f"   - messages: {len(state['messages'])} 条")
    print(f"   - tool_calls: {state['tool_calls']}")
    if state['final_answer']:
        answer_preview = state['final_answer'][:50] + "..." if len(state['final_answer']) > 50 else state['final_answer']
        print(f"   - final_answer: {answer_preview}")
    else:
        print(f"   - final_answer: {state['final_answer']}")


def main():
    """主函数"""
    
    print("=" * 60)
    print("AI Research Agent - 启动中...")
    print("=" * 60)
    
    # 1. 初始化 LLM
    print("\n[1/4] 初始化 LLM...")
    llm = create_llm()
    print("      LLM 初始化完成")
    
    # 2. 初始化 RAG 系统（知识库）
    print("\n[2/5] 初始化 RAG 系统...")
    initialize_rag_system(force_reload=False)
    
    # 3. 获取工具
    print("\n[3/5] 加载工具...")
    tools = get_all_tools()
    tools_map = create_tools_map(tools)
    print(f"      已加载 {len(tools)} 个工具: {', '.join(tools_map.keys())}")
    
    # 4. 绑定工具到 LLM
    print("\n[4/5] 绑定工具到 LLM...")
    llm_with_tools = bind_tools_to_llm(llm, tools)
    print("      工具绑定完成")
    
    print("\n[5/5] Agent 准备就绪!")
    print("=" * 60)
    
    # 5. 测试用例
    test_cases = [
        "请帮我计算 25 加 17 等于多少？",
        "华东师范大学在哪里？有几个校区？",
        "华东师范大学的校训是什么？",
        "123 + 456 = ?",
        "什么是 RAG 技术？",
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {question}")
        print('='*60)
        
        try:
            # 创建初始状态
            initial_state = create_initial_state(question)
            print_state_info(initial_state, "初始状态")
            
            # 运行 Agent
            final_state = run_agent(initial_state, llm_with_tools, tools_map)
            
            # 打印结果
            print_state_info(final_state, "最终状态")
            print(f"\n[结果] 最终答案: {final_state['final_answer']}")
            
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
        
        print('='*60)
    
    print("\n\n所有测试完成!")


# if __name__ == "__main__":
#     main()



# =============================================================================
# 文件: main_langgraph.py
# 说明: LangGraph Agent 演示程序
# =============================================================================

"""
LangGraph Agent 演示
展示如何使用 LangGraph 构建 Agent
"""

# import sys
# import io
# 
# # 设置标准输出为 UTF-8
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 
# from src.llm import create_llm
# from src.tools import get_all_tools, create_tools_map
# from src.state import create_initial_state
# from src.rag import initialize_rag_system
# from src.langgraph_agent import run_langgraph_agent


def print_state_info_langgraph(state, label):
    """打印状态信息（用于调试）"""
    print(f"\n[状态] {label}:")
    print(f"   - messages: {len(state['messages'])} 条")
    if state.get('final_answer'):
        answer_preview = state['final_answer'][:50] + "..." if len(state['final_answer']) > 50 else state['final_answer']
        print(f"   - final_answer: {answer_preview}")
    else:
        print(f"   - final_answer: {state.get('final_answer')}")


def main_langgraph():
    """主函数"""
    
    print("=" * 60)
    print("LangGraph Agent - 启动中...")
    print("=" * 60)
    
    # 1. 初始化 LLM
    print("\n[1/4] 初始化 LLM...")
    llm = create_llm()
    print("      LLM 初始化完成")
    
    # 2. 初始化 RAG 系统
    print("\n[2/4] 初始化 RAG 系统...")
    initialize_rag_system(force_reload=False)
    
    # 3. 获取工具（不需要绑定到 LLM）
    print("\n[3/4] 加载工具...")
    tools = get_all_tools()
    tools_map = create_tools_map(tools)
    print(f"      已加载 {len(tools)} 个工具: {', '.join(tools_map.keys())}")
    
    print("\n[4/4] LangGraph Agent 准备就绪!")
    print("=" * 60)
    
    # 4. 测试用例
    test_cases = [
        "请帮我计算 25 加 17 等于多少？",        # LLM 决定使用 calculator
        "华东师范大学在哪里？有几个校区？",      # LLM 决定使用 knowledge_search
        "你好，今天天气怎么样？",               # LLM 直接回答
        "123 + 456 = ?",                        # LLM 决定使用 calculator
        "什么是 RAG 技术？",                    # LLM 决定使用 knowledge_search
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {question}")
        print('='*60)
        
        try:
            # 创建初始状态
            initial_state = create_initial_state(question)
            print_state_info_langgraph(initial_state, "初始状态")
            
            # 运行 LangGraph Agent
            final_state = run_langgraph_agent(initial_state, llm, tools_map)
            
            # 打印结果
            print_state_info_langgraph(final_state, "最终状态")
            print(f"\n[结果] 最终答案: {final_state['final_answer']}")
            
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
        
        print('='*60)
    
    print("\n\n所有测试完成!")
    print("\n说明:")
    print("- LangGraph 使用标准 ReAct 模式")
    print("- LLM 通过 tool_calls 自主决定使用哪个工具")
    print("- 支持多轮推理：agent → tools → agent → ...")


# if __name__ == "__main__":
#     main_langgraph()


# =============================================================================
# 文件: view_knowledge_base.py
# 说明: 查看知识库工具
# =============================================================================

"""
查看知识库内容
"""

# import sys
# import io
# 
# # 设置标准输出为 UTF-8
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 
# from src.knowledge_base import get_knowledge_base, get_knowledge_summary


def main_view_kb():
    """主函数"""
    
    print("=" * 70)
    print("知识库内容查看")
    print("=" * 70)
    
    # 获取知识库摘要
    summary = get_knowledge_summary()
    
    print(f"\n📊 知识库统计:")
    print(f"   总文档数: {summary['total_docs']} 条")
    print(f"\n   分类统计:")
    for category, count in summary['categories'].items():
        print(f"      - {category}: {count} 条")
    
    print("\n" + "=" * 70)
    print("详细内容")
    print("=" * 70)
    
    # 获取所有文档
    knowledge_base = get_knowledge_base()
    
    for i, doc in enumerate(knowledge_base, 1):
        print(f"\n【文档 {i}】")
        print(f"ID: {doc['id']}")
        print(f"来源: {doc['metadata']['source']}")
        print(f"分类: {doc['metadata']['category']}")
        print(f"内容: {doc['content']}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("提示:")
    print("- 可以在 src/knowledge_base.py 中修改或添加文档")
    print("- 修改后需要删除 ./data/chroma_db 目录重建向量数据库")
    print("- 或者在 main.py 中设置 force_reload=True")
    print("=" * 70)


# if __name__ == "__main__":
#     main_view_kb()


# =============================================================================
# 使用说明
# =============================================================================

"""
使用说明：

1. 项目结构：
   - 状态驱动架构：所有函数遵循 state -> state 模式
   - 工具系统：calculator（计算）、knowledge_search（知识库搜索）
   - RAG 系统：Chroma 向量数据库 + HuggingFace Embeddings
   - 两种 Agent：原版（while 循环）和 LangGraph（图结构）

2. 核心概念：
   - AgentState：统一的状态结构
   - tool_calls：LLM 决定调用哪个工具
   - ReAct 模式：Reasoning（推理）+ Acting（行动）循环
   - 完全复用：LangGraph 100% 复用原版 Agent 逻辑

3. 运行方式：
   - 原版 Agent：python main.py
   - LangGraph Agent：python main_langgraph.py
   - 查看知识库：python view_knowledge_base.py

4. 关键设计：
   - LLM 驱动：通过 tool_calls 决定流程，不使用关键词匹配
   - 多轮推理：支持 agent ↔ tools 循环
   - 状态纯净：使用 partial 绑定上下文
   - 职责清晰：每个模块职责单一

5. 扩展方向：
   - 添加新工具：在 tools.py 中定义
   - 扩展知识库：在 knowledge_base.py 中添加
   - 添加新节点：在 langgraph_agent.py 中实现
   - 添加 Memory：使用 LangGraph 的 checkpointer

6. 学习建议：
   - 先理解状态驱动架构
   - 再学习 ReAct 模式
   - 最后掌握 LangGraph 用法
   - 重点理解 LLM 如何通过 tool_calls 驱动流程

=============================================================================
代码合集结束
=============================================================================
"""


# =============================================================================
# 文件: app.py
# 说明: FastAPI Web 服务
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI 后端服务
提供 /chat 接口，连接 LangGraph Agent with Memory
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from src.llm import create_llm
from src.tools import get_all_tools, create_tools_map
from src.state import create_initial_state
from src.rag import initialize_rag_system
from src.langgraph_agent import run_langgraph_agent_with_memory


# ============================================================
# 1. FastAPI 应用初始化
# ============================================================

app = FastAPI(title="AI Research Agent API", version="1.0.0")

# 配置 CORS（允许浏览器跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地开发允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. 全局变量（Agent 组件）
# ============================================================

llm = None
tools_map = None


# ============================================================
# 3. 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str  # 用户消息
    thread_id: Optional[str] = "default"  # 会话 ID（用于 Memory）


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str  # Agent 回答
    thread_id: str  # 会话 ID
    message_count: int  # 当前对话历史消息数


# ============================================================
# 4. 启动事件（初始化 Agent）
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    服务启动时初始化 Agent 组件
    """
    global llm, tools_map
    
    print("\n" + "=" * 70)
    print("🚀 正在启动 AI Research Agent 服务...")
    print("=" * 70)
    
    # 初始化 LLM
    print("\n[1/3] 初始化 LLM...")
    llm = create_llm()
    print("      ✅ LLM 初始化完成")
    
    # 初始化 RAG 系统
    print("\n[2/3] 初始化 RAG 系统...")
    initialize_rag_system(force_reload=False)
    
    # 加载工具
    print("\n[3/3] 加载工具...")
    tools = get_all_tools()
    tools_map = create_tools_map(tools)
    print(f"      ✅ 已加载 {len(tools)} 个工具: {', '.join(tools_map.keys())}")
    
    print("\n" + "=" * 70)
    print("✅ AI Research Agent 服务启动成功!")
    print("   访问地址: http://localhost:8000")
    print("=" * 70 + "\n")


# ============================================================
# 5. API 端点
# ============================================================

@app.get("/")
async def root():
    """根路径 - 返回前端页面"""
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    接收用户消息，调用 Agent，返回回答
    支持 Memory（多轮对话）
    """
    try:
        # 验证输入
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        print(f"\n💬 收到消息 [thread_id={request.thread_id}]: {request.message}")
        
        # 创建初始状态
        initial_state = create_initial_state(request.message)
        
        # 调用 Agent（带 Memory）
        final_state = run_langgraph_agent_with_memory(
            initial_state,
            llm,
            tools_map,
            thread_id=request.thread_id
        )
        
        # 提取答案
        answer = final_state.get('final_answer', '抱歉，我无法回答这个问题')
        message_count = len(final_state.get('messages', []))
        
        print(f"✅ 回答生成完成 [消息数={message_count}]")
        
        return ChatResponse(
            answer=answer,
            thread_id=request.thread_id,
            message_count=message_count
        )
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "message": "AI Research Agent is running"}


# ============================================================
# 6. 静态文件服务（前端）
# ============================================================

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# 7. 主函数
# ============================================================

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # 生产环境关闭自动重载
    )


# =============================================================================
# 文件: static/index.html
# 说明: Web 聊天界面（Vue.js 3）
# =============================================================================

"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Research Agent - 聊天界面</title>
    <script src="https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        #app {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f7f7f7;
        }

        .message {
            margin-bottom: 16px;
            display: flex;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 12px;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: #667eea;
            color: white;
        }

        .message.agent .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
        }

        .message-label {
            font-size: 12px;
            margin-bottom: 4px;
            opacity: 0.7;
        }

        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }

        .input-box {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 24px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }

        .input-box:focus {
            border-color: #667eea;
        }

        .send-button {
            padding: 12px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 14px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .send-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #999;
        }

        .new-chat-button {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .new-chat-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .info {
            font-size: 12px;
            color: #999;
            text-align: center;
            padding: 10px;
        }
    </style>
</head>
<body>
    <div id="app">
        <div class="header">
            <h1>🤖 AI Research Agent</h1>
            <p>支持工具调用 · RAG 知识库 · 多轮对话记忆</p>
            <button class="new-chat-button" @click="startNewChat">🆕 新对话</button>
        </div>

        <div class="chat-container" ref="chatContainer">
            <div v-if="messages.length === 0" class="info">
                👋 你好！我是 AI Research Agent，可以帮你计算、搜索知识库。开始聊天吧！
            </div>
            
            <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.role]">
                <div>
                    <div class="message-label">{{ msg.role === 'user' ? '👤 你' : '🤖 Agent' }}</div>
                    <div class="message-content">{{ msg.content }}</div>
                </div>
            </div>

            <div v-if="loading" class="loading">
                ⏳ Agent 正在思考...
            </div>
        </div>

        <div class="input-container">
            <input 
                v-model="userInput" 
                @keyup.enter="sendMessage"
                :disabled="loading"
                class="input-box" 
                type="text" 
                placeholder="输入你的问题..."
            />
            <button 
                @click="sendMessage" 
                :disabled="loading || !userInput.trim()"
                class="send-button"
            >
                {{ loading ? '发送中...' : '发送' }}
            </button>
        </div>

        <div class="info">
            会话 ID: {{ threadId }} | 消息数: {{ messageCount }}
        </div>
    </div>

    <script>
        const { createApp } = Vue;

        createApp({
            data() {
                return {
                    messages: [],
                    userInput: '',
                    loading: false,
                    threadId: this.generateThreadId(),
                    messageCount: 0
                }
            },
            methods: {
                generateThreadId() {
                    return 'web_' + Date.now();
                },

                startNewChat() {
                    if (confirm('确定要开始新对话吗？当前对话历史将被清空。')) {
                        this.messages = [];
                        this.threadId = this.generateThreadId();
                        this.messageCount = 0;
                    }
                },

                async sendMessage() {
                    const message = this.userInput.trim();
                    if (!message || this.loading) return;

                    // 添加用户消息到界面
                    this.messages.push({
                        role: 'user',
                        content: message
                    });

                    // 清空输入框
                    this.userInput = '';
                    this.loading = true;

                    // 滚动到底部
                    this.$nextTick(() => {
                        this.scrollToBottom();
                    });

                    try {
                        // 调用后端 API
                        const response = await fetch('http://localhost:8000/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                message: message,
                                thread_id: this.threadId
                            })
                        });

                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }

                        const data = await response.json();

                        // 添加 Agent 回答到界面
                        this.messages.push({
                            role: 'agent',
                            content: data.answer
                        });

                        // 更新消息计数
                        this.messageCount = data.message_count;

                        // 滚动到底部
                        this.$nextTick(() => {
                            this.scrollToBottom();
                        });

                    } catch (error) {
                        console.error('Error:', error);
                        this.messages.push({
                            role: 'agent',
                            content: '❌ 抱歉，发生了错误: ' + error.message
                        });
                    } finally {
                        this.loading = false;
                    }
                },

                scrollToBottom() {
                    const container = this.$refs.chatContainer;
                    container.scrollTop = container.scrollHeight;
                }
            }
        }).mount('#app');
    </script>
</body>
</html>
"""
