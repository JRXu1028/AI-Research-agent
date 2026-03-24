"""
LangGraph Agent 模块
基于 LangGraph 实现的状态驱动 Agent（标准 ReAct 模式）
"""

from langgraph.graph import StateGraph, END
from functools import partial
from .state import AgentState
from .agent import call_model, execute_tools


# ============================================================
# 1. 节点函数（完全复用现有逻辑）
# ============================================================

def agent_node(state: AgentState, llm, tools_map) -> AgentState:
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


def tool_node(state: AgentState, tools_map) -> AgentState:
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

def should_continue(state: AgentState) -> str:
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
    
    # 默认继续（理论上不应该到这里）
    print(f"🔀 [Router] 继续推理")
    return "continue"


# ============================================================
# 3. LangGraph 构建（标准 ReAct 模式）
# ============================================================

def create_langgraph_agent(llm, tools_map):
    """
    创建 LangGraph Agent（标准 ReAct 模式）
    
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
    """
    # 创建状态图
    workflow = StateGraph(AgentState) # AgentState 定义了这个图中“状态”的数据结构
    
    # 使用 partial 绑定上下文，避免污染状态

    # 使用 functools.partial 对 agent_node 进行“参数预绑定”
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
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边：agent → tools 或 END
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
            "continue": "agent"  # 支持继续推理
        }
    )
    
    # 添加边：tools → agent（形成 ReAct 循环）
    workflow.add_edge("tools", "agent")
    
    # 编译图
    app = workflow.compile()
    
    return app


# ============================================================
# 4. 便捷函数
# ============================================================

def run_langgraph_agent(state: AgentState, llm, tools_map: dict) -> AgentState:
    """
    运行 LangGraph Agent
    
    Args:
        state: 初始状态
        llm: 语言模型
        tools_map: 工具映射字典
    
    Returns:
        最终状态
    """
    # 创建 LangGraph Agent
    agent = create_langgraph_agent(llm, tools_map)
    
    # 运行
    final_state = agent.invoke(state)
    
    return final_state
