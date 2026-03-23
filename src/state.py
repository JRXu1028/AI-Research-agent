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
