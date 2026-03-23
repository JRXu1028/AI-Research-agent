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


# 可以在这里添加更多工具
# @tool
# def search(query: str) -> str:
#     """搜索工具"""
#     return f"搜索结果: {query}"


def get_all_tools():
    """
    获取所有可用工具
    
    Returns:
        工具列表
    """
    return [calculator]


def create_tools_map(tools: list) -> dict:
    """
    创建工具名称到工具对象的映射
    
    Args:
        tools: 工具列表
    
    Returns:
        {工具名称: 工具对象} 的字典
    """
    return {tool.name: tool for tool in tools}
