#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph Agent 演示
展示如何使用 LangGraph 构建 Agent
"""

import sys
import io

# 设置标准输出为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.llm import create_llm
from src.tools import get_all_tools, create_tools_map
from src.state import create_initial_state
from src.rag import initialize_rag_system
from src.langgraph_agent import run_langgraph_agent


def print_state_info(state, label):
    """打印状态信息（用于调试）"""
    print(f"\n[状态] {label}:")
    print(f"   - messages: {len(state['messages'])} 条")
    if state.get('final_answer'):
        answer_preview = state['final_answer'][:50] + "..." if len(state['final_answer']) > 50 else state['final_answer']
        print(f"   - final_answer: {answer_preview}")
    else:
        print(f"   - final_answer: {state.get('final_answer')}")


def main():
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
    tools = get_all_tools()  # 工具函数列表
    tools_map = create_tools_map(tools)  # key:工具名称  value：函数
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
            # create_initial_state(user_input: str) -> AgentState 的返回：
            # return {
            #     "messages": [HumanMessage(content=user_input)],
            #     "tool_calls": None,
            #     "final_answer": None,
            #     "error": None,
            # }
            initial_state = create_initial_state(question)
            print_state_info(initial_state, "初始状态")
            
            # 运行 LangGraph Agent
            final_state = run_langgraph_agent(initial_state, llm, tools_map)
            
            # 打印结果
            print_state_info(final_state, "最终状态")
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


if __name__ == "__main__":
    main()
