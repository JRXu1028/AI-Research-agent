#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序入口
运行 AI Research Agent
"""

import sys
import io

# 设置标准输出为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.llm import create_llm, bind_tools_to_llm
from src.tools import get_all_tools, create_tools_map
from src.state import create_initial_state
from src.agent import run_agent


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
    
    # 2. 获取工具
    print("\n[2/4] 加载工具...")
    tools = get_all_tools()
    tools_map = create_tools_map(tools)
    print(f"      已加载 {len(tools)} 个工具: {', '.join(tools_map.keys())}")
    
    # 3. 绑定工具到 LLM
    print("\n[3/4] 绑定工具到 LLM...")
    llm_with_tools = bind_tools_to_llm(llm, tools)
    print("      工具绑定完成")
    
    print("\n[4/4] Agent 准备就绪!")
    print("=" * 60)
    
    # 4. 测试用例
    test_cases = [
        "请帮我计算 25 加 17 等于多少？",
        "你好，今天天气怎么样？",
        "123 + 456 = ?",
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


if __name__ == "__main__":
    main()
