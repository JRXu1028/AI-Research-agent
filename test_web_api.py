#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Web API
验证 /chat 接口是否正常工作
"""

import requests
import json


def test_chat_api():
    """测试聊天 API"""
    
    print("=" * 70)
    print("🧪 测试 AI Research Agent Web API")
    print("=" * 70)
    
    # API 地址
    url = "http://localhost:8000/chat"
    
    # 测试用例
    test_cases = [
        {
            "message": "请帮我计算 25 加 17",
            "thread_id": "test_001"
        },
        {
            "message": "再加上 10 呢？",
            "thread_id": "test_001"  # 同一个 thread_id，测试 Memory
        },
        {
            "message": "华东师范大学在哪里？",
            "thread_id": "test_002"  # 新的 thread_id
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'=' * 70}")
        print(f"测试 {i}/{len(test_cases)}")
        print(f"{'=' * 70}")
        print(f"👤 用户: {test_case['message']}")
        print(f"🔖 Thread ID: {test_case['thread_id']}")
        
        try:
            # 发送请求
            response = requests.post(
                url,
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            # 检查响应
            if response.status_code == 200:
                data = response.json()
                print(f"\n🤖 Agent: {data['answer']}")
                print(f"\n📊 消息数: {data['message_count']}")
                print(f"✅ 测试通过")
            else:
                print(f"\n❌ 错误: HTTP {response.status_code}")
                print(f"   {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("\n❌ 连接失败: 请确保 Web 服务已启动")
            print("   运行命令: python app.py")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


def test_health_api():
    """测试健康检查 API"""
    
    print("\n\n🏥 测试健康检查接口...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data['status']}")
            print(f"   消息: {data['message']}")
        else:
            print(f"❌ 错误: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    # 测试健康检查
    test_health_api()
    
    # 测试聊天 API
    test_chat_api()
