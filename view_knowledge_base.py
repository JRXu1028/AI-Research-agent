#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看知识库内容
"""

import sys
import io

# 设置标准输出为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.knowledge_base import get_knowledge_base, get_knowledge_summary


def main():
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


if __name__ == "__main__":
    main()
