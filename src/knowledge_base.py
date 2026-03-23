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
