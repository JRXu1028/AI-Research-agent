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
