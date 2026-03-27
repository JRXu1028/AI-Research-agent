"""
RAG 模块
实现检索增强生成
支持 Chroma 和 PostgreSQL 两种向量数据库
"""

from .config import Config

# 全局 RAG 系统实例
_rag_system_instance = None


class RAGSystem:
    """RAG 系统类"""
    
    def __init__(self, vector_store):
        """
        初始化 RAG 系统
        
        Args:
            vector_store: 向量数据库实例
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
    根据配置选择 Chroma 或 PostgreSQL
    
    Args:
        force_reload: 是否强制重新加载向量数据库
    
    Returns:
        RAGSystem 实例
    """
    global _rag_system_instance
    
    if _rag_system_instance is None or force_reload:
        print("   📚 初始化知识库...")
        
        # 根据配置选择向量数据库类型
        vector_store_type = Config.VECTOR_STORE_TYPE
        print(f"   📦 使用向量数据库: {vector_store_type}")
        
        if vector_store_type == "postgres":
            from .vector_store_pg import create_vector_store
        else:
            from .vector_store import create_vector_store
        
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
