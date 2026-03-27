"""
PostgreSQL + pgvector 向量数据库模块
替代 Chroma，支持生产环境部署
"""

from langchain_postgres import PGVector
from langchain_core.documents import Document
from .embeddings import create_embeddings
from .knowledge_base import get_knowledge_base
from .config import Config


class VectorStorePG:
    """PostgreSQL 向量数据库管理类"""
    
    def __init__(self):
        """初始化向量数据库"""
        self.embeddings = create_embeddings()
        self.vectorstore = None
        self.connection_string = Config.get_postgres_connection_string()
        self.collection_name = "knowledge_base"
        
    def initialize(self, force_reload=False):
        """
        初始化或加载向量数据库
        
        Args:
            force_reload: 是否强制重新加载数据
        """
        # 创建 PGVector 实例
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
        
        # 检查是否需要初始化数据
        try:
            # 尝试查询，看是否有数据
            test_results = self.vectorstore.similarity_search("test", k=1)
            doc_count = len(test_results) if test_results else 0
            
            if doc_count == 0 or force_reload:
                print("      🔨 初始化知识库数据...")
                self._load_knowledge_base()
            else:
                print(f"      📂 加载已有向量数据库...")
                print(f"      ✅ 数据库已就绪")
                
        except Exception as e:
            print(f"      🔨 首次初始化数据库...")
            self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库数据到数据库"""
        # 如果强制重新加载，先清空集合
        try:
            self.vectorstore.delete_collection()
        except:
            pass
        
        # 重新创建
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
        
        # 加载知识库数据
        knowledge_base = get_knowledge_base()
        
        # 转换为 LangChain Document 格式
        documents = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
            for doc in knowledge_base
        ]
        
        # 添加文档
        self.vectorstore.add_documents(documents)
        print(f"      ✅ 已加载 {len(documents)} 条文档到数据库")
    
    def search(self, query: str, k: int = 3):
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_score(self, query: str, k: int = 3):
        """
        检索相关文档（带相似度分数）
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            (文档, 分数) 的列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def add_documents(self, documents):
        """
        添加新文档到向量数据库
        
        Args:
            documents: Document 对象列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先调用 initialize()")
        
        self.vectorstore.add_documents(documents)
        print(f"      ✅ 已添加 {len(documents)} 条新文档")


def create_vector_store(force_reload=False):
    """
    创建并初始化向量数据库
    
    Args:
        force_reload: 是否强制重新加载数据
    
    Returns:
        VectorStorePG 实例
    """
    store = VectorStorePG()
    store.initialize(force_reload=force_reload)
    return store
