"""
配置模块
管理环境变量和配置项
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置类"""
    
    # ECNU API 配置
    ECNU_API_KEY = os.getenv("ECNU_API_KEY")
    ECNU_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
    ECNU_MODEL = "ecnu-plus"
    
    # LLM 配置
    TEMPERATURE = 0  # 0 = 更确定的输出
    
    # PostgreSQL 配置
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "ai_research_agent")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    
    # 向量数据库类型：chroma 或 postgres
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    
    @classmethod
    def get_postgres_connection_string(cls):
        """获取 PostgreSQL 连接字符串"""
        return (
            f"postgresql+psycopg://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.ECNU_API_KEY:
            raise ValueError(
                "请设置环境变量 ECNU_API_KEY\n"
                "1. 复制 .env.example 为 .env\n"
                "2. 在 .env 中填入你的 API Key"
            )
        
        if cls.VECTOR_STORE_TYPE == "postgres" and not cls.POSTGRES_PASSWORD:
            raise ValueError(
                "使用 PostgreSQL 时请设置 POSTGRES_PASSWORD\n"
                "在 .env 中添加: POSTGRES_PASSWORD=your_password"
            )
