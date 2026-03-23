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
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.ECNU_API_KEY:
            raise ValueError(
                "请设置环境变量 ECNU_API_KEY\n"
                "1. 复制 .env.example 为 .env\n"
                "2. 在 .env 中填入你的 API Key"
            )
