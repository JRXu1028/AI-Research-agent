# 部署指南

## 部署步骤

### 1. 安装 PostgreSQL 和 pgvector

**Ubuntu/Debian:**
```bash
# 安装 PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# 安装 pgvector 扩展
sudo apt install postgresql-15-pgvector
```

**使用 Docker:**
```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=ai_research_agent \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. 创建数据库和启用扩展

```bash
# 连接到 PostgreSQL
psql -U postgres

# 创建数据库
CREATE DATABASE ai_research_agent;

# 连接到数据库
\c ai_research_agent

# 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

# 退出
\q
```

### 3. 配置环境变量

编辑 `.env` 文件：
```bash
# ECNU API
ECNU_API_KEY=your_api_key_here

# 向量数据库类型
VECTOR_STORE_TYPE=postgres

# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ai_research_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 构建前端

```bash
cd frontend
npm install
npm run build
```

### 6. 启动服务

```bash
python app.py
```

## 生产环境建议

1. **使用进程管理器**（如 systemd 或 PM2）
2. **使用 Nginx 反向代理**
3. **配置 HTTPS**
4. **限制 CORS 来源**
5. **使用云数据库**（如 AWS RDS、Supabase）

## 云数据库选项

- **Supabase**: 免费套餐，自带 pgvector
- **AWS RDS**: 需要手动安装 pgvector
- **Google Cloud SQL**: 支持 pgvector
