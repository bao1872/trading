# Git 配置说明

## .gitignore 已配置

本项目已配置 `.gitignore` 文件，以下文件不会被提交到 Git 仓库：

### 忽略的文件类型

#### 1. 股票数据文件（重要）
- ✅ `stock.xlsx` - 股票列表文件（包含个人选股）
- ✅ `stock_concepts_cache.xlsx` - 股票概念缓存文件

#### 2. 配置文件
- ✅ `.env` - 环境变量配置文件（可能包含敏感信息）
- ✅ `.env.local` - 本地配置
- ✅ `*.local` - 本地配置文件

#### 3. 日志文件
- ✅ `*.log` - 所有日志文件
- ✅ `scheduler.log` - 调度器日志

#### 4. 临时输出
- ✅ `背离扫描/` - 背离扫描输出目录
- ✅ `复盘/` - 复盘输出目录
- ✅ `*.html` - 生成的 HTML 文件
- ✅ `*.png` - 生成的图片文件

#### 5. Python 缓存
- ✅ `__pycache__/`
- ✅ `*.pyc`
- ✅ `*.egg-info/`

#### 6. 虚拟环境
- ✅ `venv/`
- ✅ `env/`

#### 7. IDE 配置
- ✅ `.idea/`
- ✅ `.vscode/`
- ✅ `.DS_Store`

#### 8. 数据库文件
- ✅ `*.db`
- ✅ `*.sqlite`

## 首次使用配置

### 1. 复制配置文件模板

```bash
# 复制环境变量配置
cp .env.example .env

# 编辑配置
vim .env
```

### 2. 准备股票数据文件

需要自行准备以下文件（不会被 Git 跟踪）：

- `stock.xlsx` - 股票列表文件
- `stock_concepts_cache.xlsx` - 股票概念缓存文件

这些文件需要您根据自己的需求创建或从其他来源获取。

### 3. 验证 Git 状态

```bash
# 查看 Git 状态
git status

# 应该看到以下文件被忽略：
#   stock.xlsx
#   stock_concepts_cache.xlsx
#   .env
#   scheduler.log
#   背离扫描/
#   复盘/
```

## 已忽略文件清单

以下文件**不会**被提交到 Git 仓库：

```
stock.xlsx                          # 股票列表（个人配置）
stock_concepts_cache.xlsx           # 概念缓存（自动生成）
.env                                # 环境变量（敏感信息）
.env.local                          # 本地配置
scheduler.log                       # 调度器日志
*.log                              # 所有日志
背离扫描/                           # 背离扫描输出
复盘/                              # 复盘输出
*.html                             # 生成的 HTML
*.png                              # 生成的图片
__pycache__/                       # Python 缓存
*.pyc                              # Python 编译文件
venv/                              # 虚拟环境
.idea/                             # PyCharm 配置
.vscode/                           # VSCode 配置
.DS_Store                          # macOS 系统文件
```

## 需要提交的文件

以下文件**应该**被提交到 Git 仓库：

```
scheduler.py                        # 调度器主程序
app/                                # 应用代码
cores/                              # 核心模块
datasource/                         # 数据源模块
.env.example                        # 配置模板
requirements.txt                    # Python 依赖
.gitignore                          # Git 忽略配置
README.md                           # 项目说明
```

## 常见问题

### Q: 如果不小心提交了 stock.xlsx 怎么办？

A: 使用以下命令从 Git 历史中删除：

```bash
# 从 Git 仓库删除（保留本地文件）
git rm --cached stock.xlsx
git rm --cached stock_concepts_cache.xlsx

# 提交更改
git commit -m "移除敏感配置文件"

# 推送到远程仓库
git push
```

### Q: 如何确认哪些文件被忽略了？

A: 使用以下命令查看：

```bash
# 查看被忽略的文件
git status --ignored

# 或查看特定文件是否被忽略
git check-ignore -v stock.xlsx
```

### Q: 如何共享我的配置？

A: 编辑 `.env.example` 文件，提供配置模板（不包含真实值）：

```bash
# .env.example
ENABLE_DIVERGENCE_SCAN=true
ENABLE_AMP_SCAN=true
LOG_LEVEL=INFO

# 不要包含真实的 webhook URL 或密钥
# FEISHU_WEBHOOK_URL=请替换为你的 webhook 地址
```

## 安全提示

⚠️ **重要**：
- 不要将包含真实 webhook URL、API 密钥的文件提交到 Git
- 定期检查 `.gitignore` 确保敏感文件被正确忽略
- 如果怀疑敏感信息已泄露，立即更新相关配置并通知相关人员

## 更新 .gitignore

如果需要忽略更多文件类型，编辑 `.gitignore` 文件：

```bash
vim .gitignore

# 添加新的忽略规则
# 例如：忽略所有 Excel 文件
# *.xlsx
```

然后验证：

```bash
git check-ignore -v 文件名
```
