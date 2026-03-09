# 交易分析系统

一个基于 Python 的股票交易分析系统，提供 AMP、背离检测、布林带、VWAP 等多种技术分析工具。

## 📖 文档索引

**详细文档请查看**: [`doc/PROJECT_STRUCTURE.md`](doc/PROJECT_STRUCTURE.md)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入必要配置
```

### 2. 运行调度器

```bash
python scheduler.py
```

### 3. 运行单个脚本

```bash
# 查看帮助
PYTHONPATH=/path/to/交易:$PYTHONPATH python app/amp_scanner.py --help

# 运行 AMP 扫描
PYTHONPATH=/path/to/交易:$PYTHONPATH python app/amp_scanner.py --symbol 600547 --years 3

# 运行背离检测可视化
PYTHONPATH=/path/to/交易:$PYTHONPATH python features/divergence_many_plotly.py --symbol 600489 --freq 60m --prd 5 --searchdiv Regular --out output/div.html
```

## 📁 项目结构

```
交易/
├── app/           # 应用脚本 (新建脚本放在这里)
├── features/      # 核心算法/可视化
├── datasource/    # 数据源模块
├── doc/           # 所有文档 (查看这里了解详细说明)
├── output/        # 临时输出文件
├── .trae/rules/   # 项目规则
└── scheduler.py   # 调度器入口
```

## 🔧 核心功能

### AMP 扫描
- 检测 AMP 形态
- 生成可视化图表
- 飞书通知

### 背离检测
- 支持多种指标 (MACD, RSI, OBV 等)
- 4 种背离类型 (正/负 Regular/Hidden)
- Plotly HTML 可视化

### 成交量分析
- ZScore 异常检测
- 流动性区域识别
- 锚定 VWAP

### 布林带特征
- 宽度 ZScore
- 股价位置分析
- 特征表格展示

## 📋 开发规范

### 新建脚本
1. 放在 `app/` 目录
2. 包含模块 docstring (Purpose/Inputs/Outputs/How to Run/Examples/Side Effects)
3. 包含 `if __name__ == "__main__":` 入口
4. 涉及数据库变更需更新 `doc/database_schema.md`

### 数据获取
- 必须使用 `datasource.pytdx_client` 模块
- 禁止直接使用 pytdx
- 遵循 SSOT 原则 (单一事实来源)

### 核心算法
- 只能写一份实现，禁止复制
- 其他脚本只能引用调用
- 遵循 DRY 原则 (Don't Repeat Yourself)

### 文档
- 所有文档放在 `doc/` 目录
- 代码注释写"为什么/约束/边界"
- 公共函数/类给一句话 docstring

## 📚 重要文档

| 文档 | 说明 |
|------|------|
| [doc/PROJECT_STRUCTURE.md](doc/PROJECT_STRUCTURE.md) | 项目结构详细说明 |
| [doc/database_schema.md](doc/database_schema.md) | 数据库表结构 |
| [doc/scheduler.md](doc/scheduler.md) | 调度器使用说明 |
| [doc/feishu_notifier_guide.md](doc/feishu_notifier_guide.md) | 飞书通知使用指南 |
| [doc/code_quality_check.md](doc/code_quality_check.md) | 代码质量检查标准 |

## ⚙️ 项目规则

项目规则位于 `.trae/rules/` 目录，所有开发必须遵守:

- `directory.md` - 目录结构规则
- `style.md` - 编码风格规则
- `consistency.md` - 一致性规则 (SSOT/DRY)
- `scripts-database.md` - 脚本数据库规则
- `dispatch.md` - 调度规则
- `datasource.md` - 数据源规则
- `caculation.md` - 计算规则

## 📝 更新日志

### 2026-03-05
- ✅ 统一 pytdx 数据获取模块，删除约 500 行重复代码
- ✅ 整理项目文档结构，统一放到 `doc/` 目录
- ✅ 创建 `output/` 目录存放临时输出文件
- ✅ 更新项目规则，明确各目录用途

### 更早版本
- 实现 AMP 扫描和可视化
- 实现背离检测 (多指标)
- 实现成交量 ZScore 分析
- 实现布林带特征分析
- 实现动态摆动锚定 VWAP
- 实现 MFE/MAE 标签系统
- 实现流动性区域检测
- 实现 SMC 概率指标

## 🤝 贡献

请遵守项目规则，确保代码质量和一致性。

## 📄 License

[待添加]
