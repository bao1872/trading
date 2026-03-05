# 项目结构说明

## 📁 目录结构

```
交易/
├── app/                      # 应用脚本目录 (新建脚本存放位置)
│   ├── __init__.py
│   ├── db.py                # 数据库操作
│   ├── models.py            # 数据模型
│   ├── logger.py            # 日志配置
│   ├── feishu_notifier.py   # 飞书通知
│   ├── stock_list_manager.py # 股票列表管理
│   ├── k_data_loader.py     # K 线数据加载
│   ├── k_data_saver.py      # K 线数据保存
│   ├── amp_scanner.py       # AMP 扫描器
│   ├── divergence_scanner.py # 背离扫描器
│   ├── quote_snapshot.py    # 行情快照
│   ├── trend_script.py      # 趋势脚本
│   ├── space_script.py      # 空间脚本
│   ├── trigger_script.py    # 触发脚本
│   └── migrations/          # 数据库迁移脚本
│       └── add_name_to_amp_features.py
│
├── cores/                    # 核心算法/可视化模块
│   ├── __init__.py
│   ├── amp_plotly.py        # AMP 可视化
│   ├── divergence_many_plotly.py  # 背离检测可视化
│   ├── volume_zscore_plotly.py    # 成交量 ZScore
│   ├── bollinger_features_plotly.py # 布林带特征
│   ├── dynamic_swing_anchored_vwap.py # 动态摆动锚定 VWAP
│   ├── labels_mfe_mae_score_h3_5m_plotly.py # MFE/MAE 标签
│   ├── time_features_5m_plotly.py # 5m 时间特征
│   ├── liquidity_zones_plotly.py # 流动性区域
│   └── smc_probability_expo_pytdx_v2.py # SMC 概率指标
│
├── datasource/               # 数据源模块 (统一数据获取入口)
│   ├── __init__.py
│   └── pytdx_client.py      # pytdx 统一封装
│
├── doc/                      # 文档目录 (所有文档存放位置)
│   ├── PROJECT_STRUCTURE.md  # 项目结构说明
│   ├── database_schema.md    # 数据库表结构
│   ├── scheduler.md          # 调度器说明
│   ├── feishu_notifier_guide.md # 飞书通知指南
│   ├── code_quality_check.md # 代码质量检查
│   ├── task_authorization.md # 任务授权
│   ├── git_config.md         # Git 配置
│   ├── stock_concepts_cache.md # 股票概念缓存
│   ├── PRD.md               # 产品需求文档
│   ├── MIGRATION_SUMMARY.md # 迁移总结
│   └── STOCK_LIST_MIGRATION_REPORT.md # 股票列表迁移报告
│
├── output/                   # 输出目录 (测试文件和临时输出)
│   ├── *.html               # 生成的 HTML 图表
│   ├── *.png                # 生成的图片
│   └── *.log                # 日志文件
│
├── .trae/                    # IDE 配置目录
│   └── rules/               # 项目规则
│       ├── directory.md     # 目录结构规则
│       ├── style.md         # 编码风格规则
│       ├── consistency.md   # 一致性规则
│       ├── scripts-database.md # 脚本数据库规则
│       ├── dispatch.md      # 调度规则
│       ├── datasource.md    # 数据源规则
│       └── caculation.md    # 计算规则
│
├── .env.example             # 环境变量示例
├── .gitignore               # Git 忽略配置
├── requirements.txt         # Python 依赖
└── scheduler.py             # 调度器入口
```

## 📋 目录用途说明

### `app/` - 应用脚本目录
- **用途**: 存放所有新建的业务脚本
- **包含**: 数据库操作、数据加载、扫描器、通知等模块
- **规范**: 新建脚本必须遵循 `scripts-database.md` 规则

### `cores/` - 核心算法模块
- **用途**: 核心算法实现和可视化
- **包含**: AMP、背离检测、布林带、VWAP 等算法
- **规范**: 遵循 `consistency.md` 和 `caculation.md` 规则

### `datasource/` - 数据源模块
- **用途**: 统一数据获取入口
- **包含**: pytdx 客户端封装
- **规范**: 所有数据获取必须通过此模块，遵循 `datasource.md` 规则

### `doc/` - 文档目录
- **用途**: 所有项目文档
- **包含**: 设计文档、操作手册、API 文档等
- **规范**: 除代码注释外的所有文档

### `output/` - 输出目录
- **用途**: 测试文件和临时输出
- **包含**: HTML 图表、PNG 图片、日志文件等
- **规范**: 临时文件，可定期清理

## 🔧 开发规范

### 1. 新建脚本
- 位置：`app/` 目录
- 必须包含模块 docstring (Purpose/Inputs/Outputs/How to Run/Examples/Side Effects)
- 必须包含 `if __name__ == "__main__":` 入口
- 涉及数据库变更需更新 `doc/database_schema.md`

### 2. 数据获取
- 必须使用 `datasource.pytdx_client` 模块
- 禁止直接使用 pytdx
- 遵循 SSOT 原则 (单一事实来源)

### 3. 核心算法
- 只能写一份实现，禁止复制
- 其他脚本只能引用调用
- 遵循 DRY 原则 (Don't Repeat Yourself)

### 4. 文档更新
- 所有文档放在 `doc/` 目录
- 代码注释写"为什么/约束/边界"
- 公共函数/类给一句话 docstring

## 📖 重要文档索引

| 文档 | 说明 |
|------|------|
| [database_schema.md](database_schema.md) | 数据库表结构说明 |
| [scheduler.md](scheduler.md) | 调度器使用说明 |
| [feishu_notifier_guide.md](feishu_notifier_guide.md) | 飞书通知使用指南 |
| [code_quality_check.md](code_quality_check.md) | 代码质量检查标准 |
| [task_authorization.md](task_authorization.md) | 任务授权流程 |
| [git_config.md](git_config.md) | Git 配置说明 |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入必要配置

# 3. 运行调度器
python scheduler.py

# 4. 运行单个脚本
PYTHONPATH=/path/to/交易:$PYTHONPATH python app/amp_scanner.py --help
```

## 📝 更新日志

- 2026-03-05: 统一 pytdx 数据获取模块，删除重复代码
- 2026-03-05: 整理项目文档结构，统一放到 `doc/` 目录
- 2026-03-05: 创建 `output/` 目录存放临时输出文件
