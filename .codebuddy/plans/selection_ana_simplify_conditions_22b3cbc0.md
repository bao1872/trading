---
name: selection_ana_simplify_conditions
overview: 简化选股条件：只保留V型形态，DSA dir和收盘价vs VWAP偏差改为观察项
todos:
  - id: update-detect-bsm
    content: 重构 detect_bsm_signals：仅保留V型形态检测，周线DSA dir/vwap/vwap_deviation作为观察项返回，删除突破买点和下轨条件
    status: completed
  - id: update-process-stock
    content: 重构 process_stock_pavp：移除成交额过滤，选股条件改为仅V型，新增周线DSA观察字段
    status: completed
    dependencies:
      - update-detect-bsm
  - id: update-db-schema
    content: 扩展DB表结构：新增 weekly_dsa_dir/weekly_dsa_vwap/weekly_vwap_deviation/daily_dsa_dir/daily_dsa_vwap 五列
    status: completed
  - id: update-save-and-print
    content: 更新 save_to_database_by_weekly_date 持久化新字段，更新 select_high_margin_stocks 打印信息
    status: completed
    dependencies:
      - update-process-stock
      - update-db-schema
  - id: update-comments-and-cleanup
    content: 更新文件头部注释，删除废弃函数 check_weekly_not_below_lower/cross_over/check_volume_filter
    status: completed
    dependencies:
      - update-detect-bsm
  - id: test-verify
    content: 运行 selection_ana.py 2026-05-06 验证V型选股结果和DB写入正确性
    status: completed
    dependencies:
      - update-save-and-print
      - update-comments-and-cleanup
---

## 用户需求

修改 `selection/selection_ana.py` 的选股逻辑：

- **选股条件**：周线BBMACD的V型形态（t_2 > t_1 and t > t_1）+ 过去5天平均成交额>1亿
- **删除的选股条件**：DSA dir=1检查、收盘价<=VWAP*1.10检查、周线bbmacd不在下轨以下（必要条件）、周线突破买点（上穿上轨+DSA dir=-1+收盘价>=VWAP*0.85）
- **保留的选股条件**：成交额过滤（5日平均>1亿）
- **观察项**：DSA dir和收盘价与VWAP的偏差（日线+周线），记录但不参与筛选
- DB表结构需新增字段持久化观察项
- 前端 `financial_score_app.py` 读同一张表，需确认兼容性

## 修改文件

`/root/trading/selection/selection_ana.py` — 唯一需改动的脚本（前端只读表，无需改动）

## 改动详情

### 1. 文件头部注释（L1-57）

更新选股条件描述：核心条件改为"周线V型形态"，删除旧条件描述，新增观察项说明（weekly_dsa_dir, weekly_vwap_deviation, daily_dsa_dir, daily_dsa_vwap）

### 2. `detect_bsm_signals` 函数（L820-901）

**当前逻辑**：获取周线BBMACD → 检查"不在下轨以下"必要条件 → 计算周线DSA → V型+DSA dir=1+VWAP阈值 → 周线突破+DSA dir=-1+VWAP阈值 → 成交额过滤
**改为**：获取周线BBMACD → V型形态检测 → 计算周线DSA dir/vwap作为观察项返回 → 不再检查下轨条件/DSA dir/VWAP阈值

- 返回dict新增 `weekly_dsa_dir`, `weekly_dsa_vwap`, `weekly_vwap_deviation`
- `weekly_reversal_buy` 改为仅V型形态触发
- `weekly_breakout_buy` 逻辑整体删除（字段保留为始终False）
- 删除不再使用的 `check_weekly_not_below_lower` 函数和 `cross_over` 函数

### 3. `process_stock_pavp` 函数（L673-817）

- 保留 `check_volume_filter` 成交额过滤调用（L719），成交额>1亿条件继续生效
- 选股条件从 `(weekly_reversal_buy or weekly_breakout_buy)` 改为仅V型形态 + 成交额过滤
- 返回dict新增 `weekly_dsa_dir`, `weekly_dsa_vwap`, `weekly_vwap_deviation`（来自detect_bsm_signals）
- `daily_dsa_dir` 和 `daily_dsa_vwap` 已在返回dict中但未持久化到DB，此次一并持久化

### 4. `select_high_margin_stocks` 函数（L1167-1262）

- 更新打印信息：核心条件改为"周线V型形态 + 成交额过滤"

### 5. DB表结构 — `ensure_table_exists` 函数（L904-1012）

新增ALTER TABLE列（IF NOT EXISTS）：

- `weekly_dsa_dir INT` — 周线DSA方向
- `weekly_dsa_vwap FLOAT` — 周线DSA VWAP值
- `weekly_vwap_deviation FLOAT` — 周线收盘价与VWAP偏差率
- `daily_dsa_dir INT` — 日线DSA方向（已计算但未持久化）
- `daily_dsa_vwap FLOAT` — 日线DSA VWAP值（已计算但未持久化）

### 6. `save_to_database_by_weekly_date` 函数（L1071-1164）

record dict新增上述5个字段的写入

### 7. 删除不再使用的函数

- `check_weekly_not_below_lower`（L410-429）— 原必要条件，已删除
- `cross_over`（L535-537）— 原周线突破用，已删除
- `check_volume_filter` 保留（成交额过滤条件继续使用）

## 影响范围

- `financial_score_app.py`：只读 `stock_selection_results` 表，新增列为可选展示，不影响现有展示
- `selection_stop.py`：有自己的 `check_volume_filter` 副本，不受影响
- DB表：只新增列，不改/不删现有列，向后兼容