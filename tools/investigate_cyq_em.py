#!/usr/bin/env python3
"""
akshare 筹码分布接口（stock_cyq_em）全面调研脚本

Purpose:
    对 akshare 的 stock_cyq_em（东方财富筹码分布）接口进行全面调研，
    包括接口可用性验证、字段梳理、数据校验、边界场景测试、性能评估。

Inputs:
    akshare 库（v1.18.55），网络连接（东方财富 API）

Outputs:
    终端输出结构化调研报告

How to Run:
    python tools/investigate_cyq_em.py

Examples:
    python tools/investigate_cyq_em.py
    python tools/investigate_cyq_em.py 2>&1 | tee tools/cyq_em_report.log

Side Effects:
    无。仅读取 akshare 接口数据，不写入数据库或文件。
"""

import time
import logging

import akshare as ak
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 调用间隔（秒），避免触发东方财富 API 频率限制 ──
CALL_INTERVAL = 3.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0


def call_cyq_em(symbol: str, adjust: str = "") -> pd.DataFrame:
    """带重试和延时的 stock_cyq_em 调用包装"""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.stock_cyq_em(symbol=symbol, adjust=adjust)
            time.sleep(CALL_INTERVAL)
            return df
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning(f"  调用失败({attempt}/{MAX_RETRIES})，{wait:.0f}s 后重试: {e}")
                time.sleep(wait)
    raise last_err


# ── 测试股票样本 ──────────────────────────────────────────────
TEST_SYMBOLS = {
    "000001": "平安银行（深市主板）",
    "600519": "贵州茅台（沪市主板）",
    "300750": "宁德时代（创业板）",
}

EDGE_CASE_SYMBOLS = {
    "688981": "中芯国际（科创板）",
    "000029": "ST美丽（ST股）",
    "430047": "诺思兰德（北交所）",
}

BATCH_SYMBOLS = ["000001", "000002", "600036", "601318", "000858",
                 "002415", "300059", "600276", "601012", "000568"]


# ══════════════════════════════════════════════════════════════
# 1. 接口可用性验证
# ══════════════════════════════════════════════════════════════
def test_availability():
    """测试不同股票、不同复权参数下接口是否可用"""
    print("\n" + "=" * 70)
    print("【1】接口可用性验证")
    print("=" * 70)

    results = []
    for symbol, desc in TEST_SYMBOLS.items():
        for adjust in ["", "qfq", "hfq"]:
            label = f"{symbol}({desc}) adjust={adjust!r}"
            try:
                t0 = time.time()
                df = call_cyq_em(symbol=symbol, adjust=adjust)
                elapsed = time.time() - t0
                results.append({
                    "label": label, "status": "OK",
                    "rows": len(df), "cols": len(df.columns),
                    "elapsed_s": round(elapsed, 2),
                })
                print(f"  OK {label}: {len(df)} rows, {len(df.columns)} cols, {elapsed:.2f}s")
            except Exception as e:
                results.append({
                    "label": label, "status": "FAIL",
                    "error": str(e)[:120],
                })
                print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:80]}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n  总计 {len(results)} 次调用，成功 {ok_count} 次，失败 {len(results) - ok_count} 次")
    return results


# ══════════════════════════════════════════════════════════════
# 2. 返回字段与数据类型梳理
# ══════════════════════════════════════════════════════════════
FIELD_METADATA = {
    "日期":       {"cn": "日期",       "meaning": "交易日"},
    "获利比例":   {"cn": "获利比例",   "meaning": "当前价格以下筹码占比（0~1）"},
    "平均成本":   {"cn": "平均成本",   "meaning": "50%筹码累积处的价格"},
    "90成本-低":  {"cn": "90%成本下限", "meaning": "90%筹码集中区间下界"},
    "90成本-高":  {"cn": "90%成本上限", "meaning": "90%筹码集中区间上界"},
    "90集中度":   {"cn": "90%集中度",  "meaning": "(高-低)/(高+低)，越小越集中"},
    "70成本-低":  {"cn": "70%成本下限", "meaning": "70%筹码集中区间下界"},
    "70成本-高":  {"cn": "70%成本上限", "meaning": "70%筹码集中区间上界"},
    "70集中度":   {"cn": "70%集中度",  "meaning": "(高-低)/(高+低)，越小越集中"},
}


def inspect_fields(df: pd.DataFrame, symbol: str):
    """梳理字段信息：类型、范围、缺失率等"""
    print(f"\n  -- {symbol} 字段梳理 --")
    print(f"  Shape: {df.shape}")
    print(f"  日期范围: {df['日期'].min()} ~ {df['日期'].max()}")

    # 字段清单表
    print(f"\n  {'字段名':<12} {'类型':<10} {'非空率':>7} {'唯一值':>7} {'最小值':>12} {'最大值':>12} {'含义'}")
    print("  " + "-" * 90)
    for col in df.columns:
        meta = FIELD_METADATA.get(col, {})
        non_null_rate = df[col].notna().mean()
        nunique = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]):
            vmin = df[col].min()
            vmax = df[col].max()
        else:
            vmin = str(df[col].min())[:12]
            vmax = str(df[col].max())[:12]
        meaning = meta.get("meaning", "")
        print(f"  {col:<12} {str(df[col].dtype):<10} {non_null_rate:>6.2%} {nunique:>7} "
              f"{str(vmin):>12} {str(vmax):>12} {meaning}")

    # 样本数据
    print(f"\n  前 3 行:")
    print(df.head(3).to_string(index=False))
    print(f"\n  后 3 行:")
    print(df.tail(3).to_string(index=False))


# ══════════════════════════════════════════════════════════════
# 3. 数据校验
# ══════════════════════════════════════════════════════════════
def validate_data(df: pd.DataFrame, symbol: str) -> dict:
    """对单只股票的筹码分布数据进行全面校验"""
    issues = []
    n = len(df)

    # ── 3.1 缺失值检查 ──
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            issues.append(f"缺失值: {col} 有 {na_count}/{n} 个 NaN ({na_count/n:.1%})")

    # ── 3.2 范围校验 ──
    # 获利比例 ∈ [0, 1]
    out_of_range = df[(df["获利比例"] < 0) | (df["获利比例"] > 1)]
    if len(out_of_range) > 0:
        issues.append(f"范围: 获利比例 超出 [0,1] 共 {len(out_of_range)} 行")

    # 集中度 ∈ [0, 1]
    for col in ["90集中度", "70集中度"]:
        out = df[(df[col] < 0) | (df[col] > 1)]
        if len(out) > 0:
            issues.append(f"范围: {col} 超出 [0,1] 共 {len(out)} 行")

    # 成本低 ≤ 成本高
    for prefix in ["90", "70"]:
        low_col, high_col = f"{prefix}成本-低", f"{prefix}成本-高"
        bad = df[df[low_col] > df[high_col]]
        if len(bad) > 0:
            issues.append(f"范围: {low_col} > {high_col} 共 {len(bad)} 行")

    # 成本值 ≥ 0
    for col in ["90成本-低", "70成本-低", "平均成本"]:
        bad = df[df[col] < 0]
        if len(bad) > 0:
            issues.append(f"范围: {col} < 0 共 {len(bad)} 行")

    # ── 3.3 逻辑一致性校验 ──
    # 70% 区间 ⊂ 90% 区间
    bad_contain = df[(df["70成本-低"] < df["90成本-低"]) | (df["70成本-高"] > df["90成本-高"])]
    if len(bad_contain) > 0:
        issues.append(f"逻辑: 70%区间 ⊄ 90%区间 共 {len(bad_contain)} 行")

    # 70集中度 ≤ 90集中度（更窄区间集中度应更低或相等）
    bad_conc = df[df["70集中度"] > df["90集中度"] + 1e-6]
    if len(bad_conc) > 0:
        issues.append(f"逻辑: 70集中度 > 90集中度 共 {len(bad_conc)} 行")

    # 集中度公式验证: concentration = (high - low) / (high + low)
    for prefix in ["90", "70"]:
        low_col, high_col, con_col = f"{prefix}成本-低", f"{prefix}成本-高", f"{prefix}集中度"
        denom = df[high_col] + df[low_col]
        mask = denom > 0
        expected = (df.loc[mask, high_col] - df.loc[mask, low_col]) / denom[mask]
        diff = (df.loc[mask, con_col] - expected).abs()
        big_diff = diff[diff > 0.01]
        if len(big_diff) > 0:
            issues.append(f"公式: {con_col} 与 (高-低)/(高+低) 偏差>0.01 共 {len(big_diff)} 行, "
                          f"最大偏差={diff.max():.4f}")

    # 获利比例极端值检查
    all_profit = df[df["获利比例"] >= 1.0]
    all_loss = df[df["获利比例"] <= 0.0]
    if len(all_profit) > 0:
        issues.append(f"逻辑: 获利比例=1.0 共 {len(all_profit)} 行（全部获利，需确认是否合理）")
    if len(all_loss) > 0:
        issues.append(f"逻辑: 获利比例=0.0 共 {len(all_loss)} 行（全部亏损，需确认是否合理）")

    # 平均成本应在 90% 成本区间内或附近
    avg_outside_90 = df[
        (df["平均成本"] < df["90成本-低"] * 0.95) |
        (df["平均成本"] > df["90成本-高"] * 1.05)
    ]
    if len(avg_outside_90) > 0:
        issues.append(f"逻辑: 平均成本偏离90%区间超5% 共 {len(avg_outside_90)} 行")

    # 日期连续性（简单检查：日期应单调递增）
    dates = pd.to_datetime(df["日期"])
    if not dates.is_monotonic_increasing:
        issues.append("逻辑: 日期非单调递增")

    # 日期间隔检查
    date_diffs = dates.diff().dt.days.dropna()
    long_gaps = date_diffs[date_diffs > 7]
    if len(long_gaps) > 0:
        issues.append(f"日期: 间隔>7天共 {len(long_gaps)} 处，最长间隔 {long_gaps.max():.0f} 天")

    passed = len(issues) == 0
    return {"symbol": symbol, "passed": passed, "issues": issues, "rows": n}


# ══════════════════════════════════════════════════════════════
# 4. 边界场景测试
# ══════════════════════════════════════════════════════════════
def test_edge_cases():
    """测试边界场景：科创板、ST、北交所"""
    print("\n" + "=" * 70)
    print("【4】边界场景测试")
    print("=" * 70)

    for symbol, desc in EDGE_CASE_SYMBOLS.items():
        print(f"\n  -- {symbol}({desc}) --")
        try:
            t0 = time.time()
            df = call_cyq_em(symbol=symbol, adjust="")
            elapsed = time.time() - t0
            print(f"  OK 调用成功: {len(df)} rows, {elapsed:.2f}s")
            result = validate_data(df, symbol)
            if result["passed"]:
                print("  OK 数据校验通过")
            else:
                print(f"  FAIL 数据校验发现问题:")
                for issue in result["issues"]:
                    print(f"    - {issue}")
        except Exception as e:
            print(f"  FAIL 调用失败: {type(e).__name__}: {str(e)[:80]}")


# ══════════════════════════════════════════════════════════════
# 5. 性能评估
# ══════════════════════════════════════════════════════════════
def test_performance():
    """评估单只和批量调用耗时"""
    print("\n" + "=" * 70)
    print("【5】性能评估")
    print("=" * 70)

    # 单只
    t0 = time.time()
    call_cyq_em(symbol="000001", adjust="")
    single_elapsed = time.time() - t0
    print(f"\n  单只调用耗时: {single_elapsed:.2f}s")

    # 批量
    t0 = time.time()
    success = 0
    for sym in BATCH_SYMBOLS:
        try:
            call_cyq_em(symbol=sym, adjust="")
            success += 1
        except Exception:
            pass
    batch_elapsed = time.time() - t0
    avg_elapsed = batch_elapsed / max(success, 1)
    print(f"  批量 {len(BATCH_SYMBOLS)} 只: 成功 {success}, 总耗时 {batch_elapsed:.2f}s, "
          f"平均 {avg_elapsed:.2f}s/只")

    return {"single_s": single_elapsed, "batch_total_s": batch_elapsed,
            "batch_avg_s": avg_elapsed, "batch_success": success}


# ══════════════════════════════════════════════════════════════
# 源码分析（无需网络连接）
# ══════════════════════════════════════════════════════════════
def source_code_analysis():
    """基于 akshare 源码的静态分析，无需网络"""
    print("\n" + "=" * 70)
    print("【源码分析】stock_cyq_em 接口结构（无需网络）")
    print("=" * 70)

    # 字段清单
    print("\n  返回字段清单（9 列）:")
    print(f"  {'字段名':<12} {'类型':<10} {'含义'}")
    print("  " + "-" * 70)
    fields = [
        ("日期", "object/date", "交易日"),
        ("获利比例", "float64", "当前价格以下筹码占比，范围 [0, 1]"),
        ("平均成本", "float64", "50% 筹码累积处的价格（中位数成本）"),
        ("90成本-低", "float64", "90% 筹码集中区间下界价格"),
        ("90成本-高", "float64", "90% 筹码集中区间上界价格"),
        ("90集中度", "float64", "90% 集中度 = (高-低)/(高+低)，越小越集中"),
        ("70成本-低", "float64", "70% 筹码集中区间下界价格"),
        ("70成本-高", "float64", "70% 筹码集中区间上界价格"),
        ("70集中度", "float64", "70% 集中度 = (高-低)/(高+低)，越小越集中"),
    ]
    for name, dtype, meaning in fields:
        print(f"  {name:<12} {dtype:<10} {meaning}")

    # 算法参数
    print("\n  算法硬编码参数:")
    params = [
        ("this.range", "120", "CYQ 回溯窗口（天）"),
        ("factor", "150", "价格离散化档位数"),
        ("lmt", "210", "拉取 K 线条数"),
        ("klt", "101", "K 线频率（101=日线）"),
        ("输出截断", "90", "仅返回最近 90 天数据"),
    ]
    for name, value, meaning in params:
        print(f"  {name:<12} {value:<8} {meaning}")

    # 市场代码推断逻辑
    print("\n  市场代码推断逻辑:")
    print("  symbol.startswith('6') -> market_code=1 (沪市)")
    print("  其他 -> market_code=0 (深市)")
    print("  注意: 北交所代码(8/4开头)会被错误映射为深市(market_code=0)")

    # 依赖
    print("\n  依赖:")
    print("  - requests: 拉取东方财富 K 线数据")
    print("  - py_mini_racer: V8 引擎执行内嵌 JS 计算 CYQ")
    print("  - pandas: 数据处理")

    # 已知限制
    print("\n  已知限制:")
    limitations = [
        "仅返回最近 90 天数据（源码硬截断 temp_df.iloc[-90:]）",
        "120 天固定回溯窗口（JS 中 this.range = 120 硬编码）",
        "150 档价格离散化（JS 中 factor = 150 硬编码）",
        "不暴露逐价筹码分布数组（仅返回聚合指标）",
        "仅支持日线频率（klt=101 硬编码）",
        "市场代码推断仅区分 6 开头（沪市）和其他（深市），北交所 8/4 开头会被错误映射为深市",
        "依赖 py_mini_racer（V8 引擎），部分平台可能不可用",
        "每次调用需拉取 210 天 K 线 + 逐日 JS 计算，性能开销较大",
        "无请求头伪装，容易被东方财富 API 限流/封禁 IP",
    ]
    for i, lim in enumerate(limitations, 1):
        print(f"  {i}. {lim}")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("akshare 筹码分布接口（stock_cyq_em）全面调研")
    print(f"akshare 版本: {ak.__version__}")
    print("=" * 70)

    # ── 0. 源码分析（无需网络） ──
    source_code_analysis()

    # ── 1. 接口可用性 ──
    avail_results = test_availability()
    ok_count = sum(1 for r in avail_results if r["status"] == "OK")

    if ok_count == 0:
        print("\n  *** 所有在线调用均失败，跳过后续在线测试 ***")
        print("  *** 可能原因: IP 被东方财富 API 临时封禁，请稍后重试 ***")
        print("\n  源码分析部分已完成，在线测试需等待 IP 解封后重新运行。")
        return

    # ── 2. 字段梳理（以 000001 为基准） ──
    print("\n" + "=" * 70)
    print("【2】返回字段与数据类型梳理")
    print("=" * 70)
    try:
        df_ref = call_cyq_em(symbol="000001", adjust="")
        inspect_fields(df_ref, "000001")
    except Exception as e:
        print(f"  调用失败: {e}")

    # ── 3. 数据校验 ──
    print("\n" + "=" * 70)
    print("【3】数据校验")
    print("=" * 70)
    all_validation = []
    for symbol, desc in TEST_SYMBOLS.items():
        print(f"\n  -- {symbol}({desc}) --")
        try:
            df = call_cyq_em(symbol=symbol, adjust="")
            result = validate_data(df, symbol)
            all_validation.append(result)
            if result["passed"]:
                print(f"  OK 全部校验通过 ({result['rows']} rows)")
            else:
                print(f"  FAIL 发现 {len(result['issues'])} 个问题:")
                for issue in result["issues"]:
                    print(f"    - {issue}")
        except Exception as e:
            print(f"  FAIL 调用失败: {type(e).__name__}: {str(e)[:80]}")

    # 数据稳定性测试
    print(f"\n  -- 数据稳定性测试（000001 连续调用两次） --")
    try:
        df1 = call_cyq_em(symbol="000001", adjust="")
        df2 = call_cyq_em(symbol="000001", adjust="")
        if df1.shape == df2.shape:
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            max_diff = (df1[numeric_cols] - df2[numeric_cols]).abs().max().max()
            print(f"  两次调用 shape 一致: {df1.shape}, 数值最大差异: {max_diff}")
        else:
            print(f"  两次调用 shape 不一致: {df1.shape} vs {df2.shape}")
    except Exception as e:
        print(f"  稳定性测试失败: {type(e).__name__}: {str(e)[:80]}")

    # ── 4. 边界场景 ──
    test_edge_cases()

    # ── 5. 性能评估 ──
    try:
        perf = test_performance()
    except Exception as e:
        print(f"  性能评估失败: {type(e).__name__}: {str(e)[:80]}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("【汇总】数据校验问题")
    print("=" * 70)
    all_issues = []
    for v in all_validation:
        all_issues.extend(v["issues"])
    if all_issues:
        unique_issues = sorted(set(all_issues))
        for i, issue in enumerate(unique_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  未发现数据质量问题")


if __name__ == "__main__":
    main()
