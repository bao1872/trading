from __future__ import annotations

"""
生成 pywencai 财务数据问句文件（批量年份/报告期版本）。

特点：
1. 只需要设置 START_YEAR
2. 自动根据当前日期，推断“最新自然报告期”
3. 自动生成 YEAR_LIST 和 REPORT_PERIOD_LIST
4. 每个问句模块都对应一个独立函数
5. 运行后会生成 6 个独立的 Python 文件
6. 每个文件内包含：
   - START_YEAR
   - YEAR_LIST
   - REPORT_PERIOD_LIST
   - QUERIES

说明：
- 这里的“最新”按当前日期对应的自然报告期推断
- 不保证该报告期所有公司都已经完成披露
"""

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# 在这里直接修改变量
# =========================
START_YEAR = 2021
OUTPUT_DIR = "./pywencai_queries"


def infer_latest_natural_report_period(today: date | None = None) -> Tuple[int, str]:
    """
    按当前日期推断最新自然报告期（不是披露完成口径）。

    规则：
    - 1~3月   -> 上一年四季度
    - 4~6月   -> 当年一季度
    - 7~9月   -> 当年二季度
    - 10~12月 -> 当年三季度
    """
    today = today or date.today()
    y, m = today.year, today.month

    if 1 <= m <= 3:
        return y - 1, "四季度"
    if 4 <= m <= 6:
        return y, "一季度"
    if 7 <= m <= 9:
        return y, "二季度"
    return y, "三季度"


def build_year_period_pairs(start_year: int, today: date | None = None) -> List[Tuple[int, str]]:
    latest_year, latest_period = infer_latest_natural_report_period(today=today)
    standard_periods = ["一季度", "二季度", "三季度", "四季度"]
    latest_idx = standard_periods.index(latest_period)

    pairs: List[Tuple[int, str]] = []
    for year in range(start_year, latest_year + 1):
        if year < latest_year:
            periods = standard_periods
        else:
            periods = standard_periods[: latest_idx + 1]

        for period in periods:
            pairs.append((year, period))

    return pairs


def build_year_list(pairs: List[Tuple[int, str]]) -> List[int]:
    seen: List[int] = []
    for y, _ in pairs:
        if y not in seen:
            seen.append(y)
    return seen


def build_report_period_list(pairs: List[Tuple[int, str]]) -> List[str]:
    return [f"{year}年{period}" for year, period in pairs]


def build_query(year: int, report_period: str, fields: List[str]) -> str:
    return f"{year}年{report_period}的" + "、".join(fields)


# =========================
# 6 个模块函数
# =========================

def query_income_profit_structure(year: int, report_period: str) -> str:
    fields = [
        "营业收入",
        "营业成本",
        "营业利润",
        "归属于母公司股东的净利润",
        "扣除非经常性损益后的归属于母公司股东的净利润",
        "销售费用",
        "管理费用",
        "研发费用",
        "财务费用",
        "投资收益",
        "公允价值变动收益",
        "信用减值损失",
        "资产减值损失",
    ]
    return build_query(year, report_period, fields)


def query_profit_quality_cash(year: int, report_period: str) -> str:
    fields = [
        "经营活动产生的现金流量净额",
        "归属于母公司股东的净利润",
        "扣除非经常性损益后的归属于母公司股东的净利润",
        "应收账款",
        "应收款项融资",
        "预付款项",
        "其他应收款",
        "存货",
        "合同资产",
        "商誉",
        "非经常性损益",
        "投资收益",
        "公允价值变动收益",
    ]
    return build_query(year, report_period, fields)


def query_working_capital_turnover(year: int, report_period: str) -> str:
    fields = [
        "应收账款",
        "应收款项融资",
        "预付款项",
        "存货",
        "合同资产",
        "应付账款",
        "营业收入",
        "营业成本",
        "应收账款周转率",
        "存货周转率",
        "总资产周转率",
    ]
    return build_query(year, report_period, fields)


def query_debt_safety(year: int, report_period: str) -> str:
    fields = [
        "货币资金",
        "短期借款",
        "一年内到期的非流动负债",
        "长期借款",
        "应付债券",
        "负债合计",
        "股东权益合计",
        "财务费用",
        "经营活动产生的现金流量净额",
        "资产负债率",
        "流动比率",
        "速动比率",
        "现金比率",
        "利息保障倍数",
    ]
    return build_query(year, report_period, fields)


def query_capex_roi(year: int, report_period: str) -> str:
    fields = [
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "固定资产",
        "在建工程",
        "无形资产",
        "开发支出",
        "营业收入",
        "经营活动产生的现金流量净额",
        "固定资产周转率",
    ]
    return build_query(year, report_period, fields)


def query_shareholder_return(year: int, report_period: str) -> str:
    fields = [
        "分红总额",
        "股利支付率",
        "股息率",
        "回购金额",
        "回购股数",
        "经营活动产生的现金流量净额",
        "归属于母公司股东的净利润",
    ]
    return build_query(year, report_period, fields)


# =========================
# 汇总函数
# =========================

def build_all_queries(start_year: int, today: date | None = None) -> Dict[str, List[str]]:
    pairs = build_year_period_pairs(start_year=start_year, today=today)
    result = {
        "01_income_profit_structure": [],
        "02_profit_quality_cash": [],
        "03_working_capital_turnover": [],
        "04_debt_safety": [],
        "05_capex_roi": [],
        "06_shareholder_return": [],
    }

    for year, report_period in pairs:
        result["01_income_profit_structure"].append(query_income_profit_structure(year, report_period))
        result["02_profit_quality_cash"].append(query_profit_quality_cash(year, report_period))
        result["03_working_capital_turnover"].append(query_working_capital_turnover(year, report_period))
        result["04_debt_safety"].append(query_debt_safety(year, report_period))
        result["05_capex_roi"].append(query_capex_roi(year, report_period))
        result["06_shareholder_return"].append(query_shareholder_return(year, report_period))

    return result


def file_header(module_name_cn: str) -> str:
    return (
        '"""\n'
        f"{module_name_cn} - pywencai 问句文件。\n"
        "本文件由 pywencai_query_builder_auto_range.py 自动生成。\n"
        '"""\n\n'
    )


def render_py_file(
    start_year: int,
    year_list: List[int],
    report_period_list: List[str],
    module_name_cn: str,
    queries: List[str],
) -> str:
    lines = [
        file_header(module_name_cn).rstrip("\n"),
        f"START_YEAR = {start_year}",
        f"YEAR_LIST = {year_list}",
        f"REPORT_PERIOD_LIST = {report_period_list!r}",
        "QUERIES = [",
    ]
    for q in queries:
        lines.append(f"    {q!r},")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def generate_files(start_year: int, outdir: Path, today: date | None = None) -> Dict[str, List[str]]:
    ensure_outdir(outdir)

    pairs = build_year_period_pairs(start_year=start_year, today=today)
    year_list = build_year_list(pairs)
    report_period_list = build_report_period_list(pairs)
    all_queries = build_all_queries(start_year=start_year, today=today)

    module_names = {
        "01_income_profit_structure": "收入与利润结构",
        "02_profit_quality_cash": "利润质量与现金含量",
        "03_working_capital_turnover": "营运资本与周转状态",
        "04_debt_safety": "偿债压力与财务安全边际",
        "05_capex_roi": "资本开支与投入产出状态",
        "06_shareholder_return": "股东回报与利润分配能力",
    }

    for filename_stem, queries in all_queries.items():
        content = render_py_file(
            start_year=start_year,
            year_list=year_list,
            report_period_list=report_period_list,
            module_name_cn=module_names[filename_stem],
            queries=queries,
        )
        (outdir / f"{filename_stem}.py").write_text(content, encoding="utf-8")

    summary_lines = [
        '"""自动生成的 pywencai 6 组财务问句总览。"""',
        f"START_YEAR = {start_year}",
        f"YEAR_LIST = {year_list}",
        f"REPORT_PERIOD_LIST = {report_period_list!r}",
        "",
        "ALL_QUERIES = {",
    ]
    for filename_stem, queries in all_queries.items():
        summary_lines.append(f'    "{filename_stem}": [')
        for q in queries:
            summary_lines.append(f"        {q!r},")
        summary_lines.append("    ],")
    summary_lines.append("}")
    summary_lines.append("")

    (outdir / "all_queries.py").write_text("\n".join(summary_lines), encoding="utf-8")
    return all_queries


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    all_queries = generate_files(start_year=START_YEAR, outdir=outdir)

    latest_year, latest_period = infer_latest_natural_report_period()
    print(f"起始年份: {START_YEAR}")
    print(f"按当前日期推断的最新自然报告期: {latest_year}年{latest_period}")
    print(f"输出目录: {outdir.resolve()}")
    print()

    for module_name, queries in all_queries.items():
        print(f"[{module_name}] 共 {len(queries)} 条")
        if queries:
            print("  首条:", queries[0])
            print("  末条:", queries[-1])


if __name__ == "__main__":
    main()