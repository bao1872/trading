# -*- coding: utf-8 -*-
"""
DeepSeek API 五维评分客户端 - 构造 Prompt 并调用 LLM 提取结构化评分

Purpose: 接收股票事实上下文，构造符合 stock-scoring skill 规范的 prompt，
         调用 DeepSeek API 获取五维评分 + 总评分 + 炒作逻辑，解析为结构化字段
Inputs:
    - context: 单只股票的上下文 dict（来自 stock_context_builder）
    - api_key: DeepSeek API Key（可选，默认从环境变量读取）
Outputs:
    - dict: {ts_code, stock_name, industry_attractiveness, competitive_position,
             business_model_quality, growth_sustainability, management_capital,
             total_score, hype_logic, score_credibility, company_tags, raw_response}
How to Run:
    python financial_factors/llm_score_client.py --test
    python financial_factors/llm_score_client.py --ts-code 000426.SZ
Examples:
    python financial_factors/llm_score_client.py --test
    python financial_factors/llm_score_client.py --ts-code 600519.SH
Side Effects: 调用外部 DeepSeek API（网络请求），无数据库写入
"""
import json
import logging
import os
import sys
import time
from typing import Dict, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-chat"
MAX_RETRIES = 3
RETRY_BACKOFF = 2
REQUEST_TIMEOUT = 90

EXPECTED_SCORE_KEYS = [
    "industry_attractiveness",
    "competitive_position",
    "business_model_quality",
    "growth_sustainability",
    "management_capital",
    "total_score",
    "hype_logic",
    "score_credibility",
    "company_tags",
]

SYSTEM_PROMPT = """你是一位资深A股股票分析师，严格按照以下五维评分卡规则评估股票质量。

## 五维评分体系（各20分，总分100分）

### 1. 行业吸引力（20分）
- 行业增速/空间/渗透率
- 行业盈利与竞争格局（集中度、是否内卷）
- 政策/技术驱动方向

### 2. 竞争地位（20分）
- 市场地位（龙头/第二梯队/跟随者）
- 技术壁垒/产品差异化
- 客户粘性/渠道/认证壁垒
- 与同行对比的相对优势

### 3. 商业模式与盈利质量（20分）
- 利润质量：毛利率、净利率、非经常性损益占比
- 现金流质量：经营现金流/净利润、应收/存货变化
- 资本效率：ROE/ROIC、资产周转率

### 4. 成长持续性（20分）
- 近2-3年收入/扣非利润增长趋势
- 增长支撑：新产品/新客户/产能扩张/订单
- 增长来源可持续性判断

### 5. 管理层与资本配置（20分）
- 资本回报：3年平均ROE/ROIC
- 资本配置：并购/分红/回购/扩产是否合理
- 治理风险：商誉/关联交易/违规处罚

## 评分规则
- 优先使用行业内相对位置（同行领先/中上/中游/偏弱→映射分数）
- 缺失数据标注"不确定"，给中性保守分
- 每维需给一句话依据

## 总分区间解释
- 85-100：高质量公司，综合优势明显
- 70-84：较优质公司，部分维度突出
- 55-69：中性公司，优缺点并存
- 40-54：质量一般，短板明显
- 0-39：高风险/弱质公司"""


def build_user_prompt(ctx: Dict) -> str:
    """根据股票上下文构造用户 prompt"""
    prompt = f"""请对以下A股公司进行五维质量评分。

## 公司基本信息
- 股票代码: {ctx.get('ts_code', '')}
- 公司名称: {ctx.get('stock_name', '')}
- 所属行业(二级): {ctx.get('industry_l2', '')}
- 所属行业(三级): {ctx.get('industry_l3', '')}
- 所属概念: {ctx.get('concepts', '')[:200]}
- 当前市值: {ctx.get('market_cap', '')}
- 行业平均PE: {ctx.get('industry_pe', '')}

## 最新财报数据（{ctx.get('report_date', '')}）
- 营业收入: {ctx.get('revenue', '')}
- 归母净利润: {ctx.get('net_profit_parent', '')}
- 扣非归母净利润: {ctx.get('net_profit_deducted', '')}
- 毛利率: {ctx.get('gross_margin', '')}
- 净利率: {ctx.get('net_margin', '')}
- ROE: {ctx.get('roe', '')}
- 资产负债率: {ctx.get('debt_ratio', '')}
- 流动比率: {ctx.get('current_ratio', '')}
- 经营现金流净额: {ctx.get('cfo', '')}
- 研发费用: {ctx.get('rd_exp', '')}

## 同期对比（上一期）
- 上期营收: {ctx.get('prev_revenue', '')}
- 上期归母净利润: {ctx.get('prev_net_profit', '')}
- 上期毛利率: {ctx.get('prev_gross_margin', '')}
- 营收同比变化: {ctx.get('rev_yoy', '')}
- 净利润同比变化: {ctx.get('np_yoy', '')}

## 利润与现金流质量
- CFO/净利润比率: {ctx.get('cfo_to_np', '')}
- 非经常性损益占比: {ctx.get('non_recurring_pct', '')}
- 应收账款: {ctx.get('ar', '')}
- 存货: {ctx.get('inventory', '')}
- 商誉: {ctx.get('goodwill', '')}
- 总负债: {ctx.get('total_debt', '')}
- 股东权益: {ctx.get('equity', '')}

## 特殊项目
- 投资收益: {ctx.get('inv_income', '')}
- 公允价值变动: {ctx.get('fv_change', '')}
- 信用减值损失: {ctx.get('credit_loss', '')}
- 资产减值损失: {ctx.get('asset_loss', '')}
- 销售费用: {ctx.get('sell_exp', '')}
- 管理费用: {ctx.get('mgmt_exp', '')}

## 季度数据（最新单季度）
- 单季营收: {ctx.get('qtr_revenue', '')}
- 单季归母净利润: {ctx.get('qtr_profit', '')}
- 单季经营现金流: {ctx.get('qtr_cfo', '')}
- 单季EBIT: {ctx.get('qtr_ebit', '')}

## 已知炒作逻辑（如存在）
{ctx.get('hype_logic', '暂无')}

## 输出要求
请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{{
  "industry_attractiveness": <0-20的数值>,
  "industry_attractiveness_reason": "<一句话依据>",
  "competitive_position": <0-20的数值>,
  "competitive_position_reason": "<一句话依据>",
  "business_model_quality": <0-20的数值>,
  "business_model_quality_reason": "<一句话依据>",
  "growth_sustainability": <0-20的数值>,
  "growth_sustainability_reason": "<一句话依据>",
  "management_capital": <0-20的数值>,
  "management_capital_reason": "<一句话依据>",
  "total_score": <五维之和>,
  "hype_logic": "<50字以内的最近炒作逻辑概述>",
  "score_credibility": "<高/中/低>",
  "company_tags": "<1-3个标签，用逗号分隔，如：细分龙头型,高成长扩张型>"
}}
```

注意：
- 每个维度分数必须是0-20之间的数值，保留1位小数
- total_score为五维之和的数值
- hype_logic必须提炼到50字以内
- 没有依据不给分，缺失数据标注不确定
"""
    return prompt


def parse_score_response(raw_text: str, ts_code: str, stock_name: str) -> Optional[Dict]:
    """从 LLM 返回文本中解析 JSON 评分

    Args:
        raw_text: LLM 原始返回文本
        ts_code: 股票代码
        stock_name: 股票名称

    Returns:
        解析后的评分 dict，解析失败返回 None
    """
    if not raw_text or not raw_text.strip():
        logger.warning(f"[{ts_code}] LLM 返回为空")
        return None

    text = raw_text.strip()

    json_str = None
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()

    if json_str is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            json_str = text[start:end + 1]

    if json_str is None:
        logger.warning(f"[{ts_code}] 未找到 JSON 结构，原始返回: {text[:200]}")
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[{ts_code}] JSON 解析失败: {e}, 片段: {json_str[:200]}")
        return None

    result = {"ts_code": ts_code, "stock_name": stock_name, "raw_response": raw_text}

    for key in EXPECTED_SCORE_KEYS:
        if key in data:
            result[key] = data[key]
        else:
            logger.warning(f"[{ts_code}] 缺少字段: {key}")
            result[key] = None

    for dim in ["industry_attractiveness", "competitive_position", "business_model_quality",
                "growth_sustainability", "management_capital", "total_score"]:
        val = result.get(dim)
        if val is not None:
            try:
                fval = float(val)
                if dim == "total_score":
                    if fval < 0 or fval > 100:
                        logger.warning(f"[{ts_code}] {dim}={fval} 超出 0-100 范围")
                else:
                    if fval < 0 or fval > 20:
                        logger.warning(f"[{ts_code}] {dim}={fval} 超出 0-20 范围")
                result[dim] = round(fval, 2)
            except (ValueError, TypeError):
                logger.warning(f"[{ts_code}] {dim}={val} 无法转为数值")
                result[dim] = None

    hype = result.get("hype_logic", "")
    if isinstance(hype, str) and len(hype) > 50:
        result["hype_logic"] = hype[:50]

    credibility = result.get("score_credibility", "")
    if credibility not in ("高", "中", "低"):
        result["score_credibility"] = "中"

    return result


def call_deepseek_api(
    ctx: Dict,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> Optional[Dict]:
    """调用 DeepSeek API 获取五维评分

    Args:
        ctx: 股票上下文 dict
        api_key: API Key，默认从环境变量/配置文件读取
        model: 模型名称
        dry_run: True 时只构造 prompt 不实际调用

    Returns:
        评分结果 dict，失败返回 None
    """
    ts_code = ctx.get("ts_code", "unknown")
    stock_name = ctx.get("stock_name", "")

    user_prompt = build_user_prompt(ctx)

    if dry_run:
        logger.info(f"[{ts_code}] DRY-RUN: prompt长度={len(user_prompt)}")
        return None

    key = api_key or DEEPSEEK_API_KEY
    if not key:
        logger.error("DEEPSEEK_API_KEY 未配置，请设置环境变量 DEEPSEEK_API_KEY")
        return None

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[{ts_code}] API 调用 (第{attempt}次)")
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()

            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")

            result = parse_score_response(content, ts_code, stock_name)
            if result is None:
                logger.warning(f"[{ts_code}] 响应解析失败，重试...")
                last_error = "响应解析失败"
                time.sleep(RETRY_BACKOFF ** attempt)
                continue

            logger.info(f"[{ts_code}] {stock_name} 总分={result.get('total_score')}")
            return result

        except requests.exceptions.Timeout:
            last_error = "超时"
            logger.warning(f"[{ts_code}] 请求超时 (第{attempt}次)")
            time.sleep(RETRY_BACKOFF ** attempt)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code == 429:
                last_error = "限流429"
                wait = RETRY_BACKOFF ** attempt * 2
                logger.warning(f"[{ts_code}] 触发限流，等待{wait}s")
                time.sleep(wait)
            else:
                last_error = f"HTTP {status_code}"
                logger.warning(f"[{ts_code}] HTTP错误 {status_code}: {e}")
                time.sleep(RETRY_BACKOFF ** attempt)
        except requests.exceptions.RequestException as e:
            last_error = f"网络错误: {e}"
            logger.warning(f"[{ts_code}] 网络错误 (第{attempt}次): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
        except Exception as e:
            last_error = f"未知错误: {e}"
            logger.warning(f"[{ts_code}] 异常 (第{attempt}次): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)

    logger.error(f"[{ts_code}] 重试{MAX_RETRIES}次后仍失败: {last_error}")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试 DeepSeek API 五维评分")
    parser.add_argument("--test", action="store_true", help="用模拟数据测试 prompt 构造")
    parser.add_argument("--ts-code", type=str, help="从数据库加载指定股票上下文并评分")
    parser.add_argument("--limit", type=int, default=1, help="评分数量")
    args = parser.parse_args()

    if args.test:
        mock_ctx = {
            "ts_code": "600519.SH",
            "stock_name": "贵州茅台",
            "industry_l2": "白酒",
            "industry_l3": "白酒",
            "concepts": "白酒龙头,大消费",
            "market_cap": "1.85万亿",
            "industry_pe": "25.00x",
            "report_date": "20250930",
            "revenue": "1354.28亿",
            "net_profit_parent": "716.25亿",
            "net_profit_deducted": "713.80亿",
            "gross_margin": "91.50%",
            "net_margin": "52.88%",
            "roe": "30.25%",
            "debt_ratio": "12.30%",
            "current_ratio": "4.50x",
            "cfo": "680.00亿",
            "rd_exp": "3.00亿",
            "prev_revenue": "1200.00亿",
            "prev_net_profit": "650.00亿",
            "prev_gross_margin": "91.00%",
            "rev_yoy": "12.86%",
            "np_yoy": "10.19%",
            "cfo_to_np": "95.00%",
            "non_recurring_pct": "0.34%",
            "ar": "1.00亿",
            "inventory": "420.00亿",
            "goodwill": "0",
            "total_debt": "250.00亿",
            "equity": "2100.00亿",
            "inv_income": "0",
            "fv_change": "0",
            "credit_loss": "0",
            "asset_loss": "0",
            "sell_exp": "35.00亿",
            "mgmt_exp": "60.00亿",
            "qtr_revenue": "420.00亿",
            "qtr_profit": "238.00亿",
            "qtr_cfo": "215.00亿",
            "qtr_ebit": "310.00亿",
            "hype_logic": "白酒龙头,消费复苏预期下的估值修复",
        }
        prompt = build_user_prompt(mock_ctx)
        print("===== 模拟 Prompt =====")
        print(prompt)
        print(f"\nPrompt 长度: {len(prompt)} 字符")

        has_key = bool(DEEPSEEK_API_KEY)
        print(f"\nDEEPSEEK_API_KEY: {'已配置' if has_key else '❌ 未配置'}")
        if has_key:
            print("正在调用 DeepSeek API...")
            result = call_deepseek_api(mock_ctx)
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("API 调用失败")
        return

    if args.ts_code:
        from datasource.database import get_session
        from financial_factors.stock_context_builder import build_context, load_stock_pool

        with get_session() as session:
            pool_df = load_stock_pool(session, limit=100)
            target = pool_df[pool_df["ts_code"] == args.ts_code]
            if target.empty:
                print(f"股票 {args.ts_code} 不在 stock_pools 中")
                return
            contexts, _ = build_context(session, stock_pool_df=target)
            for ctx in contexts[: args.limit]:
                print(f"\n===== {ctx['ts_code']} {ctx['stock_name']} =====")
                result = call_deepseek_api(ctx)
                if result:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print("评分失败")


if __name__ == "__main__":
    main()
