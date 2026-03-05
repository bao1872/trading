# -*- coding: utf-8 -*-
"""
飞书消息测试脚本

示例：
    # 发送文本消息
    python test_feishu_message.py --text "你好，这是测试消息"
    
    # 发送图片
    python test_feishu_message.py --image test_image.png
    
    # 发送策略信号
    python test_feishu_message.py --signal
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feishu_notifier import FeishuNotifier


def send_test_text(notifier: FeishuNotifier, receiver: str = None, id_type: str = "user_id"):
    """发送测试文本消息"""
    text = f"【飞书消息测试】\n时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n这是一条测试消息，请确认收到。"
    notifier.send_text(text, receiver, id_type)


def send_test_image(notifier: FeishuNotifier, image_path: str, receiver: str = None, id_type: str = "user_id"):
    """发送测试图片"""
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在：{image_path}")
        return
    
    notifier.send_image(image_path, receiver, id_type)


def send_strategy_signal(notifier: FeishuNotifier, receiver: str = None, id_type: str = "user_id"):
    """发送策略信号示例"""
    markdown = f"""## 📊 策略信号提醒

**股票**: 兴业银锡 (000426)
**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 趋势信号
- 趋势方向：上涨 ✓
- 趋势强度：0.75
- 当前位置：接近上轨

### 空间信号
- 上轨价格：12.50
- 下轨价格：11.80
- 止损位：11.65
- 风险收益比：2.5

### 触发信号
- 5 分钟：底背离 ✓
- 15 分钟：底背离 ✓
- 30 分钟：中性

**建议**: 可考虑建仓，注意止损位
"""
    # 使用 interactive 消息类型发送 Markdown
    content = {
        "config": {
            "wide_screen_mode": True
        },
        "header": {
            "template": "blue",
            "title": {
                "content": "📊 策略信号提醒",
                "tag": "plain_text"
            }
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "content": f"**股票**: 兴业银锡 (000426)\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "tag": "lark_md"
                }
            },
            {
                "tag": "hr"
            },
            {
                "tag": "div",
                "text": {
                    "content": "**趋势信号**\n- 趋势方向：上涨 ✓\n- 趋势强度：0.75\n- 当前位置：接近上轨",
                    "tag": "lark_md"
                }
            },
            {
                "tag": "div",
                "text": {
                    "content": "**空间信号**\n- 上轨价格：12.50\n- 下轨价格：11.80\n- 止损位：11.65\n- 风险收益比：2.5",
                    "tag": "lark_md"
                }
            },
            {
                "tag": "div",
                "text": {
                    "content": "**触发信号**\n- 5 分钟：底背离 ✓\n- 15 分钟：底背离 ✓\n- 30 分钟：中性",
                    "tag": "lark_md"
                }
            },
            {
                "tag": "hr"
            },
            {
                "tag": "div",
                "text": {
                    "content": "**建议**: 可考虑建仓，注意止损位 💡",
                    "tag": "lark_md"
                }
            }
        ]
    }
    
    result = notifier._send_message("interactive", content, receiver, id_type)
    if result.get("code") == 0:
        print("✅ 策略信号发送成功")
    else:
        print(f"❌ 发送失败：{result.get('msg', 'Unknown error')}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="飞书消息测试工具")
    parser.add_argument("--text", type=str, help="发送自定义文本消息")
    parser.add_argument("--image", type=str, help="发送图片消息（文件路径）")
    parser.add_argument("--signal", action="store_true", help="发送策略信号示例")
    parser.add_argument("--receiver", type=str, help="接收者 ID（不填则使用配置文件）")
    parser.add_argument("--id-type", type=str, default="user_id", 
                        choices=["user_id", "open_id", "union_id"],
                        help="用户 ID 类型")
    
    args = parser.parse_args()
    
    # 创建通知器实例
    notifier = FeishuNotifier()
    
    # 检查是否配置了接收者
    receiver = args.receiver or notifier.default_receiver
    if not receiver:
        print("=" * 60)
        print("⚠️  未配置接收者 ID")
        print("=" * 60)
        print("\n请按以下步骤操作:")
        print("1. 获取你的用户 ID:")
        print("   访问：https://open.feishu.cn/explorer/userinfo")
        print("   复制 open_id 或 user_id")
        print("\n2. 更新配置文件:")
        print("   编辑 src/config.py")
        print("   设置：FEISHU_USER_ID = '你的 user_id 或 open_id'")
        print("\n3. 或者在命令行指定接收者:")
        print("   python test_feishu_message.py --text '测试' --receiver 'ou_xxx' --id-type 'open_id'")
        print("=" * 60)
        return
    
    print(f"接收者：{receiver} (类型：{args.id_type})")
    print("=" * 60)
    
    if args.text:
        send_test_text(notifier, receiver, args.id_type)
    elif args.image:
        send_test_image(notifier, args.image, receiver, args.id_type)
    elif args.signal:
        send_strategy_signal(notifier, receiver, args.id_type)
    else:
        print("请指定消息类型：--text / --image / --signal")
        parser.print_help()


if __name__ == "__main__":
    main()
