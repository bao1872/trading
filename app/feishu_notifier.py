# -*- coding: utf-8 -*-
"""
飞书消息发送脚本

支持发送：
- 文本消息
- 图片消息（本地图片文件）
- Markdown 消息
- 富文本消息

Usage:
    # 发送文本消息
    python feishu_notifier.py --text "测试消息"
    
    # 发送图片
    python feishu_notifier.py --image /path/to/image.png
    
    # 发送 Markdown
    python feishu_notifier.py --markdown "**加粗** 和 *斜体*"
    
    # 指定接收者
    python feishu_notifier.py --text "测试" --receiver "ou_xxxxx" --id-type "open_id"
"""
import os
import sys
import base64
import requests
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID


class FeishuNotifier:
    """飞书消息通知器"""
    
    def __init__(self, app_id: str = None, app_secret: str = None, default_receiver: str = None):
        """
        初始化飞书通知器
        
        Args:
            app_id: 飞书应用 App ID
            app_secret: 飞书应用 App Secret
            default_receiver: 默认接收者 ID（user_id 或 open_id）
        """
        self.app_id = app_id or FEISHU_APP_ID
        self.app_secret = app_secret or FEISHU_APP_SECRET
        self.default_receiver = default_receiver or FEISHU_USER_ID
        self._access_token = None
    
    def get_access_token(self) -> str:
        """获取访问令牌"""
        if self._access_token:
            return self._access_token
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        
        if result.get("code") != 0:
            raise Exception(f"获取 access_token 失败：{result.get('msg', 'Unknown error')}")
        
        self._access_token = result.get("tenant_access_token")
        return self._access_token
    
    def _send_message(self, msg_type: str, content: dict, receiver_id: str = None, 
                      user_id_type: str = "user_id") -> dict:
        """
        发送消息的底层方法
        
        Args:
            msg_type: 消息类型（text/image/markdown/post）
            content: 消息内容（字典格式）
            receiver_id: 接收者 ID
            user_id_type: 用户 ID 类型（user_id/open_id/union_id）
        
        Returns:
            API 响应结果
        """
        import json
        
        token = self.get_access_token()
        
        receiver_id = receiver_id or self.default_receiver
        if not receiver_id:
            raise ValueError("未指定接收者 ID，请在构造函数中传入 default_receiver 或调用时传入 receiver_id")
        
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "receive_id": receiver_id,
            "msg_type": msg_type,
            "content": json.dumps(content)  # 必须转换为 JSON 字符串
        }
        
        params = {
            "receive_id_type": user_id_type
        }
        
        response = requests.post(url, headers=headers, json=payload, params=params)
        result = response.json()
        
        return result
    
    def send_text(self, text: str, receiver_id: str = None, user_id_type: str = "user_id") -> dict:
        """
        发送文本消息
        
        Args:
            text: 文本内容
            receiver_id: 接收者 ID
            user_id_type: 用户 ID 类型
        
        Returns:
            API 响应结果
        """
        content = {"text": text}
        result = self._send_message("text", content, receiver_id, user_id_type)
        
        if result.get("code") == 0:
            print(f"✅ 文本消息发送成功")
        else:
            print(f"❌ 消息发送失败：{result.get('msg', 'Unknown error')}")
        
        return result
    
    def send_image(self, image_path: str, receiver_id: str = None, user_id_type: str = "user_id") -> dict:
        """
        发送图片消息
        
        Args:
            image_path: 本地图片文件路径
            receiver_id: 接收者 ID
            user_id_type: 用户 ID 类型
        
        Returns:
            API 响应结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在：{image_path}")
        
        # 第一步：上传图片获取 image_key
        token = self.get_access_token()
        
        upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
        upload_headers = {
            "Authorization": f"Bearer {token}"
        }
        
        with open(image_path, 'rb') as f:
            files = {
                "image": (os.path.basename(image_path), f)
            }
            data = {
                "image_type": "message"
            }
            
            response = requests.post(upload_url, headers=upload_headers, files=files, data=data)
            upload_result = response.json()
        
        if upload_result.get("code") != 0:
            error_msg = upload_result.get("msg", "Unknown error")
            print(f"❌ 图片上传失败：{error_msg}")
            return upload_result
        
        image_key = upload_result.get("data", {}).get("image_key")
        
        # 第二步：发送图片消息
        content = {"image_key": image_key}
        result = self._send_message("image", content, receiver_id, user_id_type)
        
        if result.get("code") == 0:
            print(f"✅ 图片消息发送成功：{image_path}")
        else:
            print(f"❌ 消息发送失败：{result.get('msg', 'Unknown error')}")
        
        return result
    
    def send_markdown(self, markdown: str, receiver_id: str = None, user_id_type: str = "user_id") -> dict:
        """
        发送 Markdown 消息（飞书卡片格式）
        
        Args:
            markdown: Markdown 格式文本
            receiver_id: 接收者 ID
            user_id_type: 用户 ID 类型
        
        Returns:
            API 响应结果
        """
        # 飞书卡片格式
        content = {
            "config": {
                "wide_screen_mode": True
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": markdown
                }
            ]
        }
        result = self._send_message("interactive", content, receiver_id, user_id_type)
        
        if result.get("code") == 0:
            print(f"✅ Markdown 消息发送成功")
        else:
            print(f"❌ 消息发送失败：{result.get('msg', 'Unknown error')}")
        
        return result
    
    def send_post(self, title: str, content_list: list, receiver_id: str = None, 
                  user_id_type: str = "user_id") -> dict:
        """
        发送富文本消息
        
        Args:
            title: 消息标题
            content_list: 内容列表，每项是一个元素列表
                        例如：[
                            [{"tag": "text", "text": "第一行"}],
                            [{"tag": "a", "text": "链接", "href": "https://..."}],
                            [{"tag": "at", "user_id": "ou_xxx"}]
                        ]
            receiver_id: 接收者 ID
            user_id_type: 用户 ID 类型
        
        Returns:
            API 响应结果
        """
        content = {
            "zh_cn": {
                "title": title,
                "content": content_list
            }
        }
        result = self._send_message("post", content, receiver_id, user_id_type)
        
        if result.get("code") == 0:
            print(f"✅ 富文本消息发送成功")
        else:
            print(f"❌ 消息发送失败：{result.get('msg', 'Unknown error')}")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="飞书消息发送工具")
    parser.add_argument("--text", type=str, help="文本消息内容")
    parser.add_argument("--image", type=str, help="图片文件路径")
    parser.add_argument("--markdown", type=str, help="Markdown 消息内容")
    parser.add_argument("--receiver", type=str, help="接收者 ID（不填则使用配置文件中的默认值）")
    parser.add_argument("--id-type", type=str, default="user_id", 
                        choices=["user_id", "open_id", "union_id"],
                        help="用户 ID 类型")
    parser.add_argument("--title", type=str, help="富文本消息标题")
    
    args = parser.parse_args()
    
    notifier = FeishuNotifier()
    
    if args.text:
        notifier.send_text(args.text, args.receiver, args.id_type)
    elif args.image:
        notifier.send_image(args.image, args.receiver, args.id_type)
    elif args.markdown:
        notifier.send_markdown(args.markdown, args.receiver, args.id_type)
    else:
        print("请指定消息类型：--text / --image / --markdown")
        parser.print_help()


if __name__ == "__main__":
    main()
