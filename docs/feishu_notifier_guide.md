# 飞书消息使用指南

## 📋 目录
- [快速开始](#快速开始)
- [获取用户 ID](#获取用户-id)
- [使用方法](#使用方法)
- [API 示例](#api-示例)
- [故障排查](#故障排查)

---

## 🚀 快速开始

### 1. 获取你的飞书用户 ID

**推荐方法**：使用飞书 API 调试工具

1. 访问：https://open.feishu.cn/explorer/userinfo
2. 点击「调用」按钮
3. 复制返回结果中的 `open_id` 或 `user_id`

示例返回：
```json
{
  "code": 0,
  "data": {
    "user": {
      "user_id": "ou_xxxxxxxxx",
      "open_id": "ou_xxxxxxxxx",
      "union_id": "on_xxxxxxxxx",
      "name": "张三"
    }
  }
}
```

### 2. 更新配置文件

编辑 `/Users/zhenbao/Nextcloud/coding/交易/src/config.py`：

```python
FEISHU_USER_ID = 'ou_xxxxxxxxx'  # 填入你的 user_id 或 open_id
```

### 3. 测试消息发送

```bash
# 发送文本消息
cd /Users/zhenbao/Nextcloud/coding/交易/src
python test_feishu_message.py --text "你好，这是测试消息"

# 发送图片
python test_feishu_message.py --image /path/to/image.png

# 发送策略信号示例
python test_feishu_message.py --signal
```

---

## 📖 获取用户 ID 的三种方法

### 方法 1：API 调试工具（推荐）
1. 访问：https://open.feishu.cn/explorer/userinfo
2. 点击「调用」
3. 复制 `open_id` 或 `user_id`

### 方法 2：命令行查询
```bash
python get_feishu_userid.py 17611268685
```

**注意**：此方法需要应用在飞书开发者后台开通通讯录权限。

### 方法 3：飞书开发者应用
1. 在飞书中打开「飞书开发者」应用
2. 查看自己的用户信息

---

## 💻 使用方法

### 命令行工具

#### 发送文本消息
```bash
python test_feishu_message.py --text "测试消息内容"
```

#### 发送图片消息
```bash
python test_feishu_message.py --image /path/to/screenshot.png
```

#### 发送策略信号
```bash
python test_feishu_message.py --signal
```

#### 指定接收者
```bash
python test_feishu_message.py --text "测试" --receiver "ou_xxxxx" --id-type "open_id"
```

### Python API

```python
from feishu_notifier import FeishuNotifier

# 创建通知器
notifier = FeishuNotifier()

# 发送文本消息
notifier.send_text("这是一条测试消息")

# 发送图片
notifier.send_image("/path/to/image.png")

# 发送 Markdown 消息（使用交互式卡片）
markdown = """
## 标题
**加粗文本** 和 *斜体文本*
- 列表项 1
- 列表项 2
"""
notifier.send_markdown(markdown)

# 发送富文本消息
content_list = [
    [{"tag": "text", "text": "第一行文本"}],
    [{"tag": "a", "text": "点击链接", "href": "https://example.com"}],
    [{"tag": "at", "user_id": "ou_xxxxx"}]  # @某人
]
notifier.send_post("消息标题", content_list)

# 指定接收者
notifier.send_text("消息", receiver_id="ou_xxxxx", user_id_type="open_id")
```

---

## 🔧 API 示例

### 完整示例：发送策略报告

```python
from feishu_notifier import FeishuNotifier
from datetime import datetime

notifier = FeishuNotifier()

# 创建交互式卡片消息
content = {
    "config": {
        "wide_screen_mode": True
    },
    "header": {
        "template": "blue",
        "title": {
            "content": "📊 每日策略报告",
            "tag": "plain_text"
        }
    },
    "elements": [
        {
            "tag": "div",
            "text": {
                "content": f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "tag": "lark_md"
            }
        },
        {
            "tag": "hr"
        },
        {
            "tag": "div",
            "text": {
                "content": "**持仓股票**\n- 兴业银锡 (000426): +2.5%\n- 贵州茅台 (600519): -0.3%",
                "tag": "lark_md"
            }
        },
        {
            "tag": "img",
            "image_key": "img_xxxxx"  # 需要先上传图片获取 image_key
        }
    ]
}

notifier._send_message("interactive", content)
```

---

## ❓ 故障排查

### 问题 1：提示"未配置接收者 ID"
**原因**：config.py 中的 `FEISHU_USER_ID` 为空

**解决**：
1. 获取你的用户 ID（见上方「获取用户 ID」部分）
2. 编辑 `config.py`，设置 `FEISHU_USER_ID = '你的 ID'`

### 问题 2：提示"Access denied"或权限错误
**原因**：应用没有通讯录权限

**解决**：
1. 访问飞书开发者后台：https://open.feishu.cn/app/cli_a6b37d1d077b900e
2. 进入「权限管理」→「应用身份权限」
3. 申请并开通通讯录相关权限
4. 或使用手动方式获取用户 ID（推荐）

### 问题 3：图片发送失败
**原因**：图片路径不存在或格式不支持

**解决**：
1. 检查图片路径是否正确
2. 确保图片格式为 PNG、JPG 等常见格式
3. 图片大小不超过 10MB

### 问题 4：消息发送成功但对方未收到
**原因**：用户 ID 类型不匹配

**解决**：
- 如果填入的是 `open_id`，发送时需指定 `user_id_type="open_id"`
- 如果填入的是 `user_id`，发送时需指定 `user_id_type="user_id"`

---

## 📚 相关文档

- [飞书开放平台](https://open.feishu.cn/)
- [发送消息 API](https://open.feishu.cn/document/ukTMukTMukTM/ucjM1UjL7YzN14yN)
- [上传图片 API](https://open.feishu.cn/document/ukTMukTMukTM/uEjN1UjLxYjN14iN)
- [消息卡片构建器](https://open.feishu.cn/tool/cardbuilder)

---

## 🎯 下一步

1. ✅ 获取你的飞书用户 ID
2. ✅ 更新 `config.py` 配置文件
3. ✅ 运行测试：`python test_feishu_message.py --text "测试"`
4. 🎉 开始使用飞书消息通知功能！
