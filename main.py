# main.py
import asyncio
import sys
import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# 将 src 目录添加到Python的模块搜索路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.core.bot import MainSystem
from src.utils.logger import logger

# 创建 FastAPI 应用实例
app = FastAPI()

# 创建一个单例的 MainSystem 实例，在整个应用生命周期中共享
# 这样做可以避免每次连接都重新加载插件和模型
main_system = MainSystem()

# 一个简单的HTML页面，用于提供WebSocket测试客户端
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Shiraha_bot 聊天测试</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        h1 {
            text-align: center;
            color: #4a6cf7;
            margin: 10px 0;
            font-size: 1.5em;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        .message {
            max-width: 70%;
            padding: 10px 14px;
            margin-bottom: 12px;
            border-radius: 18px;
            line-height: 1.4;
            word-wrap: break-word;
            position: relative;
        }
        .user-message {
            background-color: #4a6cf7;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        #input-area {
            display: flex;
            padding: 12px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        #messageText {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 20px;
            outline: none;
            font-size: 1em;
        }
        #messageText:focus {
            border-color: #4a6cf7;
            box-shadow: 0 0 0 2px rgba(74, 108, 247, 0.2);
        }
        button {
            margin-left: 10px;
            padding: 10px 20px;
            background-color: #4a6cf7;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1em;
        }
        button:hover {
            background-color: #3a5bf5;
        }
        /* 滚动条美化（可选） */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>投资机器人聊天测试</h1>
    <div id="chat-container" id='messages'></div>
    <form id="input-area" onsubmit="sendMessage(event)">
        <input type="text" id="messageText" autocomplete="off" placeholder="输入消息..." autofocus />
        <button type="submit">发送</button>
    </form>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('messageText');

        // 连接到 WebSocket（路径可动态修改，这里写死用于测试）
        const ws = new WebSocket("ws://localhost:8000/ws/test_group_001/test_user_12345");

        ws.onopen = () => {
            appendMessage("欢迎使用 Shiraha_bot！请输入消息开始对话。", 'bot');
        };

        ws.onmessage = (event) => {
            appendMessage(event.data, 'bot');
        };

        ws.onerror = (error) => {
            appendMessage("❌ 连接出错，请检查服务是否运行。", 'bot');
            console.error("WebSocket error:", error);
        };

        ws.onclose = () => {
            appendMessage("⚠️ 连接已断开。", 'bot');
        };

        function appendMessage(text, sender) {
            const messageElement = document.createElement('div');
            messageElement.classList.add('message');
            messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
            messageElement.textContent = text;
            chatContainer.appendChild(messageElement);
            // 自动滚动到底部
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage(event) {
            event.preventDefault();
            const content = messageInput.value.trim();
            if (!content) return;

            // 显示用户消息
            appendMessage(content, 'user');
            messageInput.value = '';

            // 发送给后端
            ws.send(content);
        }

        // 支持回车发送
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage(e);
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    """提供一个简单的HTML页面用于测试"""
    return HTMLResponse(html)

@app.websocket("/ws/{stream_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str, user_id: str):
    """
    WebSocket 通信端点。
    每个连接代表一个用户在一个对话中的会话。
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for stream '{stream_id}', user '{user_id}'")
    
    try:
        while True:
            # 等待并接收来自客户端的消息
            message_content = await websocket.receive_text()
            logger.info(f"Received message from '{user_id}': {message_content}")

            # 调用 MainSystem 的核心处理方法
            bot_reply = await main_system.handle_message(
                stream_id=stream_id,
                user_id=user_id,
                message_content=message_content
            )

            # 将机器人的回复（或提示信息）发送回客户端
            if bot_reply:
                await websocket.send_text(f"{bot_reply}")
            else:
                # 即使不回复，也可以给前端一个反馈，表明消息已收到但被忽略
                await websocket.send_text("...（选择不回复）")

    except WebSocketDisconnect:
        logger.info(f"Client {user_id} disconnected from stream {stream_id}.")
    except Exception as e:
        logger.error(f"An error occurred in WebSocket for {user_id}: {e}")
        await websocket.send_text(f"Error: {e}")


if __name__ == "__main__":
    logger.info("Starting Shiraha_bot web server...")
    # 使用 uvicorn 启动服务器
    # host="0.0.0.0" 让服务可以被局域网访问
    # reload=True 可以在代码变更时自动重启服务，方便开发
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)