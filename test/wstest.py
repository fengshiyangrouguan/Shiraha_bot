from fastapi import FastAPI, WebSocket
import uvicorn
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.platform.sources.qq_napcat.utils.reply_at import (
    parse_reply_from_records,
    get_reply_name,
    get_reply_text,
    get_at_nickname,
)

app = FastAPI()

@app.websocket("/")
async def onebot_ws(ws: WebSocket):
    await ws.accept()
    print("NapCat 已连接")
    while True:
        text = await ws.receive_text()
        data = json.loads(text)

        # 检查是否是消息事件
        if data.get("post_type") == "message":
            print("\n【解析回复信息】")
            
            # 获取 raw 数据
            raw = data.get("raw", {})
            if raw:
                # 尝试解析被回复的消息
                reply_info = parse_reply_from_records(data)
                
                if reply_info:
                    print(f"✓ 找到被回复的消息：")
                    print(f"  发送者 UID: {reply_info['sender_uid']}")
                    print(f"  发送者名字: {reply_info['sender_name']}")
                    print(f"  回复的原文: {reply_info['text']}")
                else:
                    print("✗ 这条消息不是回复，或未找到被回复的消息记录")
                
                # 检查 @ 内容
                ats = get_at_nickname(data)
                if ats:
                    print(f"\n【解析 @ 信息】")
                    print(f"  @的内容: {ats}")
            else:
                print("✗ 没有 raw 数据")
        
        print("="*80 + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)