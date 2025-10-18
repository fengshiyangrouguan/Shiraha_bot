import asyncio
import json
import uuid # 导入 uuid 用于生成唯一的 API 请求 ID
import websockets as ws 
from typing import Dict, Any, Awaitable, Callable, Optional, Set
from asyncio import Queue, Event, wait_for, CancelledError, TimeoutError

# 假设 logger 和 ResponsePool 是从外部导入或注入的
from src.logger import logger
from src.response_pool import ResponsePool # 假设这个类存在

# 导入基类
from .base_adapter import BaseAdapter 

class QQAdapter(BaseAdapter):
    
    # 构造函数只需调用父类
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保持使用基类的 self.platform_connection 作为单个连接对象
        self.platform_connection: Optional[ws.WebSocketServerProtocol] = None
        self.server: Optional[ws.WebSocketServer] = None
        self.max_size = getattr(self.config, 'max_size', 2**26)

    # --- 抽象方法 1: I/O 启动 ---

    async def _start_platform_io(self) -> None:
        """
        QQ 平台特有逻辑：启动一个 WebSocket 服务器。
        """
        async with ws.serve(self._handle_connection, self.config.host, self.config.port, max_size=self.max_size) as server:
            self.server = server
            logger.info("QQAdapter: WebSocket 服务器已上线。")
            await server.serve_forever()

    async def _handle_connection(self, websocket: ws.WebSocketServerProtocol, path: str):
        """
        实际的 WebSocket 消息接收循环，将消息推入基类的队列。
        """

        # 关键操作 1: 记录当前活动连接
        self.platform_connection = websocket
        logger.info(f"QQAdapter: 接收到新的连接，路径: {path}")

        try:
            # 3. 循环接收并分流消息
            async for raw_message in websocket:
                decoded_raw_message: dict = json.loads(raw_message)
                echo_id = decoded_raw_message.get("echo")

                if echo_id:
                    # **路径 A: API 响应。立即送入 ResponsePool。**
                    self.response_pool.put_response(echo_id, decoded_raw_message)
                else:
                    # **路径 B: 事件/业务消息。送入队列等待消费者处理。**
                    await self.message_queue.put(decoded_raw_message)
                    
                
        except (ws.ConnectionClosedError, CancelledError):
            logger.warning("QQAdapter: 连接被远程关闭或任务取消。")
        except Exception as e:
            logger.error(f"QQAdapter: 连接处理异常: {e}")
        finally:
            # 关键操作 2: 只有当断开的连接是当前活动连接时，才清空状态
            if self.platform_connection is websocket:
                self.platform_connection = None
                logger.info("QQAdapter: 当前活动连接已断开，状态已清空。")
            else:
                 # 这可能是第二个连接被拒绝或断开
                 logger.info("QQAdapter: 非当前活动连接断开，状态保持不变。")


    # --- 抽象方法 2: I/O 停止器 ---

    async def _stop_io_connection(self) -> None:
        """
        关闭 WebSocket 服务器和单个活动连接。
        """
        if self.server:
            self.server.close()
            
        # 强制关闭任何活动的连接
        if self.platform_connection:
            try:
                await self.platform_connection.close()
            except Exception:
                pass # 忽略已关闭的连接
        
        self.platform_connection = None
        logger.info("QQAdapter: I/O 连接已清理完毕 (单连接模式)。")













    # # --- 核心功能：同步 API 命令发送 ---

    # async def send_command_and_wait(self, command: str, params: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    #     """
    #     发送一个同步 API 命令到 QQ/NapCat，并阻塞等待响应。
    #     """
        
    #     websocket_to_send = self.platform_connection
    #     if not websocket_to_send:
    #         logger.error(f"无法发送命令 '{command}'：没有活动的 WebSocket 连接。")
    #         raise RuntimeError("No active connection available for command.")
            
    #     # 1. 生成唯一的请求 ID (echo)
    #     echo_id = str(uuid.uuid4())
        
    #     # 2. 构建完整的 OneBot 命令格式
    #     request_data = {
    #         "action": command,
    #         "params": params,
    #         "echo": echo_id
    #     }
        
    #     # 3. 在 ResponsePool 中设置一个 Future 等待
    #     response_future = self.response_pool.wait_for_response(echo_id)
        
    #     try:
    #         # 4. 发送请求
    #         await websocket_to_send.send(json.dumps(request_data))
            
    #         # 5. 阻塞等待 ResponsePool 返回结果
    #         response_raw = await asyncio.wait_for(response_future, timeout=timeout)
            
    #         # 6. 检查 API 状态
    #         if response_raw.get("status") == "ok":
    #             return response_raw.get("data", {})
    #         else:
    #             # 平台返回失败状态
    #             raise RuntimeError(f"NapCat API 调用失败: {response_raw.get('msg', '未知错误')}")
                
    #     except TimeoutError:
    #         raise TimeoutError(f"等待命令 '{command}' 响应超时 ({timeout}s)。")
            
    #     finally:
    #         # 7. 清理 ResponsePool 中的等待状态
    #         self.response_pool.cancel_response_wait(echo_id)