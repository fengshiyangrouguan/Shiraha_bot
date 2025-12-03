import asyncio
import json
import time

import websockets as Server
from websockets.protocol import State # <-- 导入 State

from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.platform.platform_base import BasePlatformAdapter, PostMethod
from src.platform.sources.qq_napcat.service.event_dispatcher import NapcatEventDispatcher
from src.platform.sources.qq_napcat.service.message_service import NapcatMessageService
from src.platform.sources.qq_napcat.service.command_service import NapcatCommandService
from src.platform.sources.qq_napcat.config_schema import ConfigSchema

logger = get_logger("QQ Adapter")


RECONNECT_DELAY = 30  # seconds


class QQNapcatAdapter(BasePlatformAdapter): # 继承 BasePlatformAdapter
    """
    新结构的 QQ Napcat Adapter：
    - 连接管理
    - JSON 接收循环
    - 调用 EventDispatcher 解析事件
    - MessageAPI & CommandAPI 提供消息与命令能力
    """

    def __init__(self, adapter_id: str, platform_type: str, config: ConfigSchema, post_method: PostMethod): # 更新 __init__ 签名
        super().__init__(adapter_id, platform_type, config, post_method) # 调用父类构造函数

        # 直接从 config 对象获取 host 和 port
        self.config: ConfigSchema
        self.host = self.config.host
        self.port = self.config.port

        # WebSocket 状态
        self._server_task: Optional[asyncio.Task] = None
        self._websocket: Optional[Server.WebSocketServerProtocol] = None

        # 心跳检测相关
        self.last_heartbeat: float = 0
        self.heartbeat_interval: float = 30
        self._heartbeat_checker_task: Optional[asyncio.Task] = None

        # 初始化：事件接收分发器和消息/命令服务 ----
        self.dispatcher = NapcatEventDispatcher(self)
        self.message_api = NapcatMessageService(self)
        self.command_api = NapcatCommandService(self)

    # ======== 生命周期管理 ========

    def run(self) -> asyncio.Task:
        """适配器的启动入口，启动 WebSocket 服务器并开始监听连接"""
        logger.info(f"适配器[{self.adapter_id}] 启动中...")
        self._is_running = True
        self._server_task = asyncio.create_task(self._run_server())
        return self._server_task

    async def terminate(self):
        """适配器的关闭入口，关闭 WebSocket 连接并停止服务器任务"""
        logger.info(f"适配器[{self.adapter_id}] 停止中...")
        self._is_running = False

        if self._websocket and self._websocket.open:
            await self._websocket.close()

        if self._heartbeat_checker_task:
            self._heartbeat_checker_task.cancel()

        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        logger.info(f"适配器[{self.adapter_id}] 已停止")

    # ======== WebSocket server logic ========

    async def _run_server(self):
        """启动 WebSocket 服务器并处理连接重试"""
        while self._is_running:
            try:
                async with Server.serve(
                    self._connection_handler, self.host, self.port, max_size=2**26
                ) as server:
                    logger.info(f"适配器[{self.adapter_id}] 服务器启动成功，等待客户端连接...")
                    await server.wait_closed()

            except Exception as e:
                logger.error(f"适配器[{self.adapter_id}] 服务器异常：{e}", exc_info=True)
            if not self._is_running:
                break

            logger.warning(f"适配器[{self.adapter_id}] 连接断开，将在 {RECONNECT_DELAY}s 后重试")
            await asyncio.sleep(RECONNECT_DELAY)


    async def _connection_handler(self, websocket: Server.WebSocketServerProtocol):
        """处理 websocket 连接生命周期"""

        self._websocket = websocket
        logger.info(f"适配器[{self.adapter_id}] 客户端已连接至 ({websocket.remote_address})。")
        final_code = 1000 # 默认正常关闭
        final_reason = "Normal Closure"

        try:
            async for raw_text in websocket:
                await self._on_raw_message(raw_text)

        except asyncio.CancelledError:
            # Ctrl+C 或 terminate() 触发
            final_code = 4000
            final_reason = "Connection cancelled by adapter shutdown"
            logger.info(f"客户端 {self.adapter_id} 连接处理任务被取消。")
        
        except Server.exceptions.ConnectionClosed as e:
            # 客户端或网络主动断开
            final_code = e.code
            final_reason = e.reason or "Client Disconnected"
            logger.warning(f"客户端连接已关闭: {websocket.remote_address} (Code: {final_code}, Reason: {final_reason})")
        
        except Exception as e:
            # 捕获其他意外错误
            final_code = 1011 # 内部错误
            final_reason = f"Unexpected Internal Error: {type(e).__name__}"
            logger.error(f"处理客户端 {self.adapter_id} 时出现意外错误: {e}", exc_info=True)
        finally:
            self._websocket = None # 清除当前 WebSocket 连接引用
            # 取消心跳检查任务
            if self._heartbeat_checker_task and not self._heartbeat_checker_task.done():
                self._heartbeat_checker_task.cancel()        
            #TODO: 报告连接已关闭事件 (现在使用已赋值的 final_code/reason)


    # ======== 接收事件并交给 Dispatcher ========

    async def _on_raw_message(self, raw_text: str):
        """接收到 websocket 字符串后，解析 → 分发"""
        try:
            raw_event_dict = json.loads(raw_text)
        except Exception:
            logger.warning(f"适配器[{self.adapter_id}] 收到非 JSON 数据：{raw_text}")
            return

        # Napcat 原始数据打印（用于调试）
        logger.debug(json.dumps(raw_event_dict, ensure_ascii=False, indent=2))
        
        # 处理命令响应
        if "echo" in raw_event_dict:
            self.command_api.set_response(raw_event_dict["echo"], raw_event_dict)

        post_type = raw_event_dict.get("post_type")
        if post_type == "meta_event":
            # 处理元事件（如心跳、生命周期事件）
            # 不进入 Event 队列
            asyncio.create_task(self._handle_meta_event(raw_event_dict))
            return
        
        # 进入event_queue的交给 dispatcher（核心）
        asyncio.create_task(self.dispatcher.dispatch(raw_event_dict))

    # ======== 给 API 层使用的发送接口 ========

    async def _send_websocket(self, payload: Dict[str, Any]):
        """Adapter 内部统一的 websocket 发送方法（消息或命令 API 均依赖此方法）"""
        if not self._websocket:
            raise ConnectionError("WebSocket 未连接，无法发送消息")
        
        try:
            logger.debug(f"适配器[{self.adapter_id}] 发送数据: {payload}")
            await self._websocket.send(json.dumps(payload))

        except Exception as e:
            logger.error(f"适配器[{self.adapter_id}] WebSocket 发送失败：{e}", exc_info=True)

    # ======== 处理元事件 ========
    async def _handle_meta_event(self, meta_event: Dict[str, Any]):
        """
        异步方法：处理来自 Napcat 的元事件（meta_event）。
        处理连接生命周期事件和心跳事件。
        """
        meta_event_type = meta_event.get("meta_event_type")

        if meta_event_type == "lifecycle" and meta_event.get("sub_type") == "connect":
            self_id = meta_event.get('self_id')
            logger.info(f"适配器 '{self.adapter_id}' 已成功连接到 qq_napcat 客户端 (self_id: {self_id})。")
            #TODO:报告已连接事件 self.commit_event 

            self.last_heartbeat = time.time()
            # 如果心跳检查任务没有运行，则启动它
            if not self._heartbeat_checker_task or self._heartbeat_checker_task.done():
                self._heartbeat_checker_task = asyncio.create_task(self._heartbeat_loop())
        

        elif meta_event_type == "heartbeat":
            status = meta_event.get("status", {})
            
            # 如果心跳检查任务没有运行，则启动它(不知道应不应该加，先加上)
            if not self._heartbeat_checker_task or self._heartbeat_checker_task.done():
                self._heartbeat_checker_task = asyncio.create_task(self._heartbeat_loop())

            if status.get("online") and status.get("good"):
                self.last_heartbeat = time.time()
                self.heartbeat_interval = meta_event.get("interval", 15000) / 1000.0
                logger.debug(f"[QQNapcat:{self.adapter_id}] 收到正常心跳。")
            else:
                logger.warning(f"[QQNapcat:{self.adapter_id}] 收到异常心跳: {meta_event}。将关闭连接以触发重连。")
                if self._websocket:
                    await self._websocket.close()
                    

    # ======== 心跳检查 ========

    async def _heartbeat_loop(self):
        logger.info(f"适配器[{self.adapter_id}] 心跳检查任务启动")

        while self._is_running and self._websocket:
            await asyncio.sleep(self.heartbeat_interval * 1.5)

            if time.time() - self.last_heartbeat > self.heartbeat_interval * 2:
                logger.error(f"适配器[{self.adapter_id}] 心跳超时，关闭连接以重新连接")
                if self._websocket:
                    await self._websocket.close()
                break

        logger.info(f"适配器[{self.adapter_id}] 心跳检测任务结束")