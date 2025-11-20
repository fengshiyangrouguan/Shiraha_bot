import asyncio
import json
import time
import websockets as Server
from typing import Any, Dict, List

# 导入基类和事件对象
from src.platform.platform_base import PlatformAdapterBase, PostMethod
from src.core.event_manager import BaseEvent

# 导入日志
import logging
logger = logging.getLogger(__name__)

# 定义可配置的常量
RECONNECT_DELAY = 30  # 秒

class QQNapcatAdapter(PlatformAdapterBase):
    """
    使用Napcat的QQ 平台适配器。
    - 作为反向 WebSocket 服务器运行。
    - 对消息进行标准化处理。
    - 处理心跳事件并实现自动重连。
    """

    def __init__(self, post_method: PostMethod, platform_config: Dict[str, Any]):
        super().__init__(post_method, platform_config)
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 8080)
        
        # 状态管理
        self._server_task: asyncio.Task | None = None
        self._websocket: Server.ServerConnection | None = None
        self._heartbeat_checker_task: asyncio.Task | None = None
        self.last_heartbeat: float = 0.0
        self.heartbeat_interval: float = 30.0 # 默认30秒，心跳事件会更新此值

    def run(self) -> asyncio.Task:
        """
        启动 WebSocket 服务器并返回其任务。
        """
        adapter_id = self.get_metadata()["id"]
        logger.info(f"准备启动 QQ_Napcat 适配器 '{adapter_id}'，监听地址: ws://{self.host}:{self.port}")
        self._is_running = True
        self._server_task = asyncio.create_task(self._run_server_with_reconnect())
        return self._server_task

    async def _run_server_with_reconnect(self):
        """
        运行服务器的核心循环，包含自动重连逻辑。
        """
        adapter_id = self.get_metadata()["id"]
        while self._is_running:
            try:
                logger.info(f"正在启动 QQ_Napcat 服务器 '{adapter_id}'...")
                # 使用 async with 确保服务器在退出时被正确关闭
                async with Server.serve(self._connection_handler, self.host, self.port, max_size=2**26) as server:
                    # 等待服务器被关闭。正常情况下，这会一直阻塞。
                    # 如果连接处理器完成（例如，客户端断开），它也会完成。
                    await server.wait_closed()
            except (OSError, Server.exceptions.WebSocketException) as e:
                logger.error(f"QQ_Napcat 服务器 '{adapter_id}' 出现网络或启动错误: {e}", exc_info=True)
                # 报告连接失败事件
                self.commit_event(BaseEvent(
                    event_type="platform_connection_failed",
                    source=adapter_id,
                    data={"adapter_id": adapter_id, "error": str(e)}
                ))
            except Exception as e:
                logger.error(f"QQ_Napcat 服务器 '{adapter_id}' 出现无法恢复的错误: {e}", exc_info=True)

            if not self._is_running:
                break  # 如果是正常停止，则退出循环

            logger.info(f"连接已断开或服务器已停止，将在 {RECONNECT_DELAY} 秒后尝试重连...")
            # 报告断开连接事件
            # self.commit_event(BaseEvent(
            #     event_type="platform_disconnected",
            #     source=adapter_id,
            #     data={"adapter_id": adapter_id, "reconnect_delay": RECONNECT_DELAY}
            # ))
            await asyncio.sleep(RECONNECT_DELAY)

    async def _connection_handler(self, websocket: Server.ServerConnection):
        """
        处理单个 WebSocket 连接。
        """
        self._websocket = websocket
        adapter_id = self.get_metadata()["id"]
        logger.info(f"QQ_Napcat 客户端已连接到适配器 '{adapter_id}' ({websocket.remote_address})")

        final_code = 1000 # 默认正常关闭
        final_reason = "Normal Closure"

        try:
            async for raw_message in websocket:
                try:
                    event_data = json.loads(raw_message)
                    post_type = event_data.get("post_type")

                    if post_type == "message":
                        standardized_data = self._standardize_message_event(event_data)
                        event = BaseEvent(
                            event_type="message",
                            source=adapter_id,
                            data=standardized_data
                        )
                        self.commit_event(event)
                    elif post_type == "notice":
                        # 可以在这里添加 _standardize_notice_event
                        logger.debug(f"收到通知事件: {event_data}")
                    elif post_type == "meta_event":
                        await self._handle_meta_event(event_data)

                except json.JSONDecodeError:
                    logger.warning(f"收到了无法解析的 JSON 数据: {raw_message}")
                except Exception as e:
                    logger.error(f"处理接收到的事件时出错: {e}", exc_info=True)

        except asyncio.CancelledError:
            # Ctrl+C 或 terminate() 触发
            final_code = 4000
            final_reason = "Connection cancelled by adapter shutdown"
            logger.info(f"客户端 {adapter_id} 连接处理被取消 (主动关闭)。")

        except Server.exceptions.ConnectionClosed as e:
            # 客户端或网络主动断开
            final_code = e.code
            final_reason = e.reason or "Client Disconnected"
            logger.warning(f"OneBot V11 客户端连接已关闭: {websocket.remote_address} (Code: {final_code}, Reason: {final_reason})")
        
        except Exception as e:
            # 捕获其他意外错误
            final_code = 1011 # 内部错误
            final_reason = f"Unexpected Internal Error: {type(e).__name__}"
            logger.error(f"处理客户端 {adapter_id} 时出现意外错误: {e}", exc_info=True)

        finally:
            self._websocket = None
            # 取消心跳检查任务
            if self._heartbeat_checker_task and not self._heartbeat_checker_task.done():
                self._heartbeat_checker_task.cancel()
            
            # 报告连接已关闭事件 (现在使用已赋值的 final_code/reason)
            self.commit_event(BaseEvent(
                event_type="platform_disconnected",
                source=adapter_id,
                data={"adapter_id": adapter_id, "reason": f"连接关闭 (Code: {final_code}) - {final_reason}"}
            ))

    def _standardize_message_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        将原始的 OneBot 消息事件标准化为类似 MaiBot 的JSON结构。
        这是一个简化版本，重点在于结构对齐。
        """
        sender = raw_event.get("sender", {})
        user_info = {
            "platform": "onebot_v11",
            "user_id": str(raw_event.get("user_id")),
            "user_nickname": sender.get("nickname"),
            "user_cardname": sender.get("card"),
        }

        group_info = None
        if raw_event.get("message_type") == "group":
            group_info = {
                "platform": "onebot_v11",
                "group_id": str(raw_event.get("group_id")),
                "group_name": None, # 原始事件中通常不带，需要时需主动调用API获取
            }

        message_info = {
            "platform": "onebot_v11",
            "message_id": str(raw_event.get("message_id")),
            "time": raw_event.get("time"),
            "user_info": user_info,
            "group_info": group_info,
        }

        # 简化版消息段处理
        segments = []
        for seg in raw_event.get("message", []):
            seg_type = seg.get("type")
            seg_data = seg.get("data", {})
            if seg_type == "text":
                segments.append({"type": "text", "data": seg_data.get("text")})
            elif seg_type == "image":
                segments.append({"type": "image", "data": {"url": seg_data.get("url")}}) # 仅保留URL
            elif seg_type == "at":
                segments.append({"type": "at", "data": {"qq": seg_data.get("qq")}})
            # 其他类型可以按需添加

        return {
            "message_info": message_info,
            "message_segment": {"type": "seglist", "data": segments},
            "raw_message": raw_event.get("raw_message"),
        }

    async def _handle_meta_event(self, meta_event: Dict[str, Any]):
        """
        处理元事件，特别是生命周期和心跳。
        """
        meta_event_type = meta_event.get("meta_event_type")
        adapter_id = self.get_metadata()["id"]

        if meta_event_type == "lifecycle" and meta_event.get("sub_type") == "connect":
            self_id = meta_event.get('self_id')
            logger.info(f"适配器 '{adapter_id}' 已成功连接到 QQ_Napcat 客户端 (self_id: {self_id})。")
            # 报告已连接事件
            self.commit_event(BaseEvent(
                event_type="platform_connected",
                source=adapter_id,
                data={"adapter_id": adapter_id, "self_id": self_id}
            ))
            
            self.last_heartbeat = time.time()
            # 如果心跳检查任务没有运行，则启动它
            if not self._heartbeat_checker_task or self._heartbeat_checker_task.done():
                self._heartbeat_checker_task = asyncio.create_task(self._check_heartbeat_loop())
        

        elif meta_event_type == "heartbeat":
            status = meta_event.get("status", {})
            if status.get("online") and status.get("good"):
                self.last_heartbeat = time.time()
                # 从心跳事件中获取并更新心跳间隔
                self.heartbeat_interval = meta_event.get("interval", 15000) / 1000.0
                logger.debug("收到正常心跳。")
            else:
                logger.warning(f"收到异常心跳: {meta_event}。将关闭连接以触发重连。")
                if self._websocket:
                    await self._websocket.close()

    async def _check_heartbeat_loop(self):
        """
        一个后台任务，主动检查心跳是否超时。
        """
        adapter_id = self.get_metadata()["id"]
        logger.info(f"心跳主动检查任务已启动 for '{adapter_id}'.")
        while self._is_running and self._websocket:
            # 睡眠时间放心跳间隔的1.5倍，提供一些缓冲
            await asyncio.sleep(self.heartbeat_interval * 1.5)
            
            if not self._is_running or not self._websocket:
                break

            if time.time() - self.last_heartbeat > self.heartbeat_interval * 2:
                logger.error(f"心跳超时！适配器 '{adapter_id}' 将关闭连接以触发重连。")
                # 关闭连接，外层的重连循环会负责重启
                await self._websocket.close()
                break # 退出检查循环
        logger.info(f"心跳主动检查任务已停止 for '{adapter_id}'.")

    async def terminate(self):
        """
        停止 WebSocket 服务器和所有相关任务。
        """
        if not self._is_running:
            return
            
        adapter_id = self.get_metadata()["id"]
        logger.info(f"正在停止 QQ_Napcat 适配器 '{adapter_id}'...")
        self._is_running = False # 设置标志以停止重连循环
        
        if self._websocket and self._websocket.open:
            await self._websocket.close()
        
        if self._heartbeat_checker_task and not self._heartbeat_checker_task.done():
            self._heartbeat_checker_task.cancel()

        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass # 意料之中的取消异常

        logger.info(f"QQ Napcat 适配器 '{adapter_id}' 已成功停止。")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "qq_napcat",
            "description": "使用Napcat的QQ 平台适配器，具备心跳检测和自动重连功能。",
            "id": self.config.get("id", "default_qq_napcat")
        }

    async def send_action(self, action: Dict[str, Any]):
        if self._websocket and self._websocket.open:
            try:
                await self._websocket.send(json.dumps(action))
            except Exception as e:
                logger.error(f"发送动作失败: {e}", exc_info=True)
        else:
            logger.error("无法发送动作：无活动的 WebSocket 连接。")
