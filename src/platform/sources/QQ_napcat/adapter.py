import asyncio
import json
import time
import functools
import websockets as Server
from typing import Any, Dict, List, Optional
import logging

# 导入适配器基类和新的标准化事件模型以及实体类
from src.platform.platform_base import PlatformAdapterBase, PostMethod
from src.common.event_model.event import Event
from src.platform.sources.QQ_napcat.utils import parse

logger = logging.getLogger(__name__)

# 定义可配置的常量
RECONNECT_DELAY = 30  # 秒，连接断开后尝试重连的延迟时间

class QQNapcatAdapter(PlatformAdapterBase):
    """
    使用Napcat的QQ 平台适配器。
    - 作为反向 WebSocket 服务器运行。
    - 对消息进行标准化处理。
    - 处理心跳事件并实现自动重连。
    """

    def __init__(self, post_method: PostMethod, platform_config: Dict[str, Any]):
        """
        初始化 QQNapcatAdapter。

        Args:
            post_method (PostMethod): 一个异步函数，用于将事件提交给 EventManager。
                                      这个方法通常是 `main_system.event_manager.post`。
            platform_config (Dict[str, Any]): 该适配器的配置信息，包括 host, port, id 等。
        """
        super().__init__(post_method, platform_config)
        self.host = self.config.get("host", "127.0.0.1")  # 监听地址
        self.port = self.config.get("port", 8080)        # 监听端口
        
        # 状态管理
        self._server_task: asyncio.Task | None = None          # WebSocket 服务器的运行任务
        self._websocket: Server.WebSocketServerProtocol | None = None # 当前活动的 WebSocket 连接
        self._heartbeat_checker_task: asyncio.Task | None = None # 心跳检测任务
        self.last_heartbeat: float = 0.0                       # 最后一次收到心跳的时间戳
        self.heartbeat_interval: float = 30.0                  # 心跳间隔 (秒)，从 Napcat 的心跳事件中更新

    def run(self) -> asyncio.Task:
        """
        启动适配器，开始监听 WebSocket 连接。
        """
        adapter_id = self.get_metadata()["id"]
        logger.info(f"准备启动 QQ_Napcat 适配器 '{adapter_id}'，监听地址: ws://{self.host}:{self.port}")
        self._is_running = True  # 设置运行标志
        self._server_task = asyncio.create_task(self._run_server_with_reconnect())
        return self._server_task

    async def _run_server_with_reconnect(self):
        """
        运行 WebSocket 服务器的核心循环，包含自动重连逻辑。

        """
        adapter_id = self.get_metadata()["id"]
        while self._is_running:
            try:
                logger.info(f"正在启动 QQ_Napcat 服务器 '{adapter_id}'...")
                # 使用 async with 确保服务器在退出时被正确关闭
                async with Server.serve(self._connection_handler, self.host, self.port, max_size=2**26) as server:
                    # 等待服务器关闭。在正常连接时，此行会一直阻塞。
                    await server.wait_closed()
            except (OSError, Server.exceptions.WebSocketException) as e:
                # 捕获网络或 WebSocket 相关的启动/运行错误
                logger.error(f"QQ_Napcat 服务器 '{adapter_id}' 出现网络或启动错误: {e}", exc_info=True)
                # 报告连接失败事件，允许 MainSystem 的 EventManager 处理
                # self.commit_event(Event(
                #     event_type="platform_connection_failed",
                #     platform=adapter_id,
                #     event_data={"adapter_id": adapter_id, "error": str(e)}
                # ))
            except Exception as e:
                # 捕获其他无法恢复的通用错误
                logger.error(f"QQ_Napcat 服务器 '{adapter_id}' 出现无法恢复的错误: {e}", exc_info=True)

            if not self._is_running:
                break  # 如果适配器被请求停止，则退出重连循环

            logger.info(f"连接已断开或服务器已停止，将在 {RECONNECT_DELAY} 秒后尝试重连...")
            # 报告断开连接事件
            # self.commit_event(BaseEvent(
            #     event_type="platform_disconnected",
            #     source=adapter_id,
            #     data={"adapter_id": adapter_id, "reconnect_delay": RECONNECT_DELAY}
            # ))
            # await asyncio.sleep(RECONNECT_DELAY) # 等待指定时间后尝试重连

    async def _connection_handler(self, websocket: Server.WebSocketServerProtocol):
        """
        处理单个 WebSocket 连接。
        """
        self._websocket = websocket # 存储当前活动的 WebSocket 连接
        adapter_id = self.get_metadata()["id"]
        logger.info(f"QQ_Napcat 客户端已连接到适配器 '{adapter_id}' ({websocket.remote_address})。")

        final_code = 1000 # 默认正常关闭
        final_reason = "Normal Closure"

        try:
            # 持续接收来自客户端的原始消息
            async for raw_message in websocket:
                try:
                    event_dict: dict = json.loads(raw_message) # 解析 JSON 格式的 OneBot V11 事件
                    post_type = event_dict.get("post_type")

                    # 用于临时研究napcat输出格式用的
                    print(json.dumps(event_dict, ensure_ascii=False, indent=2))

                    if post_type == "message":
                        # 解析为message类
                        message_data = parse._parse_message_data(event_dict)
                        user_info = parse._parse_user_info(event_dict)
                        conversation_info = parse._parse_conversation_info(event_dict)

                        # 创建 Event，将 message_data、conversation_info、user_info 放入 event_data
                        event = Event(
                            event_type="message",
                            event_id=str(object=event_dict.get("message_id", time.time())),
                            time=event_dict.get("time", int(time.time())),
                            platform="qq",
                            conversation_info=conversation_info,
                            user_info=user_info,
                            event_data=message_data
                        )
                        
                        # 将完整的 Event 提交到中央事件队列
                        self.commit_event(event)

                    elif post_type == "notice":
                        # 未来可以在这里标准化并提交通知事件
                        logger.debug(f"收到通知事件: {event_dict}")
                        # 提交一个 Event
                        # self.commit_event(BaseEvent("notice", adapter_id, event_data))
                    elif post_type == "meta_event":
                        # 处理元事件（如心跳、生命周期事件）
                        # 不进入 Event 队列
                        await self._handle_meta_event(event_dict)

                except json.JSONDecodeError:
                    logger.warning(f"收到了无法解析的 JSON 数据: {raw_message}")
                except Exception as e:
                    logger.error(f"处理接收到的事件时出错: {e}", exc_info=True)

        except asyncio.CancelledError:
            # Ctrl+C 或 terminate() 触发
            final_code = 4000
            final_reason = "Connection cancelled by adapter shutdown"
            logger.info(f"客户端 {adapter_id} 连接处理任务被取消。")
        
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
            self._websocket = None # 清除当前 WebSocket 连接引用
            # 取消心跳检查任务
            if self._heartbeat_checker_task and not self._heartbeat_checker_task.done():
                self._heartbeat_checker_task.cancel()
            
            # 报告连接已关闭事件 (现在使用已赋值的 final_code/reason)
            # self.commit_event(BaseEvent(
    
    async def _send_reply(self, user_id: str, group_id: Optional[str], segments: List[Dict[str, Any]], at_sender: bool):
        """
        异步方法：发送回复消息的辅助方法。
        这个方法被 `functools.partial` 包装后作为 `BaseEvent` 的回复函数。
        它负责根据消息类型（群聊或私聊）构造 OneBot V11 的动作并发送。

        Args:
            user_id (str): 消息接收者的用户ID。
            group_id (Optional[str]): 如果是群聊，则为群组ID；私聊则为 None。
            segments (List[Dict[str, Any]]): 要发送的消息段列表。
            at_sender (bool): 在群聊中是否 @ 消息发送者。
        """
        params = {"message": segments} # 消息参数
        action_name: Optional[str] = None # 动作名称

        if group_id: # 如果是群聊消息
            action_name = "send_group_msg"
            params["group_id"] = group_id
            if at_sender:
                # 在消息段列表开头添加 @ 发送者的消息段
                at_segment = {"type": "at", "data": {"qq": user_id}}
                params["message"].insert(0, at_segment)
        elif user_id: # 如果是私聊消息 (或私聊回复)
            action_name = "send_private_msg"
            params["user_id"] = user_id
        
        if not action_name:
            logger.error(f"无法确定回复类型（私聊或群聊），user_id={user_id}, group_id={group_id}。")
            return

        await self.send_action({"action": action_name, "params": params})



    async def _handle_meta_event(self, meta_event: Dict[str, Any]):
        """
        异步方法：处理来自 Napcat 的元事件（meta_event）。
        处理连接生命周期事件和心跳事件。
        """
        meta_event_type = meta_event.get("meta_event_type")
        adapter_id = self.get_metadata()["id"]

        if meta_event_type == "lifecycle" and meta_event.get("sub_type") == "connect":
            self_id = meta_event.get('self_id')
            logger.info(f"适配器 '{adapter_id}' 已成功连接到 QQ_Napcat 客户端 (self_id: {self_id})。")
            # 报告已连接事件
            # self.commit_event(BaseEvent(
            #     event_type="platform_connected",
            #     source=adapter_id,
            #     data={"adapter_id": adapter_id, "self_id": self_id}
            # ))
            
            self.last_heartbeat = time.time()
            # 如果心跳检查任务没有运行，则启动它
            if not self._heartbeat_checker_task or self._heartbeat_checker_task.done():
                self._heartbeat_checker_task = asyncio.create_task(self._check_heartbeat_loop())
        

        elif meta_event_type == "heartbeat":
            status = meta_event.get("status", {})
            if status.get("online") and status.get("good"):
                self.last_heartbeat = time.time()
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

    