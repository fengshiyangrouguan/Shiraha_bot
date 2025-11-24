import uuid
from typing import Any, Dict, Callable, List, Union
import asyncio
import logging
import json


logger = logging.getLogger(__name__)
_send_websocket: Callable[[Dict], None]

class NapcatCommandService:

    def __init__(self, adapter):
        self.adapter = adapter
        self._futures: Dict[str, asyncio.Future] = {}

    # ======== 顶层api ========
    async def get_image(self, file_id: str) -> Dict[str, Union[str, int]]:
        """
        根据文件 ID 获取图片详细信息。

        :param file_id: 文件在 OneBot 客户端内部的唯一 ID。
        :return: 包含文件详情的字典（如 url, file_size, file_name 等）。
        :raises Exception: 发送失败或客户端返回错误时，抛出异常。
        """
        
        action = "get_image" 
        params = {"file_id": file_id}

        # 1. 发起异步命令并等待响应
        # 响应结构：{"status": "ok", "retcode": 0, "data": {...}, ...}
        try:
            response = await self.send_command(action, params)
            
            # 2. 解析响应并进行错误检查
            status = response.get("status")
            retcode = response.get("retcode", -1)
            
            # 检查状态是否为 'ok' 且 retcode 是否为 0
            if status == "ok" and retcode == 0:
                # 成功：返回 data 字段中的所有内容
                data: Dict = response.get("data", {})   
                # 尝试将 file_size 转换为 int，如果失败则保持原样
                file_size_str: str = data.get("file_size")
                if file_size_str and file_size_str.isdigit():
                    data["file_size"] = int(file_size_str)
                
                return data
        
            # 3. 失败：抛出异常
            message = response.get("message", "Unknown error")
            wording = response.get("wording", "")
        except TimeoutError:
            logger.error(f"命令超时")
            return None
        except Exception as e:
            logger.error(f"命令失败: {e}")
            return None
        error_msg = f"获取文件详情失败 (Action: {action}). Code: {retcode}. Message: {message}. Wording: {wording}"
        logger.warning(error_msg)
        return None
    
    
    async def get_login_info(self) -> Dict[str, Any]:
        """
        获取机器人自身的 QQ 账号信息（QQ号、昵称等）。
        
        Args:
            无
            
        Returns:
            Dict[str, Any]: 包含 'user_id', 'nickname' 等信息的字典。
        
        Raises:
            发送失败，抛出异常。
        """
        action = "get_login_info"
        params = {}
        # 直接调用带等待机制的 send_command
        return await self.send_command(action, params)


    async def set_group_kick(self, group_id: int, user_id: int, reject_add_request: bool = False) -> None:
        """
        将指定用户踢出群组。
        
        Args:
            group_id (int): 目标群号。
            user_id (int): 目标用户 QQ 号。
            reject_add_request (bool): 可选。是否禁止该用户在被踢后再次申请入群。默认为 False。
            
        Returns:
            None: API 请求成功发出（通常不返回结果数据，只返回状态）。
            
        Raises:
            发送失败，抛出异常。
        """
        action = "set_group_kick"
        params = {
            "group_id": group_id,
            "user_id": user_id,
            "reject_add_request": reject_add_request
        }
        # 对于不需要返回数据的操作命令，可以不等待或等待简单确认
        await self.send_command(action, params)

    async def set_group_ban(self, group_id: int, user_id: int, duration: int = 30 * 60) -> None:
        """
        群组禁言指定用户。
        
        Args:
            group_id (int): 目标群号。
            user_id (int): 目标用户 QQ 号。
            duration (int): 可选。禁言时长（秒）。0 表示解除禁言。默认为 30 分钟 (1800秒)。
            
        Returns:
            None: API 请求成功发出。
            
        Raises:
            发送失败，抛出异常。
        """
        action = "set_group_ban"
        params = {
            "group_id": group_id,
            "user_id": user_id,
            "duration": duration
        }
        await self.send_command(action, params)
    
    async def set_group_whole_ban(self, group_id: int, enable: bool = True) -> None:
        """
        设置/解除群组全体禁言。
        
        Args:
            group_id (int): 目标群号。
            enable (bool): 是否开启全体禁言。True 为开启，False 为关闭。默认为 True。
            
        Returns:
            None: API 请求成功发出。
            
        Raises:
            发送失败，抛出异常。
        """
        action = "set_group_whole_ban"
        params = {
            "group_id": group_id,
            "enable": enable
        }
        await self.send_command(action, params)


    # ======== 底层api ========
    async def send_command(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发起带 echo 的指令并等待响应"""
        echo = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._futures[echo] = future

        payload = json.dumps({
            "action": action,
            "params": params,
            "echo": echo
        })

        print(f"发送命令{payload}")
        await self.adapter._send_websocket(payload)
        # 等待future类被传回响应数据赋值
        return await future

    def set_response(self, echo: str, data: Dict[str, Any]):
        '''将adapter接收到的命令响应放入等待响应的Future对象里'''

        # 从队列中取出
        future = self._futures.pop(echo, None)

        # 给future对象赋值传回的响应数据
        if future and not future.done():
            future.set_result(data)
