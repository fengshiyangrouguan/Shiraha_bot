import asyncio
import csv
import random
from pathlib import Path
import os
from typing import Dict, Any, List
from src.common.logger import get_logger
from src.plugins.rpg_plugin.sub_agent.rpg_replyer import RPGReplyer
from src.plugins.rpg_plugin.sub_agent.planner import SubPlanner
from src.extensions.skill_framework.skill_box import SkillBox

from src.cortices.qq_chat.data_model.chat_stream import QQChatStream
logger = get_logger("rpg_agent")

# 核心配置：动态身份模板
RPG_PHASE_CONFIGS = {
    "COLLECTING_PLAYERS": {
        "identity": "跑团招募官",
        "style": "热情、富有号召力",
        "task": "参考【世界观背景】，引导对跑团感兴趣的人扣1，你要用符合背景设定的口吻来欢迎新玩家。",
        "action_limit": "使用日常口语化的回复，尽量使用短句，不描述动作（例如不要写“我摇了摇头”等）",
    },
    "CHARACTER_CREATION": {
        "identity": "命运引导者",
        "style": "神秘、严谨、循循善诱",
        "task": "你正在引导玩家创建角色。根据玩家给出的职业，给出富有代入感的点评。每次只针对一个玩家进行互动。",

    },
    "GAMING": {
        "identity": "游戏主持人 (DM)",
        "style": "叙述感强、平稳但充满戏剧性",
        "task": "根据 Planner 的判定结果进行演化描述。禁止描述玩家的动作，只描述世界的反馈。",

    }
}

class RPGAgent:
    def __init__(self, chat_stream:QQChatStream):
        self.chat_stream = chat_stream
        self.planner = SubPlanner()
        self.replyer = RPGReplyer(chat_stream)
        current_file = Path(__file__).resolve()

        self.skill_pool_path = current_file.parent / "skills_pool"
        self.skill_box = SkillBox(base_path=str(self.skill_pool_path))
        
        # --- 核心状态控制 ---
        self.phase = "COLLECTING_PLAYERS"  # 初始阶段：招募玩家
        self.is_running = True
        self.registered_players = {}  # 存储提取出的玩家信息
        self.world_config = {} # 存储导入的世界观

    async def run_session(self, reason:str):
        """
        RPG Agent 的事件驱动主循环
        """
        self.is_running = True
        event_signal = self.chat_stream.get_new_message_event()

        # 先标记以前的消息都是已读
        self.chat_stream.mark_as_replyed()
        
        world_intro = await self.skill_box.read_skill_file(
            skill_id="collect_players", 
            rel_path="references/world_summary.md"
        )
        
        # 初始招募广播
        self.current_lore_summary = world_intro

        await self.replyer.execute_performance(
            identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
            intent=f"（简短的几条消息）发布招募公告，说明跑团招募开始，想玩的扣1，限时1分钟，游戏内容参考，可适当引用勾起玩家好奇心：\n{world_intro}"
        )
        

        # 2. 状态机循环
        while self.is_running:
            if self.phase == "COLLECTING_PLAYERS":
                await self._phase_collect_players()

            elif self.phase == "CHARACTER_CREATION":
                await self._phase_create_characters()
            
            elif self.phase == "GAMING":
                await self._main_game_loop()
            
            elif self.phase == "FINISHED":
                self.is_running = False
            
            # 每一轮 Phase 切换时稍微喘息，防止死循环过快
            await asyncio.sleep(0.1)
        
        return

    # --- 阶段处理函数 ---
    async def _phase_collect_players(self):
        """阶段 1: 异步信号驱动的招募逻辑"""
        start_time = asyncio.get_event_loop().time()
        total_wait = 60.0
        reminded = False
        event_signal = self.chat_stream.get_new_message_event()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            remaining = total_wait - elapsed

            if remaining <= 0:
                break # 时间到，跳出循环

            try:
                # 核心：等待新消息信号，但最多只等剩余的时间
                # 如果 5 秒内没新消息，也会抛出 TimeoutError 让循环继续跑一遍检查时间
                await asyncio.wait_for(event_signal.wait(), timeout=min(remaining, 5.0))
                event_signal.clear()
                
                # --- 处理新消息 (即时回复) ---
                await self._handle_registration_step()

            except asyncio.TimeoutError:
                # --- 处理定时检查 (提醒逻辑) ---
                if not reminded and elapsed > 30:
                    reason_text = "目前还没人报名，语气凄凉地询问是否有人参加" if not self.registered_players else "询问还有没有人要参加，并说明时间过半，报名马上要截止了"
                    await self.replyer.execute_performance(
                        identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
                        intent=reason_text,
                    )
                    reminded = True

        # --- 退出循环后的判定 ---
        if not self.registered_players:
            await self.replyer.execute_performance(
                identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
                intent="由于没人报名，遗憾地宣布取消本次跑团",
            )
            self.is_running = False
            return

        # 正常结束
        player_names = [info["player_name"] for info in self.registered_players.values()]
        await self.replyer.execute_performance(
            identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
            intent=f"说明报名截止，宣布本次冒险者名单：{', '.join(player_names)}。并告知正在填入报名表，请大家耐心等待开启大门",
        )
        self.phase = "CHARACTER_CREATION"

    async def _phase_create_characters(self):
        """阶段 2: Planner 驱动的角色创建"""
        creator_guide = await self.skill_box.load_skill("character_creator")
        
        for player_id, nickname in self.registered_players.items():
            # 1. 引导：由 Replyer 发起带感的开场
            await self.replyer.execute_performance(
                identity_config=RPG_PHASE_CONFIGS["CHARACTER_CREATION"],
                intent=f"引导玩家 @{nickname} 开始创造角色。说明他们需要选择职业、主武器1件，副武器一件，和一件遗物。",
            )

            # 2. 该玩家的任务循环
            while True:
                # 等待特定玩家回复
                msg = await self._wait_for_player_reply(player_id)
                if not msg: continue

                # 3. 让 Planner 介入分析
                # 传入技能指南和世界观，让 Planner 决定是“查资料”还是“录入”
                plan = await self.planner.execute_planning(
                    user_input=msg.content,
                    context={
                        "player_nickname": nickname,
                        "skill_guide": creator_guide,
                        "world_summary": self.current_lore_summary
                    }
                )

                # 4. 执行 Planner 决策
                if plan.get("action") == "CALL_TOOL":
                    tool_name = plan["tool_call"]["name"]
                    args = plan["tool_call"]["args"]

                    # 处理查询或录入
                    if tool_name == "read_skill_file":
                        res = await self.skill_box.read_skill_file("character_creator", args["rel_path"])
                        # 拿到资源后，可以再次反馈给 Replyer 组织语言
                        await self.replyer.execute_performance(
                            identity_config=RPG_PHASE_CONFIGS["CHARACTER_CREATION"],
                            reason=f"根据资源内容：{res}，反馈玩家的选择是否合理并进行点评。",
                            chat_stream=self.chat_stream
                        )

                    elif tool_name == "run_skill_script":
                        # 执行录入脚本 (例如 save_character_data)
                        res = await self.skill_box.run_skill_script("character_creator", args["entry"], args["params"])
                        if res.get("success"):
                            break # 当前玩家创角成功，跳出 True 循环处理下一位

                elif plan.get("action") == "SPEAK":
                    # Planner 认为只是普通的闲聊或解释
                    await self.replyer.execute_performance(
                        identity_config=RPG_PHASE_CONFIGS["CHARACTER_CREATION"],
                        reason=f"回复玩家的询问或补充解释：{plan['response_to_user']}",
                        chat_stream=self.chat_stream
                    )

        self.phase = "GAMING"

    async def _main_game_loop(self):
        """阶段 3: Planner 驱动的正式游戏循环"""
        while self.is_running:
            latest_message = self._get_latest_message()
            if latest_message:
                # 进入你设计的 Planner 循环
                plan = self.planner.create_plan(latest_message, self.world_config)
                result = await self._execute_plan(plan)
                await self._send_to_world(result)
            
            await asyncio.sleep(1)

    # --- 工具函数 ---

    def _initialize_player_csv(self, names: List[str]):
        """根据提取的名称生成 CSV 文件"""
        file_path = f"data/sessions/{self.conversation_id}_players.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["player_id", "name", "hp", "inventory", "traits"])
            for name in names:
                # 初始数据
                self.players[name] = {"name": name, "hp": 100}
                writer.writerow([name, name, 100, "[]", ""])
        logger.info(f"CSV 玩家文件已创建: {file_path}")



    async def _handle_registration_step(self):
        """
        处理新消息流，区分‘扣1’和‘闲聊’
        """
        # 提取所有未处理的新消息
        new_msgs = [m for m in self.chat_stream.llm_context if not m.is_replyed]
        if not new_msgs:
            return

        for msg in new_msgs:
            content = msg.content.strip()
            user_id = msg.user_id

            if self.phase == "COLLECTING_PLAYERS":
                # 逻辑分支 1：精准匹配扣1
                if content == "1":
                    if user_id not in self.registered_players:
                        self.registered_players[msg.user_id] = {"player_name":msg.user_nickname, "is_replyed_again":False}
                        await self.replyer.execute_performance(
                            identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
                            intent=f"{msg.user_nickname} 扣了1，请热烈欢迎并确认登记（**只允许调用一个action，尽量简短回应！！！！**）",
                            message=msg
                        )
                        # 注册入玩家字典
                    else:
                        # 如果已经扣过1了，DM 可以调侃一下
                        # 随机生成一个 1-100 的整数
                        player_info =self.registered_players.get(msg.user_id)
                        is_replyed_again = player_info.get("is_replyed_again")
                        if random.random() < 0.5 and is_replyed_again == False:
                            dice_roll = random.randint(1, 100)

                            if dice_roll <= 30:
                                # 40% 概率：普通提醒
                                reason = f"告诉 {msg.user_nickname} 名字已经在羊皮纸上了，不用担心（**只允许调用一个action，尽量简短回应！！！！**）"
                            elif dice_roll <= 60:
                                # 40% 概率：幽默调侃
                                reason = f"调侃/威胁的语气问{msg.user_nickname} 这么急着送死吗，并告诉他名字已经登记好了（**只允许调用一个action，尽量简短回应！！！！**）"
                            else:
                                # 20% 概率：赠送小彩蛋提示
                                reason = f"告诉 {msg.user_nickname} 名字登记好了，并阴森地透露天空中里好像有个阴影在盯着他（一个恶趣味玩笑）（**只允许调用一个action，尽量简短回应！！！！**）"

                            self.registered_players[msg.user_id]["is_replyed_again"] = True

                            await self.replyer.execute_performance(
                                identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
                                intent=reason,
                                message= msg
                            )
                
                # 逻辑分支 2：其他闲聊
                else:
                    if random.random() < 0.5:
                        await self.replyer.execute_performance(
                            identity_config=RPG_PHASE_CONFIGS["COLLECTING_PLAYERS"],
                            intent=f"{msg.user_nickname} 没扣1但说了：'{content}'。以招募官身份回应他，如果有关于世界观的问题（**只允许调用一个action！！！！**），可以参考下面的世界观：\n{self.current_lore_summary}",
                            message=msg
                        )

            # 处理完后，将消息标记为已读（或者由回复器统一处理）
            msg.is_replyed = True


    async def _wait_for_player_reply(self, target_player_id: str):
        """异步等待特定玩家的消息"""
        event_signal = self.chat_stream.get_new_message_event()
        while True:
            await event_signal.wait()
            event_signal.clear()
            
            # 寻找该玩家最新的一条未处理消息
            new_msgs = [m for m in self.chat_stream.llm_context if not m.is_replyed and m.user_id == target_player_id]
            if new_msgs:
                msg = new_msgs[-1]
                msg.is_replyed = True
                return msg
            await asyncio.sleep(0.1)