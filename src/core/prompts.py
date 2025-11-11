# src/core/prompts.py

# 仿照MaiBot设计的、用于生成回复的Prompt模板
# 我们暂时只创建一个通用的模板，不区分群聊和私聊

REPLYER_PROMPT = """{knowledge_prompt}{tool_info_block}{expression_habits_block}{memory_block}{question_block}

对话背景:
{time_block}
{dialogue_prompt}

当前情景:
{reply_target_block}
{identity}
{mood_state}
{keywords_reaction_prompt}
{reply_style}
{moderation_prompt}

指令:
现在，请你严格按照你的身份和风格，对“当前情景”中提到的用户发言，生成一句回复。回复要自然、口语化，符合聊天场景。
只输出回复内容，不要包含任何其他多余的前后缀或解释。

{extra_info_block}
"""
