from pydantic import BaseModel, Field
from typing import Optional, List

# [system]
class SystemConfig(BaseModel):
    version: str = "0.0.0"
    owner_id: Optional[int] = None
    log_level: str = "INFO"

# [persona]
class PersonaConfig(BaseModel):
    bot_name: str = "藤原白羽"
    bot_identity: str
    bot_personality: str
    bot_interest: List[str] = Field(default_factory=list)
    expression_style: str

# [mood]
class MoodConfig(BaseModel):
    initial_mood: str = "刚睡醒，思维缓存正在加载中，感觉有点迷糊，需要一会儿才能完全进入状态。"
    initial_energy: int = 100


# Main Config
class BotConfig(BaseModel):
    system: SystemConfig = Field(...)
    persona: PersonaConfig = Field(...)
    mood: MoodConfig = Field(...)