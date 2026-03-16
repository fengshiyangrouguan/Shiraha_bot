from dataclasses import dataclass, field
from typing import List, Optional

from src.common.action_model.action_spec import ActionSpec


@dataclass
class ToolResult:
    success: Optional[bool] = True
    summary: Optional[str] = ""
    error_message: Optional[str] = None
    follow_up_action: List[ActionSpec] = field(default_factory=list)

    def add_action(self, action: ActionSpec):
        self.follow_up_action.append(action)
