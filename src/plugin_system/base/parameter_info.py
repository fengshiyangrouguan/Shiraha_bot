from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = False
    choices: Optional[List[str]] = None
