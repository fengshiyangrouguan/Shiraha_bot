from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExpressionPattern:
    situation: str
    style: str
    count: int
    last_active_time: float
    chat_id: str
    create_date: Optional[float]
    content_list: List[str]
    id: Optional[int] = None
    context: str = ""
    up_content: str = ""
    

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
