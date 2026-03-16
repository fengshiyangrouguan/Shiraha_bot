from typing import List, Optional, Dict, Any, Set
from sqlmodel import Text, Field, SQLModel, Relationship, JSON, Column, Boolean
from sqlalchemy import PickleType

class ConversationInfoDB(SQLModel, table=True):
    """Database model for ConversationInfo."""
    __tablename__ = "conversation_info"
    
    conversation_id: str = Field(primary_key=True)
    conversation_type: str
    conversation_name: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    parent_id: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    platform_id: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    platform_meta: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON),
        default=None 
    )
class UserInfoDB(SQLModel, table=True):
    """Database model for UserInfo."""
    __tablename__ = "user_info"
    
    user_id: str = Field(primary_key=True)
    user_nickname: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    user_cardname: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    

class EventDB(SQLModel, table=True):
    """Database model for an Event."""
    __tablename__ = "events"
    
    # --- Event Fields ---
    event_id: str = Field(primary_key=True)
    platform: str
    event_type: str
    time: int = Field(index=True)
    
    # --- Relationships ---
    conversation_id: Optional[str] = Field(default=None, foreign_key="conversation_info.conversation_id", index=True)
    conversation_type: str
    conversation_name: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    user_id: Optional[str] = Field(default=None, foreign_key="user_info.user_id", index=True)
    user_nickname: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    user_cardname: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    
    # --- Complex/Unstructured Data as JSON ---
    tags: Optional[List[str]] = Field(
        sa_column=Column(JSON),
        default=None 
    ) 
    event_content: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    event_metadata: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON),
        default=None 
    )

class StickerDB(SQLModel, table=True):
    """Database model for StickerInfo."""
    __tablename__ = "sticker"
    sticker_hash: str = Field(default=None, primary_key=True,)
    file_path: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=False)
    )

    file_format: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )

    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    
    emotion: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )

    embedding: Optional[List[float]] = Field(
        sa_column=Column(PickleType,nullable=True),
        default=None 
    )
    is_registered: Optional[bool] = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False)
    )
    register_time: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )

    last_used_time: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )

    usage_count: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )

class BookDB(SQLModel, table=True):
    """Database model for Book information."""
    __tablename__ = "books"
    book_title: str = Field(default=None, primary_key=True)
    format: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    registered_file_path: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    status: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    last_read_position: Optional[int] = Field(default=0)
    last_read_time: Optional[float] = Field(default=None)
    is_finished_reading: Optional[bool] = Field(default=False)


class BehaviorHistoryDB(SQLModel, table=True):
    """Long-term episodic behavior history extracted from completed action chains."""

    __tablename__ = "behavior_history"

    memory_id: str = Field(primary_key=True)
    created_at: float = Field(index=True)
    summary: str = Field(sa_column=Column(Text, nullable=False))

    motive: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    initial_plan_reason: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    source_cortex: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    scene: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    memory_type: Optional[str] = Field(
        default="behavior",
        sa_column=Column(Text, nullable=True),
    )
    conversation_id: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True, index=True),
    )

    source_tools: Optional[List[str]] = Field(
        sa_column=Column(JSON),
        default=None,
    )
    keywords: Optional[List[str]] = Field(
        sa_column=Column(JSON),
        default=None,
    )
    tags: Optional[List[str]] = Field(
        sa_column=Column(JSON),
        default=None,
    )
    importance: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )


class ExpressionPatternDB(SQLModel, table=True):
    """Expression style patterns learned from conversation history."""

    __tablename__ = "expression"

    id: Optional[int] = Field(default=None, primary_key=True)
    situation: str = Field(sa_column=Column(Text, nullable=False))
    style: str = Field(sa_column=Column(Text, nullable=False))
    count: int = Field(default=1)
    last_active_time: float = Field(index=True)
    chat_id: str = Field(index=True)
    context: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    create_date: Optional[float] = Field(default=None, index=True)
    up_content: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    content_list: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
        
    
    
