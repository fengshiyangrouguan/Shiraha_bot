from typing import List, Optional, Dict, Any, Set
from sqlmodel import Text, Field, SQLModel, Relationship, JSON, Column

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

