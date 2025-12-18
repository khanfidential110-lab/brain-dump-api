from pydantic import BaseModel
from typing import Literal, Optional, List, Dict

class ProcessRequest(BaseModel):
    text: str

class BrainItem(BaseModel):
    type: str # task, event, shopping, note
    content: str
    date: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    priority: str = "Medium" 
    status: str = "open" # open, done
    sentiment: str = "Neutral" # Positive, Neutral, Negative
    emoji: str = "😐" # Vibe check emoji
    due_date: Optional[str] = None # ISO 8601 for scheduling
    is_pinned: bool = False
    color_id: int = 0 # 0=Default, 1-9=Colors

class ProcessResponse(BaseModel):
    items: List[BrainItem]

class BreakdownRequest(BaseModel):
    task_content: str


class BreakdownResponse(BaseModel):
    subtasks: List[str]

class SocialRequest(BaseModel):
    content: str
    platform: str # twitter, reddit, linkedin

class AskRequest(BaseModel):
    query: str
    context_items: List[BrainItem]
    history: List[Dict[str, str]] = []

