"""
Chat Engine - Manages conversation history, memory, and prompt formatting.
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Message:
    role: str  # system, user, assistant
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }


class ChatSession:
    """Manages a single chat session with history."""
    
    def __init__(self, session_id: str, system_prompt: str = None):
        self.session_id = session_id
        self.created_at = time.time()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt or (
            "You are Genie, a helpful, harmless, and honest AI assistant. "
            "You provide clear, accurate, and concise responses. "
            "If you're unsure about something, say so."
        )
        
        # Add system message
        self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str, metadata: Dict = None) -> Message:
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        self.messages.append(msg)
        return msg
    
    def get_history_for_model(self, max_history: int = 20) -> List[Dict[str, str]]:
        """
        Get formatted history for the model.
        Only returns the final visible answers for multi-turn (not thought blocks) [^2^].
        """
        # Filter to recent messages, keeping system + last N exchanges
        relevant = []
        for msg in self.messages:
            if msg.role == "system":
                relevant.append({"role": msg.role, "content": msg.content})
            elif msg.role in ["user", "assistant"]:
                relevant.append({"role": msg.role, "content": msg.content})
        
        # Keep last max_history messages (excluding system)
        if len(relevant) > max_history + 1:
            relevant = [relevant[0]] + relevant[-max_history:]
        
        return relevant
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        # Find first user message to use as title
        first_message = ""
        for msg in self.messages:
            if msg.role == "user":
                if msg.metadata and "display_content" in msg.metadata:
                    first_message = msg.metadata["display_content"]
                else:
                    first_message = msg.content
                break
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.messages),
            "last_active": self.messages[-1].timestamp if self.messages else self.created_at,
            "first_message": first_message,
        }
    
    def clear_history(self, keep_system: bool = True):
        """Clear chat history."""
        if keep_system and self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []
            if keep_system:
                self.add_message("system", self.system_prompt)


class ChatEngine:
    """Manages all chat sessions."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.sessions: Dict[str, ChatSession] = {}
        self.storage_path = storage_path
    
    def create_session(self, session_id: Optional[str] = None, system_prompt: str = None) -> ChatSession:
        """Create a new chat session."""
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        session = ChatSession(session_id, system_prompt)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get existing session."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return [s.get_summary() for s in self.sessions.values()]
    
    def cleanup_old_sessions(self, max_age_hours: float = 24):
        """Remove sessions older than specified hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            sid for sid, s in self.sessions.items() 
            if s.get_summary()["last_active"] < cutoff
        ]
        for sid in to_remove:
            del self.sessions[sid]
        return len(to_remove)


# Global instance
chat_engine = ChatEngine()