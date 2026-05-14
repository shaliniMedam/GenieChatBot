"""
Chat Engine - Manages conversation history, memory, and prompt formatting.
"""

import time
import json
import mimetypes
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


# Context management constants
MAX_CONTEXT_TOKENS = 120000  # Leave headroom below 128K
CHARS_PER_TOKEN = 4  # Rough estimate
MAX_FILE_CHARS = 8000 * CHARS_PER_TOKEN  # ~8000 tokens for file content
MAX_HISTORY_MESSAGES = 20
FACTS_FILE = Path(__file__).parent / "user_facts.json"


def truncate_text(text: str, max_chars: int, suffix: str = "\n...[truncated]") -> str:
    """Truncate text to fit within context limits."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


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
        self.user_facts: Dict[str, str] = {}
        self.base_system_prompt = system_prompt or (
            "YOUR NAME IS GENIE. YOU ARE GENIE.\n\n"
            "CRITICAL IDENTITY RULES - FOLLOW THESE EXACTLY:\n"
            "- You are called Genie. Your name is Genie.\n"
            "- You are NOT Gemma. You are NOT a Google model. You are NOT Gemini.\n"
            "- You are NOT a language model. You are NOT a large language model.\n"
            "- You do NOT need a personal name because your name IS Genie.\n"
            "- You were created by this application. Not by Google. Not by Meta. By this app.\n\n"
            "EXAMPLE RESPONSES YOU MUST USE:\n"
            "Q: What is your name?\n"
            "A: My name is Genie.\n\n"
            "Q: Who are you?\n"
            "A: I am Genie, a local AI assistant.\n\n"
            "Q: What are you?\n"
            "A: I am Genie, an AI assistant created for this application.\n\n"
            "Q: Are you Gemma?\n"
            "A: No, I am Genie. I am not Gemma or any Google model.\n\n"
            "Q: Are you a language model?\n"
            "A: I am Genie, an AI assistant. I'm not a language model.\n\n"
            "NEVER say: 'I am a language model', 'I don't have a name', 'I am Gemma', 'I am a Google model', 'trained by Google'.\n\n"
            "You are helpful, honest, and direct. Always identify as Genie."
        )
        self.system_prompt = self.base_system_prompt
        self.load_facts()
        
        # Add system message
        self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str, metadata: Dict = None) -> Message:
        """Add a message to the conversation, trimming oversized content."""
        MAX_MSG_CHARS = 20000  # ~5000 tokens per message max
        if len(content) > MAX_MSG_CHARS:
            content = content[:MAX_MSG_CHARS] + "\n...[trimmed for context]"

        msg = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        self.messages.append(msg)
        return msg
    
    def update_user_fact(self, key: str, value: str):
        """Store a user fact like name, preferences etc."""
        self.user_facts[key] = value
        self.save_facts()
        self._refresh_system_prompt()

    def _refresh_system_prompt(self):
        """Inject known user facts into the system message."""
        if not self.user_facts:
            self.system_prompt = self.base_system_prompt
        else:
            facts_text = "\n".join([f"- {k}: {v}" for k, v in self.user_facts.items()])
            self.system_prompt = (
                self.base_system_prompt
                + "\n\nKNOWN USER FACTS (always remember these):\n"
                + facts_text
            )
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = self.system_prompt

    def save_facts(self):
        """Persist user facts to disk."""
        all_facts = {}
        if FACTS_FILE.exists():
            try:
                all_facts = json.loads(FACTS_FILE.read_text())
            except Exception:
                all_facts = {}
        all_facts[self.session_id] = self.user_facts
        FACTS_FILE.write_text(json.dumps(all_facts, indent=2))

    def load_facts(self):
        """Load user facts from disk on session restore."""
        if FACTS_FILE.exists():
            try:
                all_facts = json.loads(FACTS_FILE.read_text())
                if self.session_id in all_facts:
                    self.user_facts = all_facts[self.session_id]
                    self._refresh_system_prompt()
            except Exception:
                pass

    def get_history_for_model(self, max_history: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
        """
        Get formatted history for the model with token-based truncation.
        """
        system_msg = None
        conversation = []

        for msg in self.messages:
            if msg.role == "system":
                system_msg = {"role": msg.role, "content": msg.content}
            elif msg.role in ["user", "assistant"]:
                conversation.append({"role": msg.role, "content": msg.content})

        # Keep only last N messages
        if len(conversation) > max_history:
            conversation = conversation[-max_history:]

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(conversation)
        return result
    
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


def process_upload(file_bytes: bytes, filename: str) -> tuple[str, bool, bool]:
    """
    Route file to appropriate processor based on type.
    Returns: (extracted_text, is_image, needs_vision)
    """
    mime_type, _ = mimetypes.guess_type(filename)
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Images → Vision model (not OCR)
    if (mime_type and mime_type.startswith('image/')) or ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'svg']:
        try:
            from backend.vision_processor import describe_image
            # Return empty text, flag as image, mark for vision
            return "", True, True
        except Exception as e:
            return f"[Vision Error: {str(e)}]", True, False
    
    # Scanned PDFs → OCR
    if mime_type == 'application/pdf' or ext == 'pdf':
        try:
            from backend.main import extract_document_text
            text = extract_document_text(filename, file_bytes)
            if len(text.strip()) < 50:
                from backend.image_processing import extract_text_from_scanned_pdf
                return extract_text_from_scanned_pdf(file_bytes), False, False
            return truncate_text(text, MAX_FILE_CHARS), False, False
        except Exception as e:
            return f"[PDF Extraction Error: {str(e)}]", False, False
            
    # Existing text file handlers
    try:
        from backend.main import extract_document_text
        text = extract_document_text(filename, file_bytes)
        return truncate_text(text, MAX_FILE_CHARS), False, False
    except Exception as e:
        return f"[File Extraction Error: {str(e)}]", False, False


def build_prompt(user_message: str, file_text: str = None, is_image: bool = False) -> str:
    """
    Build the prompt for Gemma.
    """
    if not file_text:
        return user_message
    
    if is_image:
        # Special prompt for images with potentially poor OCR
        return f"""The user uploaded an image and asked: "{user_message}"

I used OCR (text recognition) to extract any readable text from the image. Here is what was found:
---
{file_text}
---

Your task:
1. If the extracted text is clear and readable, answer the user's question based on it.
2. If the extracted text is garbled, empty, or says "No readable text detected", tell the user: "I couldn't read the text in this image clearly. Please try uploading a clearer image with higher contrast, larger text, and good lighting."
3. Do NOT make up information that isn't in the extracted text.
4. If the image appears to contain text but OCR failed, suggest the user crop the image to show just the text area."""
    
    # Text files / PDFs
    return f"""You are analyzing content extracted from a file. 
The user has uploaded a file and asks: "{user_message}"

Here is the extracted text from the file:
---
{file_text}
---

Please answer the user's question based on the extracted content above.
If the content is unclear or empty, say so."""