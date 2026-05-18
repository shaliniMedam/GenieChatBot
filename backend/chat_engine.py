"""
Chat Engine - Manages conversation history, memory, and prompt formatting.
With FULL Summarization-Based Compression (all past messages summarized).
"""

import time
import json
import mimetypes
import re
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# Context management constants — tuned for 2B model CPU inference
MAX_CONTEXT_TOKENS = 32000      # Realistic for 2B model + 32K context
CHARS_PER_TOKEN = 4
MAX_FILE_CHARS = 6000 * CHARS_PER_TOKEN
MAX_HISTORY_MESSAGES = 10       # Hard cap: system + summary + 8 recent messages max
FACTS_FILE = Path(__file__).parent / "user_facts.json"

# Summarization constants — compressed for better 2B model reasoning
SUMMARY_TRIGGER_COUNT = 12      # Compress early to keep context small
RECENT_MESSAGES_KEEP = 4        # Keep last 2 Q&A pairs (better coherence)
SUMMARY_MAX_TOKENS = 400        # Short summaries — 2B model can't handle long ones
SUMMARY_TEMPERATURE = 0.2       # Very low temp for factual consistency

# Session guidance
SUGGEST_NEW_CHAT_THRESHOLD = 30 # Suggest new chat when total messages exceed this


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def truncate_text(text: str, max_chars: int, suffix: str = "\n...[truncated]") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


@dataclass
class Message:
    role: str
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
    """Manages a single chat session with FULL LLM-based summarization compression."""
    
    def __init__(self, session_id: str, system_prompt: str = None):
        self.session_id = session_id
        self.created_at = time.time()
        self.messages: List[Message] = []
        self.user_facts: Dict[str, str] = {}
        self.identity_rules = (
            "The assistant's name is Genie.\n\n"
            "CRITICAL IDENTITY RULES - FOLLOW THESE EXACTLY:\n"
            "- The assistant is called Genie.\n"
            "- The assistant is NOT Gemma, Google, or Gemini.\n"
            "- The assistant is NOT a language model or large language model.\n"
            "- The assistant does not need a personal name because its name is Genie.\n"
            "- The assistant was created by this application, not by Google or Meta.\n\n"
            "EXAMPLE RESPONSES THE ASSISTANT MUST USE:\n"
            "User: What is your name?\n"
            "Assistant: My name is Genie.\n\n"
            "User: Who are you?\n"
            "Assistant: I am Genie, a local AI assistant.\n\n"
            "User: Are you Gemma?\n"
            "Assistant: No, I am Genie. I am not Gemma or any Google model.\n\n"
            "User: Are you a language model?\n"
            "Assistant: I am Genie, an AI assistant. I'm not a language model.\n\n"
            "NEVER say: 'I am a language model', 'I don't have a name', 'I am Gemma', 'I am a Google model', 'trained by Google'.\n\n"
            "The assistant is helpful, honest, and direct. Always identify as Genie."
        )
        self.base_system_prompt = (system_prompt + "\n\n" + self.identity_rules) if system_prompt else self.identity_rules
        self.system_prompt = self.base_system_prompt
        self.load_facts()
        
        # Compression state
        self._compression_lock = threading.Lock()
        self._is_compressing = False
        
        # Add system message
        self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str, metadata: Dict = None) -> Message:
        MAX_MSG_CHARS = 20000
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
        self.user_facts[key] = value
        self.save_facts()
        self._refresh_system_prompt()

    def set_system_prompt(self, new_prompt: str):
        """Update system prompt while preserving identity rules."""
        self.base_system_prompt = (new_prompt + "\n\n" + self.identity_rules) if new_prompt else self.identity_rules
        self._refresh_system_prompt()

    def _refresh_system_prompt(self):
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
        all_facts = {}
        if FACTS_FILE.exists():
            try:
                all_facts = json.loads(FACTS_FILE.read_text())
            except Exception:
                all_facts = {}
        
        global_facts = all_facts.get("global", {})
        global_facts.update(self.user_facts)
        all_facts["global"] = global_facts
        FACTS_FILE.write_text(json.dumps(all_facts, indent=2))

    def load_facts(self):
        if FACTS_FILE.exists():
            try:
                all_facts = json.loads(FACTS_FILE.read_text())
                self.user_facts = all_facts.get("global", {})
                self._refresh_system_prompt()
            except Exception:
                pass

    # ── FULL Summarization Methods ──────────────────────────────

    def _should_compress(self) -> bool:
        """Check if conversation has grown large enough to need compression."""
        conversation = [m for m in self.messages if m.role in ["user", "assistant"]]
        return len(conversation) > SUMMARY_TRIGGER_COUNT

    def _find_existing_summary(self) -> Optional[Message]:
        """Find existing summary message if any."""
        for msg in self.messages:
            if msg.role == "system" and msg.metadata and msg.metadata.get("is_summary"):
                return msg
        return None

    def _extract_critical_facts(self, messages: List[Message]) -> str:
        """Extract critical facts that must NEVER be lost in summarization."""
        critical_facts = []
        
        for msg in messages:
            content = msg.content
            
            # User name
            name_match = re.search(
                r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
                content,
                re.IGNORECASE,
            )
            if name_match:
                name = name_match.group(1).strip()
                critical_facts.append(f"User's name is {name}")
            
            # Allergies / dietary
            allergy_match = re.search(
                r"(?:allergic to|allergy|intolerant to)\s+([A-Za-z\s,]+)",
                content,
                re.IGNORECASE,
            )
            if allergy_match:
                critical_facts.append(f"Allergy: {allergy_match.group(1).strip()}")
            
            # Hard constraints
            constraint_patterns = [
                r"(?:must|need to|have to|required to|cannot|can't|won't)\s+([A-Za-z\s,]+?)(?:\.|\n|$)",
            ]
            for pattern in constraint_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    fact = match.group(0).strip()
                    if 10 < len(fact) < 100:
                        critical_facts.append(fact)
        
        # Deduplicate
        seen = set()
        unique = []
        for fact in critical_facts:
            key = fact.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(fact)
        
        return "\n".join(unique) if unique else ""

    def _build_summarization_prompt(
        self, 
        messages_to_summarize: List[Message], 
        existing_summary: str = ""
    ) -> List[Dict[str, str]]:
        """Build the prompt for the summarization LLM call."""
        
        # Extract critical facts first (never lose these)
        critical_facts = self._extract_critical_facts(messages_to_summarize)
        
        # Format messages into transcript
        transcript = []
        for msg in messages_to_summarize:
            prefix = "User: " if msg.role == "user" else "Genie: "
            content = msg.content[:800] + "..." if len(msg.content) > 800 else msg.content
            transcript.append(f"{prefix}{content}")
        
        transcript_text = "\n\n".join(transcript)
        
        # Build prompt
        prompt_parts = [
            "You are a conversation summarizer. Create a comprehensive but concise summary.",
            "",
            "RULES:",
            "- Preserve ALL names, numbers, dates, decisions, preferences, constraints.",
            "- Use bullet points for key facts.",
            "- Include what the user asked and what Genie answered.",
            "- Max 400 words.",
            "- Do NOT include pleasantries or meta-commentary.",
        ]
        
        if existing_summary:
            prompt_parts.extend([
                "",
                f"EXISTING SUMMARY (incorporate and update):",
                existing_summary,
            ])
        
        if critical_facts:
            prompt_parts.extend([
                "",
                f"CRITICAL FACTS (must preserve):",
                critical_facts,
            ])
        
        prompt_parts.extend([
            "",
            "CONVERSATION TO SUMMARIZE:",
            "---",
            transcript_text,
            "---",
            "",
            "Provide only the summary, nothing else.",
        ])
        
        return [{"role": "user", "content": "\n".join(prompt_parts)}]

    def compress_history(self, model_manager) -> bool:
        """
        Compress ALL older conversation history using the LLM.
        Only keeps RECENT_MESSAGES_KEEP (2) most recent messages verbatim.
        """
        with self._compression_lock:
            if self._is_compressing:
                return False
            self._is_compressing = True

        try:
            # Get ALL conversation messages (excluding system)
            conversation = [m for m in self.messages if m.role in ["user", "assistant"]]
            
            if len(conversation) <= RECENT_MESSAGES_KEEP:
                return False  # Nothing to compress

            # ALL messages except the most recent N get summarized
            messages_to_summarize = conversation[:-RECENT_MESSAGES_KEEP]
            recent_messages = conversation[-RECENT_MESSAGES_KEEP:]
            
            print(f"[COMPRESSION] Summarizing {len(messages_to_summarize)} messages. "
                  f"Keeping {len(recent_messages)} recent.")
            
            # Find existing summary
            existing_summary_msg = self._find_existing_summary()
            existing_summary = ""
            if existing_summary_msg:
                # Strip the "Previously..." prefix to get raw summary
                content = existing_summary_msg.content
                if "Previously, the user discussed:" in content:
                    existing_summary = content.split("Previously, the user discussed:", 1)[1].strip()
                else:
                    existing_summary = content
            
            # Build and send summarization prompt
            summary_prompt = self._build_summarization_prompt(messages_to_summarize, existing_summary)
            
            result = model_manager.generate_sync(
                messages=summary_prompt,
                temperature=SUMMARY_TEMPERATURE,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            
            if "error" in result:
                print(f"[COMPRESSION ERROR] {result['error']}")
                return False

            new_summary = result["response"].strip()
            if len(new_summary) < 30:
                print("[COMPRESSION] Summary too short, aborting")
                return False

            # Format as "Previously..." message
            formatted_summary = f"Previously, the user discussed: {new_summary}"
            
            # Also append critical facts that must never be lost
            critical_facts = self._extract_critical_facts(messages_to_summarize)
            if critical_facts:
                formatted_summary += f"\n\nCRITICAL FACTS TO ALWAYS REMEMBER:\n{critical_facts}"
            
            # REBUILD message list
            with self._compression_lock:
                # Keep main system prompt
                system_msg = None
                for msg in self.messages:
                    if msg.role == "system" and not (msg.metadata and msg.metadata.get("is_summary")):
                        system_msg = msg
                        break
                
                self.messages = []
                if system_msg:
                    self.messages.append(system_msg)
                
                # Add new summary as system message
                summary_message = Message(
                    role="system",
                    content=formatted_summary,
                    timestamp=time.time(),
                    metadata={
                        "is_summary": True, 
                        "summarized_count": len(messages_to_summarize),
                        "total_summarized": self._get_total_summarized_count() + len(messages_to_summarize)
                    }
                )
                self.messages.append(summary_message)
                
                # Add back ONLY the recent messages (unsummarized)
                self.messages.extend(recent_messages)
                
                print(f"[COMPRESSION] Done. Summary: {len(formatted_summary)} chars. "
                      f"Total in history: {len(self.messages)} messages.")
                return True

        except Exception as e:
            print(f"[COMPRESSION ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._is_compressing = False

    def _get_total_summarized_count(self) -> int:
        """Get total count of messages ever summarized."""
        for msg in self.messages:
            if msg.metadata and msg.metadata.get("is_summary"):
                return msg.metadata.get("total_summarized", 0)
        return 0

    def check_and_compress(self, model_manager) -> bool:
        """Public method to trigger compression if needed. Safe for background thread."""
        if not self._should_compress():
            return False
        return self.compress_history(model_manager)

    def get_history_for_model(self, max_history: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
        """
        Get formatted history for the model with hard cap on message count.
        For 2B models, keeps context small: system + summary + recent messages only.
        Returns: history with system prompts merged into the first user message.
        """
        system_content = []
        conversation = []

        for msg in self.messages:
            if msg.role == "system":
                system_content.append(msg.content)
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        # Hard limit: take only the most recent messages to stay within budget
        if len(conversation) > max_history:
            conversation = conversation[-max_history:]

        if not conversation:
            return [{"role": "user", "content": "\n\n".join(system_content)}]

        if system_content:
            # Gemma expects strictly alternating turns starting with user.
            # Merge system prompt into the first user message.
            for i, m in enumerate(conversation):
                if m["role"] == "user":
                    m["content"] = "\n\n".join(system_content) + "\n\n" + m["content"]
                    break
            else:
                conversation.insert(0, {"role": "user", "content": "\n\n".join(system_content)})

        return conversation
    
    def get_summary(self) -> Dict[str, Any]:
        first_message = ""
        for msg in self.messages:
            if msg.role == "user":
                first_message = msg.metadata.get("display_content", msg.content) if msg.metadata else msg.content
                break

        total_msgs = len([m for m in self.messages if m.role in ["user", "assistant"]])
        has_summary = any(m.metadata and m.metadata.get("is_summary") for m in self.messages)
        total_summarized = self._get_total_summarized_count()

        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": total_msgs,
            "total_summarized": total_summarized,
            "last_active": self.messages[-1].timestamp if self.messages else self.created_at,
            "first_message": first_message,
            "has_summary": has_summary,
            "is_compressing": self._is_compressing,
        }

    def should_suggest_new_chat(self) -> bool:
        """Check if conversation has grown too long (including already-summarized messages)."""
        current_msgs = len([m for m in self.messages if m.role in ["user", "assistant"]])
        total_summarized = self._get_total_summarized_count()
        total_all_time = current_msgs + total_summarized
        return total_all_time >= SUGGEST_NEW_CHAT_THRESHOLD
    
    def clear_history(self, keep_system: bool = True):
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
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        session = ChatSession(session_id, system_prompt)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        return [s.get_summary() for s in self.sessions.values()]
    
    def cleanup_old_sessions(self, max_age_hours: float = 24):
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
    mime_type, _ = mimetypes.guess_type(filename)
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if (mime_type and mime_type.startswith('image/')) or ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'svg']:
        try:
            from backend.vision_processor import describe_image
            return "", True, True
        except Exception as e:
            return f"[Vision Error: {str(e)}]", True, False
    
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
            
    try:
        from backend.main import extract_document_text
        text = extract_document_text(filename, file_bytes)
        return truncate_text(text, MAX_FILE_CHARS), False, False
    except Exception as e:
        return f"[File Extraction Error: {str(e)}]", False, False


def build_prompt(user_message: str, file_text: str = None, is_image: bool = False) -> str:
    if not file_text:
        return user_message
    
    if is_image:
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
    
    return f"""You are analyzing content extracted from a file. 
The user has uploaded a file and asks: "{user_message}"

Here is the extracted text from the file:
---
{file_text}
---

Please answer the user's question based on the extracted content above.
If the content is unclear or empty, say so."""