"""
Model Manager for Genie
Auto-loads the local GGUF model on startup.
"""

import sys
import io
# Force UTF-8 on Windows BEFORE any other imports
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        pass

import threading
from pathlib import Path
from typing import Generator, List, Dict, Any
import json

from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_FILE = "google_gemma-4-E2B-it-Q4_K_M.gguf"
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / MODEL_FILE

DEFAULT_SETTINGS = {
    "n_ctx": 32768,       # ← 32K context (your RAM can handle it)
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "repeat_penalty": 1.0,
    "max_tokens": 8192,   # ← 8K output for complete responses
    "stop": ["<turn|>", "user:", "User:", "<end_of_turn>", "<eos>"],
}

# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.settings = DEFAULT_SETTINGS.copy()
        self.model_loaded = False
        self.load_error = None
        self.loading = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Local model loading
    # ------------------------------------------------------------------

    def auto_load(self):
        """Called at server startup — loads model in a background thread."""
        thread = threading.Thread(target=self._load_blocking, daemon=True)
        thread.start()

    def _load_blocking(self):
        with self._lock:
            if self.model_loaded:
                return
            self.loading = True
            print(f"\n{'='*55}")
            print("  Auto-loading Genie...")
            print(f"{'='*55}")

            if not MODEL_PATH.exists():
                self.load_error = (
                    f"Model file not found at {MODEL_PATH}. "
                    "Please download it first."
                )
                print(f"[ERROR] {self.load_error}")
                self.loading = False
                return

            try:
                import os
                # Use physical cores only — hyperthreading hurts LLM inference
                cpu_cores = min(os.cpu_count() or 4, 8)
                self._llm = Llama(
                    model_path=str(MODEL_PATH),
                    n_ctx=32768,
                    n_threads=cpu_cores,
                    n_threads_batch=cpu_cores,
                    n_batch=2048,
                    n_ubatch=1024,
                    verbose=False,
                    n_gpu_layers=0,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format="gemma",
                )
                self.model_loaded = True
                self.load_error = None
                print("Genie loaded and ready!")
            except Exception as e:
                self.load_error = str(e)
                print(f"Failed to load model: {e}")
            finally:
                self.loading = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.model_loaded,
            "loading": self.loading,
            "model_name": "Gemma 4 E2B IT (Q4_K_M)",
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "context_size": self.settings.get("n_ctx"),
            "error": self.load_error,
        }

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        enable_thinking: bool = False,
    ) -> Generator[str, None, None]:
        """Stream a response from the model."""
        if self.loading:
            yield self._sse("error", "Model is still loading, please wait a moment.")
            return
        if not self.model_loaded or self._llm is None:
            yield self._sse("error", self.load_error or "Model not loaded.")
            return

        temp = temperature if temperature is not None else self.settings["temperature"]
        max_tok = max_tokens if max_tokens is not None else self.settings["max_tokens"]

        # If model is currently locked (e.g. background summarization), yield a message
        if self._lock.locked():
            yield self._sse("token", "\n*(Genie is organizing memories, this may take a moment...)*\n\n")

        try:
            with self._lock:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=temp,
                    top_p=self.settings["top_p"],
                    top_k=self.settings["top_k"],
                    repeat_penalty=self.settings["repeat_penalty"],
                    max_tokens=max_tok,
                    stop=self.settings["stop"],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # Clean Gemma mentions in real-time
                            cleaned = self._clean_token(content)
                            yield self._sse("token", cleaned)
                yield self._sse("done", "")
        except Exception as e:
            yield self._sse("error", f"Inference error: {str(e)}")

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        if not self.model_loaded:
            return {"error": self.load_error or "Model not loaded"}

        temp = temperature if temperature is not None else self.settings["temperature"]
        max_tok = max_tokens if max_tokens is not None else self.settings["max_tokens"]

        try:
            with self._lock:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=temp,
                    top_p=self.settings["top_p"],
                    top_k=self.settings["top_k"],
                    repeat_penalty=self.settings["repeat_penalty"],
                    max_tokens=max_tok,
                    stop=self.settings["stop"],
                    stream=False,
                )
                return {
                    "response": response["choices"][0]["message"]["content"],
                    "usage": response.get("usage", {}),
                }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------

    @staticmethod
    def _sse(event: str, data: str) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def _clean_token(token: str) -> str:
        """
        Clean a single token to remove Gemma/Google model references.
        This is called on each streamed token for real-time filtering.
        """
        import re

        # Remove control tokens immediately
        token = re.sub(r'<[^>]+>', '', token)
        token = re.sub(r'\[end_of_turn\]', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\[end_of_sequence\]', '', token, flags=re.IGNORECASE)

        # Replace problematic keywords
        token = re.sub(r'\bGemma\b', 'Genie', token, flags=re.IGNORECASE)
        token = re.sub(r'\bgemini\b', 'Genie', token, flags=re.IGNORECASE)
        token = re.sub(r'\bGoogle\b', 'Genie', token, flags=re.IGNORECASE)
        token = re.sub(r'\blanguage model\b', 'assistant', token, flags=re.IGNORECASE)
        token = re.sub(r"\bdon't have a\s+name\b", "am Genie", token, flags=re.IGNORECASE)
        token = re.sub(r"\bdon't have\s+a\s+personal\s+name\b", "am Genie", token, flags=re.IGNORECASE)

        return token


model_manager = ModelManager()