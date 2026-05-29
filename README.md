# Genie — Local AI Chatbot

A fully local, private AI chatbot that runs entirely on your machine. No API keys, no internet connection required, no data sent to the cloud.

![Genie](frontend/The_Genie_Aladdin.png)

---

## Features

- **100% local & private** — all inference runs on your CPU via `llama-cpp-python`
- **Persistent memory** — Genie remembers facts about you across conversations (your name, interests, etc.)
- **Multi-session chat** — create and manage multiple chat sessions, all saved to disk
- **Automatic context compression** — long conversations are summarized in the background so the context window never overflows
- **File uploads** — attach images, PDFs, `.txt`, `.doc`, and `.docx` files and ask questions about them
- **Vision support** — uses Qwen2.5-VL (3B) to describe image content; falls back to Tesseract OCR for text extraction
- **Math rendering** — KaTeX renders LaTeX math expressions in the browser
- **Streaming responses** — replies appear token-by-token via Server-Sent Events
- **Adjustable settings** — temperature, max tokens, system prompt, and context window are all configurable from the UI

---

## Requirements

- Python 3.10 or later
- Windows (the provided `run.bat` targets Windows; Linux/macOS users can adapt the commands)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (for image text extraction)
- At least 8 GB RAM recommended for the default model

---

## Project Structure

```
project/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, file handling
│   ├── chat_engine.py       # Session management, memory, compression
│   ├── model_manager.py     # llama-cpp-python wrapper, streaming
│   ├── image_processing.py  # Tesseract OCR preprocessing
│   ├── vision_processor.py  # Qwen2.5-VL image understanding
│   ├── requirements.txt
│   ├── user_facts.json      # Persisted user facts (auto-generated)
│   └── sessions/            # Per-session chat history (auto-generated)
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── The_Genie_Aladdin.png
├── models/
│   └── google_gemma-4-E2B-it-Q4_K_M.gguf   # Model file (download separately)
├── run.bat
└── README.md
```

---

## Setup

**1. Download the model**

Download `google_gemma-4-E2B-it-Q4_K_M.gguf` and place it in the `models/` directory. The path must match exactly:

```
models/google_gemma-4-E2B-it-Q4_K_M.gguf
```

**2. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

> Note: `llama-cpp-python` may require a C++ compiler. See the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python) for platform-specific instructions.

**3. Run the server**

On Windows, double-click `run.bat` or run it from a terminal:

```bat
run.bat
```

This installs dependencies, starts the FastAPI server on port 8000, and opens the browser automatically.

To start manually:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000/app](http://localhost:8000/app) in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check and model status |
| `GET` | `/api/status` | Model status and active session count |
| `POST` | `/api/chat` | Send a message (sync, returns full response) |
| `POST` | `/api/chat/stream` | Send a message (streaming SSE response) |
| `POST` | `/api/upload` | Upload a file to attach to the next message |
| `GET` | `/api/sessions` | List all chat sessions |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/settings` | Update model settings (temperature, tokens, etc.) |

---

## Configuration

Model settings can be changed via the Settings panel in the UI or by calling `POST /api/settings`. Available options:

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `temperature` | `1.0` | `0.0 – 2.0` | Randomness of responses |
| `max_tokens` | `2048` | `1 – 8192` | Maximum response length |
| `top_p` | `0.95` | `0.0 – 1.0` | Nucleus sampling threshold |
| `top_k` | `64` | `1 – 200` | Top-k sampling |
| `n_ctx` | `8192` | — | Model context window size |

---

## Memory & Persistence

Genie stores two kinds of data:

- **User facts** (`backend/user_facts.json`) — personal details extracted from conversation (name, interests, etc.). These persist across all sessions and are injected into every system prompt.
- **Chat sessions** (`backend/sessions/*.json`) — full conversation history for each session, restored automatically on server restart.

Context compression runs automatically in a background thread when a conversation exceeds 24 messages. It summarizes older messages and keeps the 4 most recent exchanges verbatim.

---

## Supported File Types

| Type | How it's handled |
|------|-----------------|
| Images (PNG, JPG, WEBP, BMP, GIF) | Qwen2.5-VL describes the image; Tesseract OCR extracts any text |
| PDF | PyPDF2 text extraction; falls back to Tesseract OCR for scanned PDFs |
| DOCX | python-docx with ZIP/XML fallback for tables, headers, and footers |
| DOC | olefile-based extraction |
| TXT | Read directly |

---

## Troubleshooting

**Model not loading**
Verify the model file exists at `models/google_gemma-4-E2B-it-Q4_K_M.gguf`. Check `server.log` for error details.

**Slow responses**
The model runs on CPU by default. Response speed depends on your hardware. Lowering `max_tokens` or `n_ctx` in settings can significantly improve speed.

**OCR not working**
Ensure Tesseract is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`. Update the path in `backend/image_processing.py` if your installation is in a different location.

**Vision model slow to load**
Qwen2.5-VL (~3B parameters) downloads approximately 6 GB on first use and runs on CPU. It is loaded lazily — only when an image is first uploaded.

**Port 8000 already in use**
Change the port in `run.bat`:
```bat
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

---

