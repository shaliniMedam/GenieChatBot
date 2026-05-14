/**
 * Genie Chatbot Frontend
 * With KaTeX math rendering, proper markdown, and error handling
 */

const API_BASE = window.location.origin;

class ChatApp {
    constructor() {
        this.currentSessionId = null;
        this.isGenerating = false;
        this.modelReady = false;
        this.sessions = [];
        this.abortController = null;
        this.pendingFile = null;

        this.settings = {
            temperature: 1.0,
            maxTokens: 8192,
            systemPrompt: 'Your name is Genie. You are a helpful AI. You must NEVER say your name is Gemma. You are NOT Gemma.',
            contextWindow: 32768,
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettingsFromStorage();
        try {
            this.currentSessionId = localStorage.getItem('genie_session_id') || null;
        } catch (e) {}
        this.loadSessions();
        this.pollModelStatus();
    }

    // ── Model status polling ──────────────────────────────────────
    async pollModelStatus() {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');

        try {
            const res = await fetch(`${API_BASE}/api/status`);
            const data = await res.json();
            const model = data.model;

            if (model.loaded) {
                indicator.className = 'status-indicator online';
                statusText.textContent = 'Genie is ready';
                this.modelReady = true;
                this.updateSendButton();
                return;
            } else if (model.error) {
                indicator.className = 'status-indicator offline';
                statusText.textContent = 'Load failed';
                console.error('Model load error:', model.error);
                return;
            } else {
                indicator.className = 'status-indicator loading';
                statusText.textContent = 'Loading model...';
            }
        } catch (err) {
            indicator.className = 'status-indicator offline';
            statusText.textContent = 'Server offline';
        }
        setTimeout(() => this.pollModelStatus(), 2000);
    }

    // ── Events ───────────────────────────────────────────────────
    bindEvents() {
        const imageUpload = document.getElementById('imageUpload');
        if (imageUpload) {
            imageUpload.addEventListener('change', (e) => {
                const file = e.target.files?.[0];
                if (file) this.handleFileSelection(file);
            });
        }

        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('tempValue').textContent = parseFloat(e.target.value).toFixed(1);
        });

        document.getElementById('maxTokens').addEventListener('input', (e) => {
            document.getElementById('tokenValue').textContent = e.target.value;
        });
    }

    // ── Settings ─────────────────────────────────────────────────
    loadSettingsFromStorage() {
        try {
            const saved = localStorage.getItem('genie_settings');
            if (saved) {
                let parsed = JSON.parse(saved);
                // Aggressively overwrite old cached prompts to force the name change
                if (parsed.systemPrompt && (parsed.systemPrompt.includes('honest AI assistant') || parsed.systemPrompt.includes('helpful and honest'))) {
                    parsed.systemPrompt = 'Your name is Genie. You are a helpful AI. You must NEVER say your name is Gemma. You are NOT Gemma.';
                    localStorage.setItem('genie_settings', JSON.stringify(parsed));
                }
                this.settings = { ...this.settings, ...parsed };
            }
        } catch (e) {}
        this.applySettingsToUI();
    }

    applySettingsToUI() {
        document.getElementById('temperature').value = this.settings.temperature;
        document.getElementById('tempValue').textContent = parseFloat(this.settings.temperature).toFixed(1);
        document.getElementById('maxTokens').value = this.settings.maxTokens;
        document.getElementById('tokenValue').textContent = this.settings.maxTokens;
        document.getElementById('systemPrompt').value = this.settings.systemPrompt;
        document.getElementById('contextWindow').value = this.settings.contextWindow;
    }

    saveSettings() {
        this.settings = {
            temperature: parseFloat(document.getElementById('temperature').value),
            maxTokens: parseInt(document.getElementById('maxTokens').value),
            systemPrompt: document.getElementById('systemPrompt').value,
            contextWindow: parseInt(document.getElementById('contextWindow').value),
        };
        try {
            localStorage.setItem('genie_settings', JSON.stringify(this.settings));
        } catch (e) {}

        fetch(`${API_BASE}/api/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temperature: this.settings.temperature,
                max_tokens: this.settings.maxTokens,
            }),
        });

        this.toggleSettings();
        this.showBanner('Settings saved', 'success');
    }

    toggleSettings() {
        document.getElementById('settingsPanel').classList.toggle('open');
    }

    toggleSidebar() {
        document.getElementById('sidebar').classList.toggle('collapsed');
    }

    // ── File handling ────────────────────────────────────────────
    handleFileSelection(file) {
        // Get extension reliably from filename
        const ext = file.name.split('.').pop().toLowerCase();
        console.log('File selected:', file.name, 'Extension:', ext, 'MIME:', file.type);
        
        // Allowed extensions (more reliable than MIME types)
        const allowedExts = ['txt', 'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'];
        
        if (!allowedExts.includes(ext)) {
            this.showBanner(`Unsupported file type: .${ext}. Supported: ${allowedExts.join(', ')}`, 'error');
            return;
        }

        this.pendingFile = file;
        const previewContainer = document.getElementById('filePreviewContainer');
        const previewImg = document.getElementById('filePreviewImg');
        const previewName = document.getElementById('filePreviewName');
        
        previewContainer.style.display = 'flex';
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewImg.style.display = 'block';
                previewName.style.display = 'none';
            };
            reader.readAsDataURL(file);
        } else {
            previewImg.style.display = 'none';
            previewName.textContent = file.name;
            previewName.style.display = 'block';
        }
        
        this.updateSendButton();
        document.getElementById('messageInput').focus();
    }

    removeFile() {
        this.pendingFile = null;
        document.getElementById('filePreviewContainer').style.display = 'none';
        const imageUpload = document.getElementById('imageUpload');
        if (imageUpload) imageUpload.value = '';
        this.updateSendButton();
    }

    autoResize(element) {
        this.updateSendButton();
        element.style.height = 'auto';
        element.style.height = element.scrollHeight + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    // ── Chat ─────────────────────────────────────────────────────
    async sendMessage() {
        if (this.isGenerating) {
            this.stopGeneration();
            return;
        }

        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message && !this.pendingFile) return;
        if (!this.modelReady) {
            this.showBanner('Model is still loading, please wait...', 'warning');
            return;
        }

        if (this.pendingFile) {
            await this.uploadFile(this.pendingFile, message);
            return;
        }

        input.value = '';
        input.style.height = 'auto';
        this.updateSendButton();

        const welcome = document.getElementById('welcomeScreen');
        if (welcome) welcome.remove();

        this.addMessage('user', message);
        const typingId = this.showTypingIndicator();
        this.isGenerating = true;
        this.setSendButtonStop();
        this.abortController = new AbortController();

        try {
            const res = await fetch(`${API_BASE}/api/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: this.abortController.signal,
                body: JSON.stringify({
                    message,
                    session_id: this.currentSessionId,
                    temperature: this.settings.temperature,
                    max_tokens: this.settings.maxTokens,
                    system_prompt: this.settings.systemPrompt,
                    enable_thinking: false,
                }),
            });

            this.removeTypingIndicator(typingId);
            await this.handleStream(res);
            if (this.currentSessionId) {
                try {
                    localStorage.setItem('genie_session_id', this.currentSessionId);
                } catch (e) {}
            }
            this.loadSessions();
        } catch (err) {
            this.removeTypingIndicator(typingId);
            if (err.name !== 'AbortError') {
                this.addMessage('assistant', `Error: ${err.message}`, true);
            }
        } finally {
            this.isGenerating = false;
            this.abortController = null;
            this.setSendButtonSend();
            this.updateSendButton();
        }
    }

    async uploadFile(file, messageText) {
        if (this.isGenerating) {
            this.showBanner('Please wait for the current response to finish.', 'warning');
            return;
        }

        const welcome = document.getElementById('welcomeScreen');
        if (welcome) welcome.remove();

        const isImage = file.type.startsWith('image/');
        let displayMessage = '';
        if (isImage) {
            const imageUrl = URL.createObjectURL(file);
            displayMessage = messageText 
                ? `![${file.name}](${imageUrl})\n\n${messageText}` 
                : `![${file.name}](${imageUrl})`;
        } else {
            displayMessage = messageText 
                ? `📄 **${file.name}**\n\n${messageText}` 
                : `📄 **${file.name}**`;
        }
        
        this.addMessage('user', displayMessage);
        
        const input = document.getElementById('messageInput');
        input.value = '';
        input.style.height = 'auto';
        this.removeFile();
        
        this.isGenerating = true;
        this.setSendButtonStop();

        // Show "Analyzing..." immediately
        const analyzingId = this.showTypingIndicator();
        
        // Update status text
        const statusElement = document.getElementById('inputStatus');
        if (statusElement) {
            statusElement.textContent = 'Analyzing document... this may take 30-60 seconds on CPU';
        }

        const formData = new FormData();
        formData.append('file', file);
        if (this.currentSessionId) formData.append('session_id', this.currentSessionId);
        if (messageText) formData.append('prompt', messageText);

        try {
            const res = await fetch(`${API_BASE}/api/analyze-file`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                let errorMsg = `Upload failed with status ${res.status}`;
                try {
                    const errData = await res.json();
                    if (errData.detail) errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
                } catch (e) {}
                throw new Error(errorMsg);
            }

            const data = await res.json();
            if (data.session_id) {
                this.currentSessionId = data.session_id;
                try { localStorage.setItem('genie_session_id', this.currentSessionId); } catch (e) {}
            }
            
            const message = data.response || 'File uploaded.';
            this.addMessage('assistant', message);
            this.loadSessions();
        } catch (err) {
            this.addMessage('assistant', `Error: ${err.message}`, true);
        } finally {
            this.removeTypingIndicator(analyzingId);
            const statusElement = document.getElementById('inputStatus');
            if (statusElement) {
                statusElement.textContent = 'Genie can make mistakes. Consider checking important information.';
            }
            this.isGenerating = false;
            this.abortController = null;
            this.setSendButtonSend();
            this.updateSendButton();
        }
    }

    stopGeneration() {
        if (this.abortController) this.abortController.abort();
    }

    setSendButtonStop() {
        const btn = document.getElementById('sendBtn');
        btn.disabled = false;
        btn.title = 'Stop generation';
        btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
    }

    setSendButtonSend() {
        const btn = document.getElementById('sendBtn');
        btn.title = 'Send message';
        btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
    }

    // ── SSE stream handler ────────────────────────────────────────
    async handleStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let messageDiv = null;
        let contentDiv = null;
        let fullContent = '';

        const processBlock = (block) => {
            const lines = block.split('\n');
            let event = null;
            let dataStr = null;

            for (const line of lines) {
                if (line.startsWith('event: ')) event = line.slice(7).trim();
                else if (line.startsWith('data: ')) dataStr = line.slice(6);
            }

            if (dataStr === null) return;

            let data;
            try { data = JSON.parse(dataStr); } catch { return; }

            if (event === 'token' && typeof data === 'string') {
                if (!messageDiv) {
                    messageDiv = this.createMessageElement('assistant');
                    contentDiv = messageDiv.querySelector('.bubble');
                    document.getElementById('messagesArea').appendChild(messageDiv);
                }
                fullContent += data;
                contentDiv.innerHTML = this.formatMarkdown(fullContent);
                // Render KaTeX math after each update
                this.renderMath(contentDiv);
                const area = document.getElementById('messagesArea');
                area.scrollTop = area.scrollHeight;

            } else if (event === 'error') {
                this.addMessage('assistant', `Error: ${data}`, true);

            } else if (event === 'session') {
                if (!this.currentSessionId) {
                    this.currentSessionId = data;
                    try { localStorage.setItem('genie_session_id', data); } catch (e) {}
                }
            }
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const blocks = buffer.split('\n\n');
            buffer = blocks.pop();

            for (const block of blocks) {
                if (block.trim()) processBlock(block);
            }
        }
        if (buffer.trim()) processBlock(buffer);
    }

    // ── KaTeX Math Rendering ─────────────────────────────────────
    renderMath(element) {
        if (typeof renderMathInElement === 'undefined') return;
        try {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false,
                errorColor: '#cc0000',
                trust: false,
                strict: false
            });
        } catch (e) {
            // Silently fail if KaTeX not loaded yet
        }
    }

    // ── Markdown formatting ───────────────────────────────────────
    formatMarkdown(text) {
        // Escape HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks
        html = html.replace(/```([^\n]*)\n?([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        // Inline code
        html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        // Images ![alt](url)
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; border-radius: 8px; margin: 8px 0;">');
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    // ── Message rendering ────────────────────────────────────────
    addMessage(role, content, isError = false) {
        const area = document.getElementById('messagesArea');
        const div = this.createMessageElement(role, isError);
        const bubble = div.querySelector('.bubble');
        
        if (isError) {
            bubble.textContent = content;
        } else {
            bubble.innerHTML = this.formatMarkdown(content);
            // Render math after adding to DOM
            this.renderMath(bubble);
        }
        
        area.appendChild(div);
        area.scrollTop = area.scrollHeight;
        return div;
    }

    createMessageElement(role, isError = false) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? 'U' : 'G';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (isError) {
            bubble.style.background = 'rgba(239,68,68,0.1)';
            bubble.style.color = '#ef4444';
        }
        div.appendChild(avatar);
        div.appendChild(bubble);
        return div;
    }

    showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const area = document.getElementById('messagesArea');
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.id = id;
        div.innerHTML = `
            <div class="avatar">G</div>
            <div class="bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>`;
        area.appendChild(div);
        area.scrollTop = area.scrollHeight;
        return id;
    }

    removeTypingIndicator(id) {
        document.getElementById(id)?.remove();
    }

    updateSendButton() {
        if (this.isGenerating) return;
        const input = document.getElementById('messageInput');
        const btn = document.getElementById('sendBtn');
        btn.disabled = (!input.value.trim() && !this.pendingFile) || !this.modelReady;
    }

    // ── Sessions ─────────────────────────────────────────────────
    async loadSessions() {
        try {
            const res = await fetch(`${API_BASE}/api/sessions`);
            const data = await res.json();
            this.sessions = data.sessions || [];
            this.renderSessionsList();
        } catch {}
    }

    renderSessionsList() {
        const container = document.getElementById('sessionsList');
        container.querySelectorAll('.session-item').forEach(el => el.remove());
        this.sessions.forEach(session => {
            const div = document.createElement('div');
            div.className = 'session-item' + (session.session_id === this.currentSessionId ? ' active' : '');
            const title = session.first_message
                ? session.first_message.slice(0, 35) + (session.first_message.length > 35 ? '…' : '')
                : `Chat ${session.session_id.slice(-6)}`;
            div.innerHTML = `<span>${title}</span>`;
            div.onclick = () => this.loadSession(session.session_id);
            container.appendChild(div);
        });
    }

    async loadSession(sessionId) {
        try {
            const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
            const data = await res.json();
            this.currentSessionId = sessionId;
            const area = document.getElementById('messagesArea');
            area.innerHTML = '';
            data.messages.forEach(msg => {
                if (msg.role !== 'system') {
                    const displayContent = (msg.metadata && msg.metadata.display_content) 
                        ? msg.metadata.display_content 
                        : msg.content;
                    this.addMessage(msg.role, displayContent);
                }
            });
            this.renderSessionsList();
        } catch {}
    }

    newChat() {
        this.currentSessionId = null;
        try { localStorage.removeItem('genie_session_id'); } catch (e) {}
        const area = document.getElementById('messagesArea');
        area.innerHTML = `
            <div class="welcome-screen" id="welcomeScreen">
                <div class="welcome-content">
                    <div class="welcome-icon">
                        <img src="/static/The_Genie_Aladdin.png" alt="Genie logo" class="welcome-logo">
                    </div>
                    <h2>Welcome to Genie</h2>
                    <p>A fully local AI chatbot running on your machine.<br>No API keys. No cloud. 100% private.</p>
                </div>
            </div>`;
        this.renderSessionsList();
    }

    clearCurrentChat() {
        if (this.currentSessionId) {
            fetch(`${API_BASE}/api/sessions/${this.currentSessionId}`, { method: 'DELETE' })
                .then(() => {
                    this.showBanner('Chat deleted', 'success');
                    this.newChat();
                    this.loadSessions();
                })
                .catch(() => this.showBanner('Failed to delete chat', 'error'));
        } else {
            this.newChat();
        }
    }

    // ── Notifications ────────────────────────────────────────────
    showBanner(message, type = 'info') {
        const colors = { success: '#2563eb', error: '#ef4444', warning: '#f59e0b', info: '#3b82f6' };
        document.getElementById('_banner')?.remove();
        const el = document.createElement('div');
        el.id = '_banner';
        el.style.cssText = `
            position:fixed; top:16px; left:50%; transform:translateX(-50%);
            background:${colors[type] || colors.info}; color:white;
            padding:10px 20px; border-radius:8px; font-size:14px;
            z-index:9999; box-shadow:0 4px 12px rgba(0,0,0,0.4);
            max-width:90%; text-align:center;`;
        el.textContent = message;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 3500);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});