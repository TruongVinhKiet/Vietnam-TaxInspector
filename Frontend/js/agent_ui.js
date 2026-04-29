// e:\TaxInspector\Frontend\js\agent_ui.js

document.addEventListener('DOMContentLoaded', () => {
    const chatFeed = document.getElementById('chatFeed');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');
    const emptyState = document.getElementById('emptyState');
    const backToDashBtn = document.getElementById('backToDashBtn');
    const audioVisualizer = document.getElementById('audioVisualizer');

    // Theme Switcher Elements
    const themeSettingsBtn = document.getElementById('themeSettingsBtn');
    const wardrobeModal = document.getElementById('wardrobeModal');
    const wardrobeModalContent = document.getElementById('wardrobeModalContent');
    const closeWardrobeBtn = document.getElementById('closeWardrobeBtn');
    const themeOptions = document.querySelectorAll('.theme-option');
    const applyThemeBtn = document.getElementById('applyThemeBtn');
    const headAvatarImg = document.querySelector('header img[alt="AI Agent Avatar"]');
    const emptyStateAvatar = document.getElementById('emptyStateAvatar');

    let currentTheme = localStorage.getItem('taxAgentTheme') || 'ai_avatar.png';
    let selectedTheme = currentTheme;

    // Apply initial theme
    if (headAvatarImg) {
        headAvatarImg.src = `../assets/img/${currentTheme}`;
    }
    if (emptyStateAvatar) {
        emptyStateAvatar.src = `../assets/img/${currentTheme}`;
    }

    // Modal Logic
    if (themeSettingsBtn) {
        themeSettingsBtn.addEventListener('click', () => {
            wardrobeModal.classList.remove('hidden');
            // Small delay for transition
            setTimeout(() => {
                wardrobeModal.classList.remove('opacity-0');
                wardrobeModalContent.classList.remove('scale-95');
            }, 10);
            
            // Highlight current theme
            themeOptions.forEach(opt => {
                const border = opt.firstElementChild;
                const check = opt.querySelector('.checkmark');
                if (opt.dataset.theme === currentTheme) {
                    border.classList.add('border-sky-500');
                    border.classList.remove('border-transparent');
                    check.classList.remove('hidden');
                } else {
                    border.classList.remove('border-sky-500');
                    border.classList.add('border-transparent');
                    check.classList.add('hidden');
                }
            });
            applyThemeBtn.disabled = true;
        });
    }

    function closeWardrobeModal() {
        wardrobeModal.classList.add('opacity-0');
        wardrobeModalContent.classList.add('scale-95');
        setTimeout(() => {
            wardrobeModal.classList.add('hidden');
        }, 300);
    }

    if (closeWardrobeBtn) closeWardrobeBtn.addEventListener('click', closeWardrobeModal);
    
    // Select Theme
    themeOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            selectedTheme = opt.dataset.theme;
            
            themeOptions.forEach(o => {
                o.firstElementChild.classList.remove('border-sky-500');
                o.firstElementChild.classList.add('border-transparent');
                o.querySelector('.checkmark').classList.add('hidden');
            });
            
            opt.firstElementChild.classList.add('border-sky-500');
            opt.firstElementChild.classList.remove('border-transparent');
            opt.querySelector('.checkmark').classList.remove('hidden');
            
            if (selectedTheme !== currentTheme) {
                applyThemeBtn.disabled = false;
            } else {
                applyThemeBtn.disabled = true;
            }
        });
    });

    // Save Theme
    if (applyThemeBtn) {
        applyThemeBtn.addEventListener('click', () => {
            currentTheme = selectedTheme;
            localStorage.setItem('taxAgentTheme', currentTheme);
            
            // Update UI immediately
            if (headAvatarImg) headAvatarImg.src = `../assets/img/${currentTheme}`;
            if (emptyStateAvatar) emptyStateAvatar.src = `../assets/img/${currentTheme}`;
            
            // Update all chat bubble avatars
            document.querySelectorAll('.agent-message-avatar').forEach(img => {
                img.src = `../assets/img/${currentTheme}`;
            });
            
            closeWardrobeModal();
        });
    }

    // Return to dashboard
    backToDashBtn.addEventListener('click', () => {
        window.history.back(); // Or window.location.href = 'dashboard.html'
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() !== '') {
            sendBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            sendBtn.removeAttribute('disabled');
        } else {
            sendBtn.classList.add('opacity-50', 'cursor-not-allowed');
            sendBtn.setAttribute('disabled', 'true');
        }
    });

    // Handle Enter to send (Shift+Enter for newline)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // Voice Input (Web Speech API)
    let recognition;
    let isRecording = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'vi-VN';
        recognition.continuous = false;
        recognition.interimResults = true;

        recognition.onstart = () => {
            isRecording = true;
            micBtn.classList.add('mic-active');
            chatInput.placeholder = "Đang rảnh nghe...";
            audioVisualizer.classList.remove('opacity-0');
            audioVisualizer.classList.add('opacity-30'); // show waves behind chat
        };

        recognition.onresult = (event) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            if (finalTranscript) {
                chatInput.value = finalTranscript;
                chatInput.dispatchEvent(new Event('input')); // trigger resize and button state
                sendMessage(); // auto send when voice stops
            }
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            stopRecording();
        };

        recognition.onend = () => {
            stopRecording();
        };
    } else {
        micBtn.style.display = 'none'; // hide if not supported
    }

    micBtn.addEventListener('click', () => {
        if (!recognition) return;
        if (isRecording) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });

    function stopRecording() {
        isRecording = false;
        micBtn.classList.remove('mic-active');
        chatInput.placeholder = "Nhập yêu cầu phân tích thuế hoặc MST doanh nghiệp...";
        audioVisualizer.classList.remove('opacity-30');
        audioVisualizer.classList.add('opacity-0');
    }

    // --- Message Handling ---

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Hide empty state
        if (emptyState) {
            emptyState.style.display = 'none';
        }

        // Add user message to UI
        addMessageToFeed('user', text);
        
        // Reset input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.classList.add('opacity-50', 'cursor-not-allowed');
        sendBtn.setAttribute('disabled', 'true');

        // Add loading indicator
        const loadingId = addTypingIndicator();

        try {
            // Trigger UI logic related to multi-agents (simulate analyzing)
            triggerAgentIndicators(text);

            // Fetch from backend
            const response = await fetch('http://localhost:8000/api/tax-agent/chat/v2', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: "demo-session-01",
                    message: text,
                    use_streaming: false
                })
            });

            const data = await response.json();
            
            // Remove loading
            document.getElementById(loadingId).remove();

            // Add Assistant message
            addMessageToFeed('agent', formatMarkdown(data.content || data.response), data);

        } catch (error) {
            console.error('API Error:', error);
            document.getElementById(loadingId).remove();
            addMessageToFeed('agent', "Xin lỗi, đã xảy ra lỗi khi kết nối tới Hệ thống Multi-Agent.");
        } finally {
            resetAgentIndicators();
        }
    }

    function addMessageToFeed(role, contentHTML, agentData = null) {
        const wrapper = document.createElement('div');
        wrapper.className = 'w-full flex mb-6 chat-message';

        if (role === 'user') {
            wrapper.innerHTML = `
                <div class="user-message text-[15px]">
                    ${escapeHTML(contentHTML)}
                </div>
            `;
        } else {
            // Agent message
            let citationCards = '';
            
            // If we have parsed entities or citations, create cards
            if (agentData && agentData.metadata) {
                const meta = agentData.metadata;
                if (meta.entities && meta.entities.tax_codes && meta.entities.tax_codes.length > 0) {
                    const code = meta.entities.tax_codes[0];
                    citationCards += `
                        <div class="agent-card">
                            <div class="text-xs text-sky-600 font-bold mb-1"><i class="fa-solid fa-building"></i> Đối tượng Phân tích</div>
                            <div class="font-semibold text-slate-800">MST: ${code}</div>
                        </div>
                    `;
                }
            }

            wrapper.innerHTML = `
                <div class="flex gap-4 w-full">
                    <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200 mt-1">
                        <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover agent-message-avatar">
                    </div>
                    <div class="flex flex-col gap-2 max-w-[85%]">
                        <div class="text-xs font-semibold text-slate-500 ml-1">TaxInspector AI</div>
                        <div class="agent-message text-[15px]">
                            ${contentHTML}
                            ${citationCards}
                        </div>
                    </div>
                </div>
            `;
        }

        chatFeed.appendChild(wrapper);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const wrapper = document.createElement('div');
        wrapper.id = id;
        wrapper.className = 'w-full flex mb-6 chat-message';
        wrapper.innerHTML = `
            <div class="flex gap-4">
                <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200">
                    <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover grayscale opacity-70 agent-message-avatar">
                </div>
                <div class="agent-message flex items-center h-12 typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatFeed.appendChild(wrapper);
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function triggerAgentIndicators(text) {
        // Simple heuristic to light up UI elements based on text keywords for visual feedback
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('luật') || lowerText.includes('quy định') || lowerText.includes('điều')) {
            document.getElementById('status-legal').classList.add('status-active');
        }
        if (lowerText.includes('phân tích') || lowerText.includes('trễ hạn') || lowerText.includes('dự báo') || /\d{10}/.test(text)) {
            document.getElementById('status-analytics').classList.add('status-active');
        }
        if (lowerText.includes('gian lận') || lowerText.includes('mạng lưới') || lowerText.includes('rửa tiền') || lowerText.includes('sở hữu')) {
            document.getElementById('status-investigation').classList.add('status-active');
        }
        
        // If nothing specific matched, activate analytics by default to show work
        if (!document.querySelectorAll('.status-active').length) {
            document.getElementById('status-analytics').classList.add('status-active');
        }
    }

    function resetAgentIndicators() {
        document.querySelectorAll('.status-active').forEach(el => el.classList.remove('status-active'));
    }

    // Very simple markdown parser for bold, bullet points, and paragraphs
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Bold: **text**
        let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Split into lines to handle lists and paragraphs
        const lines = html.split('\n');
        let formattedLines = [];
        let inList = false;
        
        for (let line of lines) {
            line = line.trim();
            if (!line) continue;
            
            if (line.startsWith('- ') || line.startsWith('* ')) {
                if (!inList) {
                    formattedLines.push('<ul>');
                    inList = true;
                }
                formattedLines.push(`<li>${line.substring(2)}</li>`);
            } else {
                if (inList) {
                    formattedLines.push('</ul>');
                    inList = false;
                }
                formattedLines.push(`<p>${line}</p>`);
            }
        }
        if (inList) formattedLines.push('</ul>');
        
        return formattedLines.join('');
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
