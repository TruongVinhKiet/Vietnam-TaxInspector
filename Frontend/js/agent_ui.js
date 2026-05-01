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

    // ─── Model Mode State ────────────────────────────────────────────
    let currentModelMode = 'full';
    let pendingFile = null;
    const modelSelectorBtn = document.getElementById('modelSelectorBtn');
    const modelDropdown = document.getElementById('modelDropdown');
    const modelSelectorLabel = document.getElementById('modelSelectorLabel');
    const modelIcon = document.getElementById('modelIcon');
    const fileUploadInput = document.getElementById('fileUploadInput');
    const fileDropOverlay = document.getElementById('fileDropOverlay');
    const fileChipContainer = document.getElementById('fileChipContainer');
    const fileChipName = document.getElementById('fileChipName');
    const fileChipRemove = document.getElementById('fileChipRemove');
    const attachmentBtn = document.getElementById('attachmentBtn');

    const MODE_CONFIG = {
        full:        { label: 'Toàn diện',   icon: 'fa-bolt',                     color: 'text-amber-500' },
        fraud:       { label: 'Gian lận',    icon: 'fa-magnifying-glass-chart',   color: 'text-red-500' },
        vat:         { label: 'VAT & HĐ',    icon: 'fa-file-invoice-dollar',      color: 'text-blue-500' },
        delinquency: { label: 'Dự báo Nợ',   icon: 'fa-chart-line',               color: 'text-violet-500' },
        macro:       { label: 'Vĩ mô',       icon: 'fa-globe',                    color: 'text-emerald-500' },
    };

    // Model selector dropdown toggle
    if (modelSelectorBtn) {
        modelSelectorBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            modelDropdown.classList.toggle('hidden');
        });
        document.addEventListener('click', () => modelDropdown?.classList.add('hidden'));
        modelDropdown?.addEventListener('click', (e) => e.stopPropagation());
    }

    // Model option click
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.addEventListener('click', () => {
            currentModelMode = opt.dataset.mode;
            const cfg = MODE_CONFIG[currentModelMode];
            if (modelSelectorLabel) modelSelectorLabel.textContent = cfg.label;
            if (modelIcon) {
                modelIcon.className = `fa-solid ${cfg.icon} ${cfg.color}`;
            }
            // Highlight active option
            document.querySelectorAll('.model-option').forEach(o => o.classList.remove('bg-sky-50', 'ring-1', 'ring-sky-300'));
            opt.classList.add('bg-sky-50', 'ring-1', 'ring-sky-300');
            modelDropdown.classList.add('hidden');
        });
    });

    // File Upload Logic
    if (attachmentBtn) {
        attachmentBtn.addEventListener('click', () => fileUploadInput?.click());
    }
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) setPendingFile(file);
        });
    }
    if (fileChipRemove) {
        fileChipRemove.addEventListener('click', () => clearPendingFile());
    }

    // Drag & Drop
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropOverlay?.classList.remove('hidden');
    });
    document.addEventListener('dragleave', (e) => {
        if (e.relatedTarget === null) fileDropOverlay?.classList.add('hidden');
    });
    document.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropOverlay?.classList.add('hidden');
        const file = e.dataTransfer?.files[0];
        if (file && file.name.toLowerCase().endsWith('.csv')) {
            setPendingFile(file);
        }
    });

    function setPendingFile(file) {
        pendingFile = file;
        if (fileChipContainer) fileChipContainer.classList.remove('hidden');
        if (fileChipName) fileChipName.textContent = file.name;
        // Enable send button
        sendBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        sendBtn.removeAttribute('disabled');
    }
    function clearPendingFile() {
        pendingFile = null;
        if (fileChipContainer) fileChipContainer.classList.add('hidden');
        if (fileUploadInput) fileUploadInput.value = '';
    }

    // Suggestion Chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const msg = chip.dataset.msg;
            if (!msg) {
                // Upload CSV chip
                fileUploadInput?.click();
                return;
            }
            chatInput.value = msg;
            chatInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });

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
        if (!text && !pendingFile) return;

        if (emptyState) emptyState.style.display = 'none';
        const displayText = text || (pendingFile ? `📎 ${pendingFile.name}` : '');
        addMessageToFeed('user', displayText);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.classList.add('opacity-50', 'cursor-not-allowed');
        sendBtn.setAttribute('disabled', 'true');

        try {
            triggerAgentIndicators(text);

            // File uploads use non-streaming endpoint
            if (pendingFile) {
                const loadingId = addTypingIndicator();
                const formData = new FormData();
                formData.append('message', text || `Phân tích rủi ro file ${pendingFile.name}`);
                formData.append('file', pendingFile);
                formData.append('session_id', 'demo-session-01');
                formData.append('model_mode', currentModelMode);
                const response = await fetch('http://localhost:8000/api/tax-agent/chat/v2/with-file', {
                    method: 'POST', body: formData,
                });
                clearPendingFile();
                const data = await response.json();
                document.getElementById(loadingId)?.remove();
                const answerHtml = formatMarkdown(data.answer || data.content || data.response || '');
                const vizHtml = buildVisualizationCards(data.visualization_data || {}, data);
                const metaHtml = buildMetaCards(data);
                addMessageToFeed('agent', answerHtml + vizHtml + metaHtml, data);
                requestAnimationFrame(() => renderPendingCharts());
            } else {
                // SSE Streaming mode
                await sendMessageStreaming(text);
            }

        } catch (error) {
            console.error('API Error:', error);
            addMessageToFeed('agent', 'Xin lỗi, đã xảy ra lỗi khi kết nối tới Hệ thống Multi-Agent.');
        } finally {
            resetAgentIndicators();
            sendBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            sendBtn.removeAttribute('disabled');
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  SSE STREAMING ENGINE
    // ═══════════════════════════════════════════════════════════

    async function sendMessageStreaming(text) {
        // Create thinking bubble
        const thinkingEl = createThinkingBubble();
        chatFeed.appendChild(thinkingEl);
        scrollToBottom();

        // Create response bubble (will be populated incrementally)
        const responseWrapper = document.createElement('div');
        responseWrapper.className = 'w-full flex mb-6 chat-message';
        responseWrapper.innerHTML = `
            <div class="flex gap-4 w-full">
                <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200 mt-1">
                    <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover agent-message-avatar">
                </div>
                <div class="flex flex-col gap-2 max-w-[90%] w-full">
                    <div class="text-xs font-semibold text-slate-500 ml-1">TaxInspector AI <span class="streaming-latency"></span></div>
                    <div class="agent-message text-[15px]">
                        <div class="stream-text-content"></div>
                        <div class="stream-viz-content"></div>
                    </div>
                </div>
            </div>
        `;
        responseWrapper.style.display = 'none'; // Hidden until first text_chunk

        const streamTextEl = responseWrapper.querySelector('.stream-text-content');
        const streamVizEl = responseWrapper.querySelector('.stream-viz-content');
        const latencyEl = responseWrapper.querySelector('.streaming-latency');
        let fullData = null;
        let answerAccumulator = '';

        try {
            const response = await fetch('http://localhost:8000/api/tax-agent/chat/v2/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: 'demo-session-01',
                    message: text,
                    model_mode: currentModelMode,
                }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                let currentEventType = '';
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        currentEventType = line.slice(7).trim();
                    } else if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        try {
                            const data = JSON.parse(dataStr);
                            handleSSEEvent(currentEventType, data, {
                                thinkingEl, responseWrapper, streamTextEl, streamVizEl, latencyEl,
                            });

                            // Accumulate answer
                            if (currentEventType === 'text_chunk') {
                                answerAccumulator += data.chunk || '';
                            }

                            // Store final data
                            if (currentEventType === 'done') {
                                fullData = data;
                            }
                        } catch (e) { /* skip malformed */ }
                    }
                }
            }
        } catch (err) {
            console.error('SSE Error:', err);
        }

        // Remove thinking bubble
        thinkingEl.remove();

        // If no text was streamed, show response wrapper with fallback
        if (responseWrapper.style.display === 'none') {
            responseWrapper.style.display = '';
            if (fullData?.answer) {
                streamTextEl.innerHTML = formatMarkdown(fullData.answer);
            }
            chatFeed.appendChild(responseWrapper);
        }

        // Final: add viz + meta if available from done event
        if (fullData) {
            const vizHtml = buildVisualizationCards(fullData.visualization_data || {}, fullData);
            const metaHtml = buildMetaCards(fullData);
            streamVizEl.innerHTML = vizHtml + metaHtml;
            if (latencyEl && fullData.latency_ms) {
                latencyEl.innerHTML = `<i class="fa-solid fa-bolt"></i> ${fullData.latency_ms.toFixed(0)}ms`;
                latencyEl.className = 'text-[10px] text-slate-400 ml-2 streaming-latency';
            }
            // Bind table row clicks
            responseWrapper.querySelectorAll('.viz-table-row-click').forEach(row => {
                row.addEventListener('click', () => {
                    const taxCode = row.dataset.taxCode;
                    if (taxCode) {
                        chatInput.value = `Phân tích chi tiết MST ${taxCode}`;
                        chatInput.dispatchEvent(new Event('input'));
                        sendMessage();
                    }
                });
            });
            appendFeedbackButtons(responseWrapper, fullData);
            requestAnimationFrame(() => renderPendingCharts());
        }

        // Remove typewriter cursor
        streamTextEl.classList.add('stream-done');

        scrollToBottom();
    }

    function handleSSEEvent(eventType, data, els) {
        const { thinkingEl, responseWrapper, streamTextEl, streamVizEl, latencyEl } = els;
        const stepsEl = thinkingEl.querySelector('.thinking-steps');

        switch (eventType) {
            case 'thinking': {
                const step = data.step || '';
                const detail = data.detail || '';
                const icon = step === 'intent' || step === 'intent_done' ? 'fa-brain'
                    : step === 'planning' || step === 'plan_done' ? 'fa-sitemap'
                    : step === 'synthesis' ? 'fa-wand-magic-sparkles'
                    : step === 'nl_query' || step === 'batch' ? 'fa-database'
                    : step === 'react' ? 'fa-scale-balanced'
                    : step === 'conv_intel' ? 'fa-lightbulb'
                    : 'fa-circle-notch fa-spin';

                // Update or add step
                const isDone = step.endsWith('_done');
                if (isDone) {
                    // Mark previous step as done
                    const lastStep = stepsEl.lastElementChild;
                    if (lastStep && !lastStep.classList.contains('special-step')) {
                        lastStep.querySelector('i')?.classList.remove('fa-spin');
                        lastStep.querySelector('i')?.classList.add('text-emerald-500');
                        lastStep.querySelector('.step-detail')?.classList.add('text-emerald-600');
                        const checkIcon = lastStep.querySelector('i');
                        if (checkIcon) checkIcon.className = 'fa-solid fa-circle-check text-emerald-500 text-xs';
                    }
                    // Add done detail
                    const doneEl = document.createElement('div');
                    doneEl.className = 'thinking-step flex items-center gap-2 text-xs animate-fadeIn';
                    doneEl.innerHTML = `<i class="fa-solid fa-circle-check text-emerald-500 text-xs"></i> <span class="step-detail text-emerald-600">${detail}</span>`;
                    stepsEl.appendChild(doneEl);
                } else if (step === 'react' || step === 'conv_intel') {
                    const stepEl = document.createElement('div');
                    stepEl.className = 'thinking-step special-step mt-1 animate-fadeIn';
                    
                    if (step === 'react') {
                        stepEl.innerHTML = `
                            <div class="flex items-start gap-2">
                                <i class="fa-solid ${icon} text-amber-500 text-xs mt-[3px]"></i> 
                                <div>
                                    <span class="step-detail text-slate-500 font-semibold">Tự đánh giá & Điều chỉnh</span>
                                    <div class="mt-1 text-[11px] px-2 py-1 bg-amber-50 text-amber-700 rounded border border-amber-200 inline-block">
                                        <i class="fa-solid fa-arrows-rotate mr-1"></i> ${detail}
                                    </div>
                                </div>
                            </div>
                        `;
                    } else if (step === 'conv_intel') {
                        stepEl.innerHTML = `
                            <div class="flex items-start gap-2">
                                <i class="fa-solid ${icon} text-sky-500 text-xs mt-[3px]"></i> 
                                <div>
                                    <span class="step-detail text-slate-500 font-semibold">Nhận diện Ngữ cảnh</span>
                                    <div class="mt-1 text-[11px] px-2 py-1 bg-sky-50 text-sky-700 rounded border border-sky-200 inline-block">
                                        <i class="fa-solid fa-link mr-1"></i> ${detail}
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    stepsEl.appendChild(stepEl);
                } else {
                    const stepEl = document.createElement('div');
                    stepEl.className = 'thinking-step flex items-center gap-2 text-xs animate-fadeIn';
                    stepEl.innerHTML = `<i class="fa-solid ${icon} text-sky-500 text-xs"></i> <span class="step-detail text-slate-500">${detail}</span>`;
                    stepsEl.appendChild(stepEl);
                }
                scrollToBottom();
                break;
            }

            case 'tool_start': {
                const stepEl = document.createElement('div');
                stepEl.className = 'thinking-step flex items-center gap-2 text-xs animate-fadeIn';
                stepEl.dataset.tool = data.tool;
                stepEl.innerHTML = `<i class="fa-solid fa-gear fa-spin text-indigo-500 text-xs"></i> <span class="step-detail text-slate-500">🔧 ${data.tool}</span>`;
                stepsEl.appendChild(stepEl);
                scrollToBottom();
                break;
            }

            case 'tool_done': {
                const toolStep = stepsEl.querySelector(`[data-tool="${data.tool}"]`);
                if (toolStep) {
                    const statusIcon = data.status === 'success' ? 'fa-circle-check text-emerald-500' : 'fa-circle-xmark text-red-400';
                    toolStep.querySelector('i').className = `fa-solid ${statusIcon} text-xs`;
                    toolStep.querySelector('.step-detail').classList.add(data.status === 'success' ? 'text-emerald-600' : 'text-red-500');
                    toolStep.querySelector('.step-detail').textContent += ` ✓ ${data.latency_ms}ms`;
                }
                break;
            }

            case 'sub_agent': {
                if (data.status === 'running') {
                    const stepEl = document.createElement('div');
                    stepEl.className = 'thinking-step flex items-center gap-2 text-xs animate-fadeIn';
                    stepEl.dataset.agent = data.agent;
                    const agentIcon = data.agent === 'legal' ? '📋' : data.agent === 'analytics' ? '📊' : '🔍';
                    stepEl.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin text-violet-500 text-xs"></i> <span class="step-detail text-slate-500">${agentIcon} ${data.detail || data.agent}</span>`;
                    stepsEl.appendChild(stepEl);
                } else if (data.status === 'done') {
                    const agentStep = stepsEl.querySelector(`[data-agent="${data.agent}"]`);
                    if (agentStep) {
                        agentStep.querySelector('i').className = 'fa-solid fa-circle-check text-emerald-500 text-xs';
                        agentStep.querySelector('.step-detail').classList.add('text-emerald-600');
                    }
                }
                scrollToBottom();
                break;
            }

            case 'text_chunk': {
                // Show response wrapper on first chunk
                if (responseWrapper.style.display === 'none') {
                    responseWrapper.style.display = '';
                    chatFeed.appendChild(responseWrapper);
                }
                // Append chunk with typewriter effect
                streamTextEl.innerHTML = formatMarkdown(
                    (streamTextEl._rawText || '') + (data.chunk || '')
                );
                streamTextEl._rawText = (streamTextEl._rawText || '') + (data.chunk || '');
                scrollToBottom();
                break;
            }

            case 'debate': {
                // Render debate panel inline
                if (responseWrapper.style.display === 'none') {
                    responseWrapper.style.display = '';
                    chatFeed.appendChild(responseWrapper);
                }
                const debateHtml = buildDebatePanel(data);
                if (debateHtml) {
                    const debateDiv = document.createElement('div');
                    debateDiv.innerHTML = debateHtml;
                    streamVizEl.insertAdjacentElement('beforebegin', debateDiv);
                }
                scrollToBottom();
                break;
            }

            case 'viz_data': {
                const vizHtml = buildVisualizationCards(data, {});
                streamVizEl.innerHTML = vizHtml;
                requestAnimationFrame(() => renderPendingCharts());
                scrollToBottom();
                break;
            }

            case 'error': {
                streamTextEl.innerHTML = `<p class="text-red-500">⚠️ Lỗi: ${data.error || 'Unknown error'}</p>`;
                break;
            }
        }
    }

    function createThinkingBubble() {
        const wrapper = document.createElement('div');
        wrapper.className = 'w-full flex mb-4 chat-message thinking-bubble-wrapper';
        wrapper.innerHTML = `
            <div class="flex gap-4 w-full">
                <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200 mt-1">
                    <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover agent-message-avatar opacity-70">
                </div>
                <div class="flex flex-col gap-1.5 max-w-[80%]">
                    <div class="thinking-bubble bg-gradient-to-r from-slate-50 to-sky-50 border border-slate-200 rounded-2xl rounded-bl px-4 py-3">
                        <div class="flex items-center gap-2 mb-2">
                            <div class="thinking-pulse"></div>
                            <span class="text-xs font-bold text-slate-500 uppercase tracking-wide">Đang suy nghĩ...</span>
                        </div>
                        <div class="thinking-steps flex flex-col gap-1.5"></div>
                    </div>
                </div>
            </div>
        `;
        return wrapper;
    }

    // ═══════════════════════════════════════════════════════════
    //  RICH VISUALIZATION CARD BUILDERS
    // ═══════════════════════════════════════════════════════════

    let chartCounter = 0;
    const pendingCharts = [];

    function buildVisualizationCards(viz, fullData) {
        if (!viz || Object.keys(viz).length === 0) return '';
        let html = '<div class="viz-cards-container">';

        // 1. Risk Gauge
        if (viz.risk_gauge) {
            const g = viz.risk_gauge;
            const gaugeId = `gauge-${++chartCounter}`;
            pendingCharts.push({ type: 'gauge', id: gaugeId, data: g });
            html += `
            <div class="viz-card viz-card-gauge">
                <div class="viz-card-header"><i class="fa-solid fa-gauge-high"></i> Điểm Rủi Ro Tổng Hợp</div>
                <div class="viz-card-body" style="display:flex;align-items:center;gap:20px;">
                    <div style="width:130px;height:130px;position:relative;">
                        <canvas id="${gaugeId}" width="130" height="130"></canvas>
                        <div class="gauge-center-text" style="color:${g.color}">${g.score.toFixed(0)}</div>
                    </div>
                    <div>
                        <div class="viz-badge" style="background:${g.color}20;color:${g.color};border:1px solid ${g.color}40">${g.level.toUpperCase()}</div>
                        <div class="text-xs text-slate-500 mt-2">Độ tin cậy: ${g.confidence}%</div>
                    </div>
                </div>
            </div>`;
        }

        // 2. Delinquency Timeline
        if (viz.delinquency_timeline) {
            const dt = viz.delinquency_timeline;
            const chartId = `dqchart-${++chartCounter}`;
            pendingCharts.push({ type: 'delinquency', id: chartId, data: dt });
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-chart-line"></i> Dự Báo Nợ Đọng — ML vs Deep Learning</div>
                <div class="viz-card-body"><canvas id="${chartId}" height="180"></canvas></div>
                ${dt.dl_architecture ? `<div class="viz-card-footer"><i class="fa-solid fa-microchip"></i> ${dt.dl_architecture}</div>` : ''}
            </div>`;
        }

        // 3. Anomaly Scatter
        if (viz.anomaly_scatter) {
            const as = viz.anomaly_scatter;
            const chartId = `anomaly-${++chartCounter}`;
            pendingCharts.push({ type: 'anomaly', id: chartId, data: as });
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-triangle-exclamation"></i> Phát Hiện Bất Thường Hóa Đơn (VAE)</div>
                <div class="viz-card-body">
                    <div style="display:flex;gap:16px;margin-bottom:12px;">
                        <div class="viz-stat"><span class="viz-stat-num">${as.total}</span><span class="viz-stat-label">Tổng HĐ</span></div>
                        <div class="viz-stat"><span class="viz-stat-num" style="color:#DC2626">${as.anomaly_count}</span><span class="viz-stat-label">Bất thường</span></div>
                        <div class="viz-stat"><span class="viz-stat-num">${as.anomaly_ratio}%</span><span class="viz-stat-label">Tỷ lệ</span></div>
                    </div>
                    <canvas id="${chartId}" height="160"></canvas>
                </div>
                ${as.architecture ? `<div class="viz-card-footer"><i class="fa-solid fa-microchip"></i> ${as.architecture}</div>` : ''}
            </div>`;
        }

        // 4. Network Graph
        if (viz.network_graph) {
            const ng = viz.network_graph;
            const graphId = `netgraph-${++chartCounter}`;
            pendingCharts.push({ type: 'network', id: graphId, data: ng });
            html += `
            <div class="viz-card">
                <div class="viz-card-header"><i class="fa-solid fa-circle-nodes"></i> Đồ Thị Quan Hệ Doanh Nghiệp (HGT)</div>
                <div class="viz-card-body"><svg id="${graphId}" width="100%" height="200" viewBox="0 0 400 200"></svg></div>
                ${ng.architecture ? `<div class="viz-card-footer"><i class="fa-solid fa-microchip"></i> ${ng.architecture}</div>` : ''}
            </div>`;
        }

        // 5. Uplift Actions
        if (viz.uplift_actions) {
            const ua = viz.uplift_actions;
            const chartId = `uplift-${++chartCounter}`;
            pendingCharts.push({ type: 'uplift', id: chartId, data: ua });
            html += `
            <div class="viz-card">
                <div class="viz-card-header"><i class="fa-solid fa-arrows-split-up-and-left"></i> Đề Xuất Hành Động Thu Nợ (Causal AI)</div>
                <div class="viz-card-body">
                    <div class="text-sm mb-2">CATE Score: <strong>${ua.cate_score.toFixed(4)}</strong></div>
                    <div class="text-sm mb-3 text-emerald-600"><i class="fa-solid fa-star"></i> ${ua.recommended}</div>
                    <canvas id="${chartId}" height="120"></canvas>
                </div>
                ${ua.architecture ? `<div class="viz-card-footer"><i class="fa-solid fa-microchip"></i> ${ua.architecture}</div>` : ''}
            </div>`;
        }

        // 6. Model Comparison
        if (viz.model_comparison && viz.model_comparison.length > 0) {
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-layer-group"></i> Tổng Kết Mô Hình AI Đã Sử Dụng (${viz.model_comparison.length} models)</div>
                <div class="viz-card-body">
                    <table class="viz-table">
                        <thead><tr><th>Model</th><th>Kiến trúc</th><th>Kết quả</th></tr></thead>
                        <tbody>${viz.model_comparison.map(m => `
                            <tr>
                                <td><span class="viz-model-badge">${m.model}</span></td>
                                <td class="text-xs text-slate-500">${m.architecture}</td>
                                <td><span class="viz-risk-pill viz-risk-${m.risk_level}">${m.risk_level}</span></td>
                            </tr>`).join('')}
                        </tbody>
                    </table>
                </div>
            </div>`;
        }

        // 7. Tool Timeline
        if (viz.tool_timeline && viz.tool_timeline.length > 0) {
            html += `
            <details class="viz-card viz-card-wide viz-collapsible">
                <summary class="viz-card-header cursor-pointer"><i class="fa-solid fa-timeline"></i> Pipeline Thực Thi (${viz.tool_timeline.length} tools) <i class="fa-solid fa-chevron-down text-xs ml-auto"></i></summary>
                <div class="viz-card-body">
                    <div class="viz-timeline">${viz.tool_timeline.map((t, i) => `
                        <div class="viz-timeline-step ${t.status === 'analyzed' || t.status === 'found' || t.status === 'success' ? 'step-ok' : t.status === 'error' ? 'step-err' : 'step-skip'}">
                            <div class="step-dot"></div>
                            <div class="step-content">
                                <div class="step-name">${t.tool}</div>
                                <div class="step-meta">${t.description || ''} ${t.latency_ms ? `• ${t.latency_ms}ms` : ''}</div>
                            </div>
                        </div>`).join('')}
                    </div>
                </div>
            </details>`;
        }

        // 8. Top-N Companies Table
        if (viz.top_companies && viz.top_companies.rows && viz.top_companies.rows.length) {
            const rows = viz.top_companies.rows;
            const total = viz.top_companies.total || rows.length;
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-ranking-star text-amber-500"></i> Top ${rows.length} Doanh nghiệp Rủi ro (${total} tổng)</div>
                <div class="viz-card-body p-0">
                    <div class="overflow-x-auto">
                        <table class="viz-data-table w-full text-sm">
                            <thead>
                                <tr class="bg-slate-50 text-slate-500 text-xs uppercase">
                                    <th class="px-3 py-2 text-left">#</th>
                                    <th class="px-3 py-2 text-left">MST</th>
                                    <th class="px-3 py-2 text-left">Tên DN</th>
                                    <th class="px-3 py-2 text-left">Ngành</th>
                                    <th class="px-3 py-2 text-right">Điểm RR</th>
                                    <th class="px-3 py-2 text-center">Mức</th>
                                </tr>
                            </thead>
                            <tbody>${rows.map((r, i) => {
                                const levelColors = {critical:'bg-red-100 text-red-700',high:'bg-orange-100 text-orange-700',medium:'bg-yellow-100 text-yellow-700',low:'bg-green-100 text-green-700'};
                                const badge = levelColors[r.risk_level] || 'bg-slate-100 text-slate-600';
                                return `<tr class="border-t border-slate-100 hover:bg-sky-50 cursor-pointer viz-table-row-click" data-tax-code="${r.tax_code}">
                                    <td class="px-3 py-2 text-slate-400 font-mono">${r.stt || i+1}</td>
                                    <td class="px-3 py-2 font-mono font-bold text-sky-700">${r.tax_code}</td>
                                    <td class="px-3 py-2 font-medium text-slate-700 max-w-[200px] truncate">${r.company_name || '—'}</td>
                                    <td class="px-3 py-2 text-slate-500 text-xs">${r.industry || '—'}</td>
                                    <td class="px-3 py-2 text-right font-bold">${(r.risk_score || 0).toFixed(1)}</td>
                                    <td class="px-3 py-2 text-center"><span class="px-2 py-0.5 rounded-full text-xs font-bold ${badge}">${r.risk_level || '?'}</span></td>
                                </tr>`;
                            }).join('')}</tbody>
                        </table>
                    </div>
                </div>
            </div>`;
        }

        // 9. Batch Summary Card
        if (viz.batch_summary) {
            const b = viz.batch_summary;
            const bl = b.by_level || {};
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-file-csv text-emerald-500"></i> Kết quả Batch: ${b.filename || 'CSV'} (${b.total} DN)</div>
                <div class="viz-card-body">
                    <div class="grid grid-cols-4 gap-3 mb-4">
                        <div class="text-center p-3 rounded-lg bg-red-50"><div class="text-2xl font-black text-red-600">${bl.critical||0}</div><div class="text-xs text-red-500">Rất cao</div></div>
                        <div class="text-center p-3 rounded-lg bg-orange-50"><div class="text-2xl font-black text-orange-600">${bl.high||0}</div><div class="text-xs text-orange-500">Cao</div></div>
                        <div class="text-center p-3 rounded-lg bg-yellow-50"><div class="text-2xl font-black text-yellow-600">${bl.medium||0}</div><div class="text-xs text-yellow-500">Trung bình</div></div>
                        <div class="text-center p-3 rounded-lg bg-green-50"><div class="text-2xl font-black text-green-600">${bl.low||0}</div><div class="text-xs text-green-500">An toàn</div></div>
                    </div>
                    ${b.top_5 && b.top_5.length ? `
                    <div class="mt-3">
                        <p class="text-xs font-bold text-slate-500 mb-2">Top 5 rủi ro cao nhất:</p>
                        ${b.top_5.map((c, i) => `
                            <div class="flex items-center justify-between py-1.5 border-b border-slate-100 hover:bg-sky-50 cursor-pointer viz-table-row-click" data-tax-code="${c.tax_code}">
                                <span class="text-sm"><span class="font-mono text-sky-600 mr-2">${c.tax_code}</span>${c.company_name || ''}</span>
                                <span class="text-sm font-bold">${(c.risk_score||0).toFixed(1)}</span>
                            </div>
                        `).join('')}
                    </div>` : ''}
                </div>
            </div>`;
        }

        // 10. Company Name Search Results
        if (viz.company_search_results && viz.company_search_results.matches) {
            const matches = viz.company_search_results.matches;
            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-building text-sky-500"></i> Kết quả tìm kiếm: "${viz.company_search_results.query}" (${matches.length} kết quả)</div>
                <div class="viz-card-body p-0">
                    <div class="divide-y divide-slate-100">
                        ${matches.map(m => `
                            <div class="flex items-center justify-between px-4 py-3 hover:bg-sky-50 cursor-pointer viz-table-row-click" data-tax-code="${m.tax_code}">
                                <div>
                                    <p class="font-medium text-slate-700">${m.company_name}</p>
                                    <p class="text-xs text-slate-400">MST: <span class="font-mono text-sky-600">${m.tax_code}</span> • ${m.industry || '—'}</p>
                                </div>
                                <i class="fa-solid fa-chevron-right text-slate-300"></i>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>`;
        }

        // 11. XAI — SHAP Waterfall Chart
        if (viz.xai_shap && viz.xai_shap.shap_waterfall) {
            const sw = viz.xai_shap.shap_waterfall;
            const attrs = sw.attributions || [];
            const chartId = `shap-${++chartCounter}`;
            pendingCharts.push({ type: 'shap_waterfall', id: chartId, data: sw });
            html += `
            <div class="viz-card viz-card-wide xai-card">
                <div class="viz-card-header"><i class="fa-solid fa-brain text-violet-500"></i> XAI — Giải Thích Dự Đoán (SHAP)</div>
                <div class="viz-card-body">
                    <div class="text-sm text-slate-600 mb-3">${viz.xai_shap.summary || ''}</div>
                    <canvas id="${chartId}" height="${Math.max(120, attrs.length * 28)}"></canvas>
                    <div class="flex items-center gap-4 mt-3 text-xs text-slate-400">
                        <span><span class="inline-block w-3 h-3 rounded-sm mr-1" style="background:#EF4444"></span> Tăng rủi ro</span>
                        <span><span class="inline-block w-3 h-3 rounded-sm mr-1" style="background:#22C55E"></span> Giảm rủi ro</span>
                        <span class="ml-auto">Base: ${(sw.base_value * 100).toFixed(1)}% → Pred: ${(sw.prediction * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>`;
        }

        // 12. XAI — VAE Anomaly Breakdown
        if (viz.xai_vae && viz.xai_vae.vae_breakdown) {
            const vb = viz.xai_vae.vae_breakdown;
            const attrs = vb.attributions || [];
            html += `
            <div class="viz-card xai-card">
                <div class="viz-card-header"><i class="fa-solid fa-magnifying-glass-chart text-amber-500"></i> XAI — VAE Bất Thường Chi Tiết</div>
                <div class="viz-card-body">
                    <div class="text-sm text-slate-600 mb-3">${viz.xai_vae.summary || ''}</div>
                    <div class="space-y-2">
                        ${attrs.map(a => {
                            const pct = Math.min(100, a.attribution);
                            const color = a.direction === 'anomaly' ? '#EF4444' : '#22C55E';
                            return `
                            <div class="flex items-center gap-2 text-xs">
                                <span class="w-32 text-right text-slate-500 truncate" title="${a.feature}">${a.label}</span>
                                <div class="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
                                    <div class="h-full rounded-full transition-all duration-500" style="width:${pct}%;background:${color}"></div>
                                </div>
                                <span class="w-12 text-right font-mono font-bold" style="color:${color}">${a.attribution.toFixed(1)}%</span>
                            </div>`;
                        }).join('')}
                    </div>
                </div>
            </div>`;
        }

        // 13. XAI — Counterfactual (What-If)
        if (viz.xai_shap && viz.xai_shap.counterfactual) {
            const cf = viz.xai_shap.counterfactual;
            html += `
            <div class="viz-card xai-card">
                <div class="viz-card-header"><i class="fa-solid fa-shuffle text-sky-500"></i> Phân Tích Phản Thực (Counterfactual)</div>
                <div class="viz-card-body">
                    <div class="text-sm text-slate-600 mb-3">${cf.summary || ''}</div>
                    <div class="flex items-center gap-3 mb-3">
                        <div class="text-center px-4 py-2 rounded-lg bg-red-50">
                            <div class="text-lg font-black text-red-600">${(cf.original_prediction * 100).toFixed(1)}%</div>
                            <div class="text-xs text-red-400">Hiện tại</div>
                        </div>
                        <i class="fa-solid fa-arrow-right text-slate-300"></i>
                        <div class="text-center px-4 py-2 rounded-lg ${cf.found ? 'bg-green-50' : 'bg-amber-50'}">
                            <div class="text-lg font-black ${cf.found ? 'text-green-600' : 'text-amber-600'}">${(cf.new_prediction * 100).toFixed(1)}%</div>
                            <div class="text-xs ${cf.found ? 'text-green-400' : 'text-amber-400'}">${cf.found ? 'Sau thay đổi' : 'Chưa đủ'}</div>
                        </div>
                    </div>
                    ${cf.changes && cf.changes.length ? `
                    <div class="space-y-1.5 text-xs">
                        <div class="font-bold text-slate-500 mb-1">Thay đổi cần thiết:</div>
                        ${cf.changes.map(c => `
                        <div class="flex items-center gap-2 px-2 py-1 rounded bg-slate-50">
                            <span class="text-slate-500">${c.label}</span>
                            <span class="ml-auto font-mono text-red-400 line-through">${c.from.toFixed(2)}</span>
                            <i class="fa-solid fa-arrow-right text-slate-300 text-[10px]"></i>
                            <span class="font-mono text-emerald-600 font-bold">${c.to.toFixed(2)}</span>
                        </div>`).join('')}
                    </div>` : ''}
                </div>
            </div>`;
        }

        // 14. Multi-Agent Debate Panel
        if (viz.agent_debate) {
            html += buildDebatePanel(viz.agent_debate);
        }

        html += '</div>';
        return html;
    }

    function buildDebatePanel(debate) {
        if (!debate || !debate.stances || debate.stances.length < 2) return '';

        const stanceIcons = { safe: '✅', cautious: '⚠️', suspicious: '🔶', dangerous: '🔴' };
        const stanceColors = { safe: '#22C55E', cautious: '#EAB308', suspicious: '#F97316', dangerous: '#DC2626' };
        const severityBadge = { minor: 'bg-slate-100 text-slate-600', moderate: 'bg-amber-100 text-amber-700', major: 'bg-orange-100 text-orange-700', critical: 'bg-red-100 text-red-700' };

        // Consensus header
        const cColor = stanceColors[debate.consensus_stance] || '#64748B';
        let html = `
        <div class="viz-card viz-card-wide debate-card">
            <div class="viz-card-header"><i class="fa-solid fa-scale-balanced text-violet-500"></i> Hội Đồng Agent — Tranh Luận Đa Chiều</div>
            <div class="viz-card-body">
                <!-- Consensus Gauge -->
                <div class="flex items-center gap-4 mb-4 p-3 rounded-xl" style="background: ${cColor}10; border: 1px solid ${cColor}30">
                    <div class="text-center">
                        <div class="text-3xl font-black" style="color: ${cColor}">${debate.consensus_pct}%</div>
                        <div class="text-xs text-slate-500">Đồng thuận</div>
                    </div>
                    <div class="flex-1">
                        <div class="h-3 bg-slate-100 rounded-full overflow-hidden mb-1">
                            <div class="h-full rounded-full transition-all duration-700" style="width:${debate.consensus_pct}%; background: ${cColor}"></div>
                        </div>
                        <div class="text-sm font-semibold" style="color: ${cColor}">${debate.consensus_label}</div>
                    </div>
                </div>

                <!-- Agent Stances -->
                <div class="grid gap-2 mb-4" style="grid-template-columns: repeat(${Math.min(debate.stances.length, 3)}, 1fr)">
                    ${debate.stances.map(s => {
                        const sc = stanceColors[s.stance] || '#64748B';
                        return `
                        <div class="debate-stance-card" style="border-color: ${sc}30">
                            <div class="flex items-center gap-2 mb-2">
                                <i class="${s.icon} text-sm" style="color: ${sc}"></i>
                                <span class="font-bold text-sm">${s.label}</span>
                                <span class="ml-auto text-xs px-2 py-0.5 rounded-full font-bold" style="background: ${sc}15; color: ${sc}">${stanceIcons[s.stance] || '?'} ${(s.risk_score).toFixed(0)}%</span>
                            </div>
                            <div class="text-xs text-slate-500 space-y-1">
                                ${s.findings.slice(0, 3).map(f => `<div>${f}</div>`).join('')}
                            </div>
                            <div class="mt-2 text-xs text-slate-400">Độ tin cậy: ${(s.confidence * 100).toFixed(0)}%</div>
                        </div>`;
                    }).join('')}
                </div>

                <!-- Disagreements -->
                ${debate.disagreements && debate.disagreements.length ? `
                <div class="mb-3">
                    <div class="text-xs font-bold text-slate-500 mb-2"><i class="fa-solid fa-bolt text-amber-500"></i> Bất đồng phát hiện (${debate.disagreements.length})</div>
                    <div class="space-y-2">
                        ${debate.disagreements.map(d => `
                        <div class="debate-disagreement">
                            <div class="flex items-center gap-2 mb-1">
                                <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${severityBadge[d.severity] || 'bg-slate-100'}">${d.severity.toUpperCase()}</span>
                                <span class="text-xs font-semibold text-slate-600">${d.topic}</span>
                            </div>
                            <div class="grid grid-cols-2 gap-2 text-xs mb-1">
                                ${Object.entries(d.positions).map(([agent, pos]) =>
                                    `<div class="px-2 py-1 rounded bg-slate-50"><span class="font-semibold">${agent}:</span> ${pos}</div>`
                                ).join('')}
                            </div>
                            ${d.resolution ? `<div class="text-xs text-slate-500 italic">💡 ${d.resolution}</div>` : ''}
                        </div>`).join('')}
                    </div>
                </div>` : ''}

                <!-- Minority Opinions -->
                ${debate.minority_opinions && debate.minority_opinions.length ? `
                <div class="mb-3">
                    <div class="text-xs font-bold text-slate-500 mb-1"><i class="fa-solid fa-user-slash text-red-400"></i> Ý kiến thiểu số</div>
                    ${debate.minority_opinions.map(m => `
                    <div class="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 text-xs">
                        <i class="${m.icon} text-red-500"></i>
                        <span class="font-semibold">${m.agent}</span>: ${m.reason}
                        <span class="ml-auto text-red-600 font-bold">${m.risk_score}%</span>
                    </div>`).join('')}
                </div>` : ''}

                <!-- Recommendation -->
                <div class="p-3 rounded-xl bg-sky-50 border border-sky-200 text-sm text-sky-800">
                    <i class="fa-solid fa-lightbulb text-sky-500"></i> ${debate.recommendation}
                </div>

                ${debate.summary ? `<div class="text-xs text-slate-400 mt-2 italic">${debate.summary}</div>` : ''}
            </div>
        </div>`;

        return html;
    }

    function buildMetaCards(data) {
        if (!data) return '';
        let html = '';

        // Recommendation Pills
        if (data.recommendations && data.recommendations.length > 0) {
            html += '<div class="viz-rec-pills">';
            data.recommendations.forEach(r => {
                html += `<div class="viz-rec-pill">${r}</div>`;
            });
            html += '</div>';
        }

        return html;
    }

    // ═══════════════════════════════════════════════════════════
    //  CHART.JS RENDERERS (called after DOM insertion)
    // ═══════════════════════════════════════════════════════════

    function renderPendingCharts() {
        while (pendingCharts.length > 0) {
            const cfg = pendingCharts.shift();
            const el = document.getElementById(cfg.id);
            if (!el) continue;

            try {
                if (cfg.type === 'gauge') renderGaugeChart(el, cfg.data);
                else if (cfg.type === 'delinquency') renderDelinquencyChart(el, cfg.data);
                else if (cfg.type === 'anomaly') renderAnomalyChart(el, cfg.data);
                else if (cfg.type === 'network') renderNetworkGraph(el, cfg.data);
                else if (cfg.type === 'uplift') renderUpliftChart(el, cfg.data);
                else if (cfg.type === 'shap_waterfall') renderShapWaterfall(el, cfg.data);
            } catch (e) { console.warn('Chart render error:', e); }
        }
    }

    function renderGaugeChart(canvas, data) {
        const score = data.score;
        const remaining = 100 - score;
        new Chart(canvas, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [score, remaining],
                    backgroundColor: [data.color, '#E2E8F0'],
                    borderWidth: 0, borderRadius: 6,
                }]
            },
            options: {
                cutout: '75%', responsive: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                animation: { animateRotate: true, duration: 1200, easing: 'easeOutQuart' },
            }
        });
    }

    function renderDelinquencyChart(canvas, data) {
        const datasets = [];
        if (data.ml_values && data.ml_values.length) {
            datasets.push({
                label: 'XGBoost (ML)', data: data.ml_values,
                borderColor: '#6366F1', backgroundColor: '#6366F120',
                fill: true, tension: 0.3, borderWidth: 2, pointRadius: 5,
            });
        }
        if (data.dl_values && data.dl_values.length) {
            datasets.push({
                label: 'Transformer (DL)', data: data.dl_values,
                borderColor: '#F97316', backgroundColor: '#F9731620',
                fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 5, borderDash: [],
            });
        }
        new Chart(canvas, {
            type: 'line',
            data: { labels: data.labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, title: { display: true, text: 'Xác suất (%)' },
                         grid: { color: '#F1F5F9' } },
                    x: { grid: { display: false } },
                },
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%` } },
                },
                animation: { duration: 1000, easing: 'easeOutCubic' },
            }
        });
    }

    function renderAnomalyChart(canvas, data) {
        const anomalies = data.top_anomalies || [];
        const labels = anomalies.map((a, i) => a.invoice_number || `HĐ ${i+1}`);
        const scores = anomalies.map(a => a.anomaly_score);
        const colors = anomalies.map(a => a.is_anomaly ? '#DC262680' : '#22C55E80');
        const borders = anomalies.map(a => a.is_anomaly ? '#DC2626' : '#22C55E');

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Anomaly Score', data: scores,
                    backgroundColor: colors, borderColor: borders, borderWidth: 1.5,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: {
                    x: { title: { display: true, text: 'Reconstruction Error' }, grid: { color: '#F1F5F9' } },
                    y: { grid: { display: false }, ticks: { font: { size: 10 } } },
                },
                plugins: {
                    legend: { display: false },
                    annotation: data.threshold ? {
                        annotations: { thresholdLine: {
                            type: 'line', xMin: data.threshold, xMax: data.threshold,
                            borderColor: '#EF4444', borderWidth: 2, borderDash: [5, 5],
                            label: { content: `Threshold: ${data.threshold}`, display: true }
                        }}
                    } : undefined,
                },
                animation: { duration: 800 },
            }
        });
    }

    function renderNetworkGraph(svg, data) {
        const nodes = data.nodes || [];
        const edges = data.edges || [];
        if (!nodes.length) return;

        const w = 400, h = 200;
        const cx = w / 2, cy = h / 2;

        // Position nodes in a circle
        nodes.forEach((n, i) => {
            if (i === 0) { n.x = cx; n.y = cy; }
            else {
                const angle = (2 * Math.PI * (i - 1)) / (nodes.length - 1);
                n.x = cx + Math.cos(angle) * 80;
                n.y = cy + Math.sin(angle) * 70;
            }
        });

        const nodeMap = {};
        nodes.forEach(n => nodeMap[n.id] = n);

        let svgHtml = '';
        // Edges
        edges.forEach(e => {
            const src = nodeMap[e.source];
            const tgt = nodeMap[e.target];
            if (src && tgt) {
                svgHtml += `<line x1="${src.x}" y1="${src.y}" x2="${tgt.x}" y2="${tgt.y}" stroke="#94A3B8" stroke-width="1.5" stroke-opacity="0.6"/>`;
            }
        });
        // Nodes
        const typeColors = { company: '#3B82F6', person: '#F97316', offshore_entity: '#DC2626' };
        nodes.forEach((n, i) => {
            const r = i === 0 ? 18 : 12;
            const color = typeColors[n.type] || '#64748B';
            const riskOpacity = 0.3 + (n.risk || 0) * 0.7;
            svgHtml += `<circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${color}" fill-opacity="${riskOpacity}" stroke="${color}" stroke-width="2"/>`;
            svgHtml += `<text x="${n.x}" y="${n.y + r + 14}" text-anchor="middle" font-size="9" fill="#475569">${n.label}</text>`;
            if (i === 0) {
                svgHtml += `<circle cx="${n.x}" cy="${n.y}" r="${r + 4}" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 2" opacity="0.5"/>`;
            }
        });

        svg.innerHTML = svgHtml;
    }

    function renderShapWaterfall(canvas, data) {
        const attrs = data.attributions || [];
        const labels = attrs.map(a => a.label || a.feature);
        const values = attrs.map(a => a.attribution);
        const colors = attrs.map(a => a.direction === 'positive' ? '#EF4444' : '#22C55E'); // Red for +risk, Green for -risk

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: {
                    x: {
                        grid: { color: '#F1F5F9' },
                        title: { display: true, text: 'SHAP Value (Tác động tới dự đoán)' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { size: 10 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const val = ctx.raw > 0 ? '+' + ctx.raw.toFixed(4) : ctx.raw.toFixed(4);
                                return ` Tác động: ${val}`;
                            }
                        }
                    }
                },
                animation: { duration: 800 }
            }
        });
    }

    function renderUpliftChart(canvas, data) {
        const actions = data.actions || [];
        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: actions.map(a => a.action),
                datasets: [{
                    label: 'Expected Lift', data: actions.map(a => a.expected_lift),
                    backgroundColor: actions.map((a, i) =>
                        a.action === data.recommended ? '#059669' : '#06B6D4' + (i === 0 ? '' : '90')),
                    borderRadius: 6, borderSkipped: false,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: {
                    x: { title: { display: true, text: 'CATE (Expected Lift)' }, grid: { color: '#F1F5F9' } },
                    y: { grid: { display: false }, ticks: { font: { size: 10 } } },
                },
                plugins: { legend: { display: false } },
                animation: { duration: 900, easing: 'easeOutQuart' },
            }
        });
    }

    // ═══════════════════════════════════════════════════════════
    //  MESSAGE FEED RENDERING
    // ═══════════════════════════════════════════════════════════

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
            // Build entity card
            let entityCard = '';
            if (agentData && agentData.active_tax_code) {
                entityCard = `
                    <div class="agent-card">
                        <div class="text-xs text-sky-600 font-bold mb-1"><i class="fa-solid fa-building"></i> Đối tượng Phân tích</div>
                        <div class="font-semibold text-slate-800">MST: ${agentData.active_tax_code}</div>
                        ${agentData.intent ? `<div class="text-xs text-slate-500 mt-1">Intent: ${agentData.intent} (${(agentData.intent_confidence * 100).toFixed(0)}%)</div>` : ''}
                    </div>`;
            }

            // Latency badge
            let latencyBadge = '';
            if (agentData && agentData.latency_ms) {
                latencyBadge = `<span class="text-[10px] text-slate-400 ml-2"><i class="fa-solid fa-bolt"></i> ${agentData.latency_ms.toFixed(0)}ms</span>`;
            }

            // Tools badge
            let toolsBadge = '';
            if (agentData && agentData.tools_used && agentData.tools_used.length > 0) {
                toolsBadge = `<span class="text-[10px] text-indigo-400 ml-2"><i class="fa-solid fa-wrench"></i> ${agentData.tools_used.length} tools</span>`;
            }

            wrapper.innerHTML = `
                <div class="flex gap-4 w-full">
                    <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200 mt-1">
                        <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover agent-message-avatar">
                    </div>
                    <div class="flex flex-col gap-2 max-w-[90%] w-full">
                        <div class="text-xs font-semibold text-slate-500 ml-1">TaxInspector AI ${latencyBadge} ${toolsBadge}</div>
                        <div class="agent-message text-[15px]">
                            ${contentHTML}
                            ${entityCard}
                        </div>
                    </div>
                </div>
            `;
        }

        chatFeed.appendChild(wrapper);
        scrollToBottom();

        // Delegated click: table row → auto-analyze that MST
        wrapper.querySelectorAll('.viz-table-row-click').forEach(row => {
            row.addEventListener('click', () => {
                const taxCode = row.dataset.taxCode;
                if (taxCode) {
                    chatInput.value = `Phân tích chi tiết MST ${taxCode}`;
                    chatInput.dispatchEvent(new Event('input'));
                    sendMessage();
                }
            });
        });

        if (role === 'agent' && agentData) {
            appendFeedbackButtons(wrapper, agentData);
        }
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
        // New DL indicators
        if (/\d{10}/.test(text) || lowerText.includes('phân tích') || lowerText.includes('rủi ro')) {
            document.getElementById('status-deeplearning')?.classList.add('status-active');
        }
        if (lowerText.includes('thu nợ') || lowerText.includes('cưỡng chế') || lowerText.includes('hành động')) {
            document.getElementById('status-causal')?.classList.add('status-active');
        }
        
        if (!document.querySelectorAll('.status-active').length) {
            document.getElementById('status-analytics').classList.add('status-active');
        }
    }

    function resetAgentIndicators() {
        document.querySelectorAll('.status-active').forEach(el => el.classList.remove('status-active'));
    }

    // Enhanced markdown parser with tables and headers
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Headers
        let html = text.replace(/^### (.*$)/gm, '<h4 class="font-bold text-slate-700 mt-3 mb-1">$1</h4>');
        html = html.replace(/^## (.*$)/gm, '<h3 class="font-bold text-slate-800 mt-4 mb-2 text-lg">$1</h3>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Tables (markdown-style)
        const tableRegex = /\|(.+)\|\n\|[-| ]+\|\n((\|.+\|\n?)+)/g;
        html = html.replace(tableRegex, (match, header, body) => {
            const heads = header.split('|').filter(h => h.trim()).map(h => `<th>${h.trim()}</th>`).join('');
            const rows = body.trim().split('\n').map(row => {
                const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${cells}</tr>`;
            }).join('');
            return `<table class="viz-table"><thead><tr>${heads}</tr></thead><tbody>${rows}</tbody></table>`;
        });
        
        // Lists and paragraphs
        const lines = html.split('\n');
        let formatted = [];
        let inList = false;
        
        for (let line of lines) {
            line = line.trim();
            if (!line) continue;
            
            if (line.startsWith('- ') || line.startsWith('* ')) {
                if (!inList) { formatted.push('<ul>'); inList = true; }
                formatted.push(`<li>${line.substring(2)}</li>`);
            } else if (/^\d+\. /.test(line)) {
                if (!inList) { formatted.push('<ol>'); inList = true; }
                formatted.push(`<li>${line.replace(/^\d+\. /, '')}</li>`);
            } else {
                if (inList) { formatted.push('</ul>'); inList = false; }
                if (!line.startsWith('<')) formatted.push(`<p>${line}</p>`);
                else formatted.push(line);
            }
        }
        if (inList) formatted.push('</ul>');
        
        return formatted.join('');
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // ═══════════════════════════════════════════════════════════
    //  FEEDBACK BUTTONS (Gap 5)
    // ═══════════════════════════════════════════════════════════
    function appendFeedbackButtons(wrapper, agentData) {
        if (!agentData) return;
        
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'flex items-center gap-2 mt-3 text-slate-400 text-xs';
        feedbackContainer.innerHTML = `
            <button class="feedback-btn hover:text-emerald-500 transition-colors" data-type="positive" title="Câu trả lời tốt">
                <i class="fa-regular fa-thumbs-up"></i>
            </button>
            <button class="feedback-btn hover:text-red-500 transition-colors" data-type="negative" title="Câu trả lời chưa tốt">
                <i class="fa-regular fa-thumbs-down"></i>
            </button>
            <span class="feedback-status ml-2 opacity-0 transition-opacity"></span>
        `;
        
        wrapper.querySelector('.agent-message').appendChild(feedbackContainer);
        
        const btns = feedbackContainer.querySelectorAll('.feedback-btn');
        const status = feedbackContainer.querySelector('.feedback-status');
        
        btns.forEach(btn => {
            btn.addEventListener('click', async () => {
                const type = btn.dataset.type;
                
                // Visual toggle
                btns.forEach(b => b.classList.remove('text-emerald-500', 'text-red-500', 'fa-solid'));
                btn.querySelector('i').classList.remove('fa-regular');
                btn.querySelector('i').classList.add('fa-solid');
                if (type === 'positive') btn.classList.add('text-emerald-500');
                else btn.classList.add('text-red-500');
                
                status.textContent = 'Đang gửi...';
                status.classList.remove('opacity-0');
                
                try {
                    const reqBody = {
                        session_id: agentData.session_id || 'demo-session-01',
                        turn_id: agentData.turn_index || 1,
                        feedback_type: type,
                        intent: agentData.intent || '',
                        confidence: agentData.intent_confidence || 0,
                    };
                    
                    const res = await fetch('http://localhost:8000/api/tax-agent/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(reqBody)
                    });
                    
                    if (res.ok) {
                        status.innerHTML = '<i class="fa-solid fa-check text-emerald-500"></i> Cảm ơn bạn!';
                        setTimeout(() => status.classList.add('opacity-0'), 2000);
                    } else {
                        status.textContent = 'Lỗi gửi phản hồi';
                    }
                } catch (e) {
                    status.textContent = 'Lỗi kết nối';
                }
            });
        });
    }
});
