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
        legal:       { label: 'Tư vấn Pháp lý', icon: 'fa-scale-balanced',        color: 'text-teal-500' },
        fraud:       { label: 'Gian lận',    icon: 'fa-user-secret',              color: 'text-red-500' },
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

            // Instead of auto-opening, just show the toggle button if in legal mode
            const lwToggleBtn = document.getElementById('lwToggleBtn');
            if (lwToggleBtn) {
                if (currentModelMode === 'legal') {
                    lwToggleBtn.classList.remove('hidden');
                } else {
                    lwToggleBtn.classList.add('hidden');
                    // Hide panel if it's open
                    const lwPanel = document.getElementById('legalWorkspacePanel');
                    if (lwPanel && !lwPanel.classList.contains('hidden')) {
                        lwPanel.classList.add('translate-x-full', 'opacity-0');
                        setTimeout(() => lwPanel.classList.add('hidden'), 300);
                    }
                }
            }
        });
    });

    const closeLwBtn = document.getElementById('closeLegalWorkspaceBtn');
    if (closeLwBtn) {
        closeLwBtn.addEventListener('click', () => {
            const lwPanel = document.getElementById('legalWorkspacePanel');
            if (lwPanel) {
                lwPanel.classList.add('translate-x-full', 'opacity-0');
                setTimeout(() => lwPanel.classList.add('hidden'), 300);
            }
        });
    }

    const lwToggleBtn = document.getElementById('lwToggleBtn');
    if (lwToggleBtn) {
        lwToggleBtn.addEventListener('click', () => {
            const lwPanel = document.getElementById('legalWorkspacePanel');
            if (lwPanel) {
                if (lwPanel.classList.contains('hidden')) {
                    lwPanel.classList.remove('hidden');
                    setTimeout(() => lwPanel.classList.remove('translate-x-full', 'opacity-0'), 10);
                } else {
                    lwPanel.classList.add('translate-x-full', 'opacity-0');
                    setTimeout(() => lwPanel.classList.add('hidden'), 300);
                }
            }
        });
    }

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

    // Drag & Drop — depth counter avoids overlay stuck when drag crosses child elements
    let fileDragDepth = 0;
    function hideFileDropOverlay() {
        fileDragDepth = 0;
        fileDropOverlay?.classList.add('hidden');
    }
    document.addEventListener('dragenter', (e) => {
        e.preventDefault();
        fileDragDepth++;
        fileDropOverlay?.classList.remove('hidden');
    });
    document.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileDragDepth--;
        if (fileDragDepth <= 0) hideFileDropOverlay();
    });
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
    });
    document.addEventListener('drop', (e) => {
        e.preventDefault();
        hideFileDropOverlay();
        const file = e.dataTransfer?.files[0];
        if (file && file.name.toLowerCase().endsWith('.csv')) {
            setPendingFile(file);
        }
    });

    function setPendingFile(file) {
        pendingFile = file;
        if (fileChipContainer) fileChipContainer.classList.remove('hidden');
        if (fileChipName) fileChipName.textContent = file.name;
        
        const sizeKb = (file.size / 1024).toFixed(1);
        const fileChipSize = document.getElementById('fileChipSize');
        if (fileChipSize) fileChipSize.textContent = `(${sizeKb} KB)`;

        const ext = file.name.split('.').pop().toLowerCase();
        const iconEl = document.getElementById('fileChipIcon');
        const thumbEl = document.getElementById('fileChipThumb');
        
        // Default reset
        if (iconEl) iconEl.className = 'fa-solid fa-file relative z-10';
        if (thumbEl) { thumbEl.classList.add('hidden'); thumbEl.style.backgroundImage = ''; }

        if (ext === 'csv') {
            if (iconEl) iconEl.className = 'fa-solid fa-file-csv text-emerald-600 relative z-10';
        } else if (ext === 'pdf') {
            if (iconEl) iconEl.className = 'fa-solid fa-file-pdf text-red-500 relative z-10';
        } else if (['png', 'jpg', 'jpeg'].includes(ext)) {
            if (iconEl) iconEl.className = 'fa-solid fa-image text-sky-600 relative z-10';
            // Image preview
            if (thumbEl) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    thumbEl.style.backgroundImage = `url(${e.target.result})`;
                    thumbEl.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            }
        }

        // Enable send button
        sendBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        sendBtn.removeAttribute('disabled');
    }
    function clearPendingFile() {
        pendingFile = null;
        if (fileChipContainer) fileChipContainer.classList.add('hidden');
        if (fileUploadInput) fileUploadInput.value = '';
        const thumbEl = document.getElementById('fileChipThumb');
        if (thumbEl) { thumbEl.classList.add('hidden'); thumbEl.style.backgroundImage = ''; }
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

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) return;
        if (recognition && isRecording) {
            try { recognition.stop(); } catch (_) { /* noop */ }
        }
        stopRecording();
    });
    window.addEventListener('pagehide', () => {
        if (recognition && isRecording) {
            try { recognition.stop(); } catch (_) { /* noop */ }
        }
        stopRecording();
    });

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
                const loadingId = addTypingIndicator(true);
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
        responseWrapper.className = 'max-w-4xl mx-auto w-full flex mb-6 chat-message';
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
        window.currentCitations = null; // reset per message

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
                            // Capture citations whenever they appear (not else-if!)
                            if (data.citations && data.citations.length) {
                                window.currentCitations = data.citations;
                            }
                            if (data.legal_workspace) {
                                renderLegalWorkspace(data.legal_workspace);
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
                streamTextEl._rawText = fullData.answer;
            }
            chatFeed.appendChild(responseWrapper);
        }

        // Ensure citations from fullData are captured before final re-render
        if (fullData && fullData.citations && fullData.citations.length) {
            window.currentCitations = fullData.citations;
        }

        // Final re-render of markdown to guarantee citation badges are clickable
        if (streamTextEl._rawText) {
            streamTextEl.innerHTML = formatMarkdown(streamTextEl._rawText, window.currentCitations || []);
            streamTextEl.classList.add('stream-done');
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
                        const isContradiction = data.is_contradiction || (typeof detail === 'string' && (detail.toLowerCase().includes('hết hiệu lực') || detail.toLowerCase().includes('xung đột') || detail.toLowerCase().includes('không áp dụng')));
                        
                        stepEl.innerHTML = `
                            <div class="flex items-start gap-2">
                                <i class="fa-solid ${icon} ${isContradiction ? 'text-red-500' : 'text-amber-500'} text-xs mt-[3px]"></i> 
                                <div>
                                    <span class="step-detail text-slate-500 font-semibold">${isContradiction ? 'Cảnh báo Pháp lý' : 'Tự đánh giá & Điều chỉnh'}</span>
                                    <div class="mt-1 text-[11px] px-2 py-1 ${isContradiction ? 'bg-red-50 text-red-700 border-red-200' : 'bg-amber-50 text-amber-700 border-amber-200'} rounded border inline-block">
                                        <i class="fa-solid ${isContradiction ? 'fa-triangle-exclamation' : 'fa-arrows-rotate'} mr-1"></i> ${detail}
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
                stepEl.className = 'thinking-step w-full animate-fadeIn mt-1.5 mb-1.5';
                stepEl.dataset.tool = data.tool;
                stepEl.innerHTML = `
                    <div class="flex items-center justify-between text-xs mb-1">
                        <div class="flex items-center gap-1.5">
                            <i class="fa-solid fa-gear fa-spin text-indigo-500 text-[10px]"></i>
                            <span class="step-detail text-slate-600 font-bold uppercase tracking-wider text-[10px]">Thực thi: ${data.tool}</span>
                        </div>
                        <span class="text-[9px] text-indigo-400 font-mono tracking-widest uppercase"><i class="fa-solid fa-bolt fa-fade mr-1"></i>Running...</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1 overflow-hidden tool-progress-bar">
                        <div class="h-full bg-indigo-400 rounded-full w-1/3 animate-pulse"></div>
                    </div>
                `;
                stepsEl.appendChild(stepEl);
                scrollToBottom();
                break;
            }

            case 'tool_done': {
                const toolStep = stepsEl.querySelector(`[data-tool="${data.tool}"]`);
                if (toolStep) {
                    const isSuccess = data.status === 'success';
                    const statusColor = isSuccess ? 'text-emerald-500' : 'text-red-500';
                    const bgColor = isSuccess ? 'bg-emerald-400' : 'bg-red-400';
                    const icon = isSuccess ? 'fa-check' : 'fa-xmark';
                    
                    toolStep.querySelector('i.fa-gear').className = `fa-solid ${icon} ${statusColor} text-[10px]`;
                    toolStep.querySelector('.step-detail').className = `step-detail font-bold text-[10px] ${statusColor} uppercase tracking-wider`;
                    toolStep.querySelector('.step-detail').textContent = `Hoàn tất: ${data.tool}`;
                    
                    toolStep.querySelector('span.font-mono').innerHTML = `<i class="fa-solid fa-clock mr-0.5"></i> ${data.latency_ms || 0}ms`;
                    toolStep.querySelector('span.font-mono').className = `text-[9px] font-mono font-bold ${statusColor}`;
                    
                    const pBar = toolStep.querySelector('.tool-progress-bar > div');
                    if (pBar) {
                        pBar.className = `h-full ${bgColor} rounded-full w-full transition-all duration-300`;
                    }
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
                    (streamTextEl._rawText || '') + (data.chunk || ''),
                    window.currentCitations || []
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
        wrapper.className = 'max-w-4xl mx-auto w-full flex mb-4 chat-message thinking-bubble-wrapper';
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

        // 4.5 Knowledge Graph (GraphRAG)
        if (viz.knowledge_graph) {
            const kg = viz.knowledge_graph;
            const graphId = `kg-${++chartCounter}`;
            pendingCharts.push({ type: 'knowledge_graph', id: graphId, data: kg });
            // Store KG data globally for fullscreen re-render
            window._lastKgData = kg;
            html += `
            <div class="viz-card" style="background: linear-gradient(145deg, #1e1b4b, #312e81); border: 1px solid #4338ca; color: white; cursor: pointer;" onclick="window._openKgFullscreen && window._openKgFullscreen()">
                <div class="viz-card-header" style="color: #c7d2fe; border-bottom: 1px solid #3730a3;">
                    <i class="fa-solid fa-diagram-project text-indigo-400"></i> Đồ Thị Tri Thức Pháp Luật (GraphRAG)
                </div>
                <div class="viz-card-body" style="padding: 0; position: relative;">
                    <div style="position: absolute; top: 10px; left: 10px; z-index: 10; font-size: 10px; color: #a5b4fc; background: rgba(0,0,0,0.4); padding: 4px 8px; border-radius: 4px; backdrop-filter: blur(4px);">
                        <i class="fa-solid fa-circle-nodes mr-1"></i> ${kg.total_entities} Nodes • ${kg.total_relations} Edges • Độ sâu: ${kg.expansion_depth}
                    </div>
                    <svg id="${graphId}" width="100%" height="320" viewBox="-350 -220 700 440"></svg>
                    <div class="kg-click-hint"><i class="fa-solid fa-expand mr-1"></i> Bấm để phóng to</div>
                </div>
                <div class="viz-card-footer" style="background: rgba(30, 27, 75, 0.8); color: #818cf8; border-top: 1px solid #3730a3;">
                    <i class="fa-solid fa-bolt text-amber-400 mr-1"></i> Engine: <span class="font-mono text-xs uppercase ml-1">${kg.retrieval_tier || 'GraphRAG'}</span>
                    <span class="ml-auto"><i class="fa-solid fa-clock mr-1"></i> ${kg.latency_ms}ms</span>
                </div>
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

        // 9.5 OCR Extraction Results
        if (viz.ocr_extraction) {
            const ocr = viz.ocr_extraction;
            const tables = ocr.tables || [];
            const fields = ocr.extracted_fields || {};
            
            let badgeClass = ocr.table_extraction_method === 'table_transformer' 
                ? 'bg-purple-100 text-purple-700 border-purple-200' 
                : 'bg-slate-100 text-slate-600 border-slate-200';
            let methodLabel = ocr.table_extraction_method === 'table_transformer' 
                ? 'AI DETR Transformer' 
                : (ocr.table_extraction_method === 'pdfplumber' ? 'PDF Plumber' : 'Heuristic Alignment');

            html += `
            <div class="viz-card viz-card-wide">
                <div class="viz-card-header"><i class="fa-solid fa-file-invoice text-sky-500"></i> Dữ liệu Trích xuất OCR</div>
                <div class="viz-card-body p-4">
                    <div class="mb-4 flex flex-wrap gap-2">`;
            
            for (const [key, val] of Object.entries(fields)) {
                if (!val || typeof val === 'object') continue;
                html += `<div class="bg-slate-50 px-3 py-2 rounded border border-slate-100 min-w-[120px]">
                            <p class="text-[10px] text-slate-400 uppercase tracking-wider">${key}</p>
                            <p class="text-sm font-bold text-slate-700">${val}</p>
                         </div>`;
            }

            html += `</div>`;

            if (tables.length > 0) {
                html += `
                    <div class="mt-4 pt-4 border-t border-slate-100">
                        <div class="flex items-center justify-between mb-3">
                            <h4 class="text-xs font-bold text-slate-600 uppercase">Bảng Dữ Liệu (${tables.length})</h4>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${badgeClass}">Engine: ${methodLabel}</span>
                        </div>`;
                
                tables.forEach(table => {
                    html += `<div class="mb-4 overflow-x-auto border border-slate-200 rounded">
                        <table class="w-full text-left text-[11px] whitespace-nowrap">`;
                    
                    if (table.headers && table.headers.length > 0) {
                        html += `<thead class="bg-slate-50 text-slate-600 border-b border-slate-200"><tr>`;
                        table.headers.forEach(h => {
                            html += `<th class="px-3 py-1.5 font-semibold">${h}</th>`;
                        });
                        html += `</tr></thead>`;
                    }
                    
                    if (table.rows && table.rows.length > 0) {
                        html += `<tbody class="divide-y divide-slate-100">`;
                        table.rows.forEach(row => {
                            html += `<tr class="hover:bg-sky-50/50">`;
                            row.forEach(cell => {
                                html += `<td class="px-3 py-1.5 text-slate-700">${cell || ''}</td>`;
                            });
                            html += `</tr>`;
                        });
                        html += `</tbody>`;
                    }
                    html += `</table></div>`;
                });
                html += `</div>`;
            }
            html += `</div></div>`;
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
                    <div style="position: relative; height: ${Math.max(250, attrs.length * 30)}px; width: 100%;">
                        <canvas id="${chartId}"></canvas>
                    </div>
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
            html += buildDebatePanel(viz.agent_debate, fullData.escalation_required);
        }

        html += '</div>';
        return html;
    }

    function buildDebatePanel(debate, isEscalated = false) {
        if (!debate || !debate.stances || debate.stances.length < 2) return '';

        const stanceIcons = { safe: '✅', cautious: '⚠️', suspicious: '🔶', dangerous: '🔴' };
        const stanceColors = { safe: '#22C55E', cautious: '#EAB308', suspicious: '#F97316', dangerous: '#DC2626' };
        const severityBadge = { minor: 'bg-slate-100 text-slate-600', moderate: 'bg-amber-100 text-amber-700', major: 'bg-orange-100 text-orange-700', critical: 'bg-red-100 text-red-700' };

        // Consensus header
        const cColor = stanceColors[debate.consensus_stance] || '#64748B';
        let html = `
        <div class="viz-card viz-card-wide debate-card relative overflow-hidden ${isEscalated ? 'border-red-300 ring-1 ring-red-200' : ''}">
            ${isEscalated ? `<div class="absolute -right-12 top-6 bg-red-600 text-white text-[10px] font-bold uppercase tracking-wider py-1 px-12 rotate-45 shadow-lg flex items-center gap-1 z-10"><i class="fa-solid fa-gavel"></i> Tòa Án AI</div>` : ''}
            <div class="viz-card-header ${isEscalated ? 'bg-red-50 text-red-700 border-b border-red-100' : ''}">
                <i class="fa-solid ${isEscalated ? 'fa-scale-unbalanced text-red-600' : 'fa-scale-balanced text-violet-500'}"></i> 
                ${isEscalated ? 'Cảnh Báo Bất Đồng — Kích Hoạt Tòa Án Adjudicator' : 'Hội Đồng Agent — Tranh Luận Đa Chiều'}
            </div>
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
                                ${s.findings.slice(0, 3).map(f => `<div>${formatMarkdown(f)}</div>`).join('')}
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
                                <span class="text-xs font-semibold text-slate-600">${formatMarkdown(d.topic)}</span>
                            </div>
                            <div class="grid grid-cols-2 gap-2 text-xs mb-1">
                                ${Object.entries(d.positions).map(([agent, pos]) =>
                                    `<div class="px-2 py-1 rounded bg-slate-50"><span class="font-semibold">${agent}:</span> ${formatMarkdown(pos)}</div>`
                                ).join('')}
                            </div>
                            ${d.resolution ? `<div class="text-xs text-slate-500 italic">💡 ${formatMarkdown(d.resolution)}</div>` : ''}
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
                html += `<div class="viz-rec-pill">${formatMarkdown(r)}</div>`;
            });
            html += '</div>';
        }
        // Planner DAG & Reasoning Trace
        if (data.reasoning_trace || (data.plan_steps && data.plan_steps.length > 0)) {
            html += `
            <div class="viz-card viz-card-wide bg-slate-50 border border-slate-200 mt-4 shadow-none">
                <div class="viz-card-header text-slate-500 border-b border-slate-200">
                    <i class="fa-solid fa-code-branch text-slate-400"></i> Dấu vết suy luận (DAG Planner Trace)
                </div>
                <div class="viz-card-body p-4 text-sm text-slate-600">
                    ${data.reasoning_trace ? `<div class="mb-3 italic border-l-2 border-slate-300 pl-3">" ${formatMarkdown(data.reasoning_trace)} "</div>` : ''}
                    
                    ${data.plan_steps && data.plan_steps.length > 0 ? `
                    <div class="relative pl-4 border-l-2 border-dashed border-sky-200 mt-4 space-y-4">
                        ${data.plan_steps.map((step, idx) => `
                        <div class="relative">
                            <div class="absolute -left-[23px] top-1 bg-sky-100 text-sky-600 font-bold rounded-full w-4 h-4 flex items-center justify-center text-[9px] ring-2 ring-white">${idx + 1}</div>
                            <div class="font-mono text-xs font-bold text-sky-700 mb-0.5"><i class="fa-solid fa-gear text-[10px] mr-1 text-sky-400"></i>${step.tool || step.tool_name} ${step.optional ? '<span class="text-slate-400 font-normal text-[9px]">(Optional)</span>' : ''}</div>
                            <div class="text-[11px] text-slate-500 leading-tight">${step.description}</div>
                        </div>
                        `).join('')}
                    </div>
                    ` : ''}
                </div>
            </div>`;
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
                else if (cfg.type === 'knowledge_graph') renderKnowledgeGraph(el, cfg.data);
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

    function renderKnowledgeGraph(svg, data) {
        const nodes = data.nodes || [];
        const edges = data.edges || [];
        if (!nodes.length) return;

        // Distribute all nodes evenly using Archimedean spiral
        nodes.forEach((n, i) => {
            if (i === 0) {
                n.x = 0; n.y = 0;
            } else {
                const angle = i * 2.39996; // Golden angle (137.5 degrees)
                const radius = 65 + Math.pow(i, 0.65) * 25; // Increased spacing
                n.x = Math.cos(angle) * radius;
                n.y = Math.sin(angle) * radius * 0.8;
            }
        });

        const nodeMap = {};
        nodes.forEach(n => nodeMap[n.id] = n);

        let svgHtml = `
        <defs>
            <filter id="kg-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>`;

        edges.forEach((e, i) => {
            const src = nodeMap[e.source];
            const tgt = nodeMap[e.target];
            if (src && tgt) {
                const isAmends = e.relation === 'amends' || e.relation === 'replaces' || e.relation === 'supplements';
                const strokeColor = isAmends ? '#f43f5e' : '#6366f1'; 
                svgHtml += `
                <g opacity="0">
                    <line x1="${src.x}" y1="${src.y}" x2="${tgt.x}" y2="${tgt.y}" 
                          stroke="${strokeColor}" stroke-width="1.5" stroke-opacity="0.5"
                          stroke-dasharray="4" stroke-dashoffset="100">
                        <animate attributeName="stroke-dashoffset" from="100" to="0" dur="1s" fill="freeze" />
                    </line>
                    <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="${i * 0.05}s" fill="freeze" />
                </g>`;
            }
        });

        const typeColors = {
            'law': '#ec4899',
            'decree': '#8b5cf6',
            'circular': '#3b82f6',
            'decision': '#0ea5e9',
            'default': '#94a3b8'
        };

        nodes.forEach((n, i) => {
            const color = typeColors[n.type] || typeColors['default'];
            const r = n.is_anchor ? 16 : 8;
            const filter = n.is_anchor ? 'filter="url(#kg-glow)"' : '';
            
            svgHtml += `
            <g opacity="0" style="transform-origin: ${n.x}px ${n.y}px">
                <circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${color}" fill-opacity="0.2" stroke="${color}" stroke-width="2" ${filter}>
                </circle>
                ${n.is_anchor ? `<circle cx="${n.x}" cy="${n.y}" r="${r+4}" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="2 4" opacity="0.6">
                    <animateTransform attributeName="transform" type="rotate" from="0 ${n.x} ${n.y}" to="360 ${n.x} ${n.y}" dur="8s" repeatCount="indefinite"/>
                </circle>` : ''}
                <text x="${n.x}" y="${n.y + r + 14}" text-anchor="middle" font-size="9" fill="#e2e8f0" font-weight="bold"><title>${n.label}</title>${n.label.length > 30 ? n.label.substring(0, 27) + '...' : n.label}</text>
                <text x="${n.x}" y="${n.y + r + 26}" text-anchor="middle" font-size="7" fill="#818cf8" style="text-transform: uppercase;">${n.type || 'unknown'}</text>
                <animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="${i * 0.05}s" fill="freeze" />
            </g>`;
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
                    barThickness: 24,
                    maxBarThickness: 24,
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
        wrapper.className = 'max-w-4xl mx-auto w-full flex mb-6 chat-message';

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

    function addTypingIndicator(isScanning = false) {
        const id = 'typing-' + Date.now();
        const wrapper = document.createElement('div');
        wrapper.id = id;
        wrapper.className = 'w-full flex mb-6 chat-message';
        
        let contentHtml = '';
        if (isScanning) {
            contentHtml = `
                <div class="agent-message flex flex-col gap-2 p-4 w-64 border border-sky-200 bg-sky-50/50">
                    <div class="flex items-center gap-2 text-sky-700 text-sm font-bold">
                        <i class="fa-solid fa-camera-viewfinder fa-beat"></i> Đang quét dữ liệu...
                    </div>
                    <div class="w-full h-1 bg-slate-200 rounded-full overflow-hidden relative">
                        <div class="absolute top-0 left-0 h-full bg-sky-400 w-full animate-pulse"></div>
                    </div>
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider">OCR & Schema Detect</div>
                </div>
            `;
        } else {
            contentHtml = `
                <div class="agent-message flex items-center h-12 typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            `;
        }

        wrapper.innerHTML = `
            <div class="flex gap-4">
                <div class="w-10 h-10 rounded-full flex-shrink-0 overflow-hidden border border-slate-200">
                    <img src="../assets/img/${currentTheme}" alt="AI" class="w-full h-full object-cover grayscale opacity-70 agent-message-avatar">
                </div>
                ${contentHtml}
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
        
        if (lowerText.includes('luật') || lowerText.includes('quy định') || lowerText.includes('điều') || 
            lowerText.includes('thuế') || lowerText.includes('phạt') || lowerText.includes('lương') || lowerText.includes('hóa đơn')) {
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

    // Enhanced markdown parser with tables, headers, and citations
    function formatMarkdown(text, citations = []) {
        if (!text) return '';
        
        // Headers
        let html = text.replace(/^### (.*$)/gm, '<h4 class="font-bold text-slate-700 mt-3 mb-1">$1</h4>');
        html = html.replace(/^## (.*$)/gm, '<h3 class="font-bold text-slate-800 mt-4 mb-2 text-lg">$1</h3>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Citations [1], [2] — clickable to open modal
        html = html.replace(/\[(\d+)\]/g, (match, id) => {
            const idx = parseInt(id) - 1;
            const cit = citations[idx];
            if (cit) {
                const tooltipText = (cit.text || '').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;').substring(0, 200);
                const rawStatus = cit.effective_status || 'chưa xác định';
                const status = (typeof rawStatus === 'string' ? rawStatus : (rawStatus.status || 'chưa xác định')).toLowerCase();
                const statusColor = status.includes('còn hiệu lực') || status.includes('hiệu lực') && !status.includes('hết') ? 'text-emerald-400' : 
                                   (status.includes('hết hiệu lực') ? 'text-red-400' : 'text-slate-400');
                const statusIcon = status.includes('còn hiệu lực') || status.includes('hiệu lực') && !status.includes('hết') ? 'fa-check-circle' : 
                                  (status.includes('hết hiệu lực') ? 'fa-circle-xmark' : 'fa-circle-question');

                return `<span class="relative group citation-badge-clickable text-sky-600 bg-sky-50 px-1 py-0.5 rounded text-[10px] font-bold mx-0.5 border border-sky-200 hover:bg-sky-100 transition-colors" data-citation-idx="${idx}" onclick="event.stopPropagation(); window._openCitationModal && window._openCitationModal(${idx})">` +
                    `[${id}]` +
                    `<span class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block w-72 bg-slate-800 text-slate-100 text-[11px] p-2.5 rounded-lg shadow-2xl z-50 whitespace-normal text-left font-normal border border-slate-700 leading-relaxed pointer-events-none">` +
                        `<div class="flex justify-between items-start mb-1.5">` +
                            `<strong class="text-emerald-400 block font-mono text-[10px]"><i class="fa-solid fa-scale-balanced mr-1"></i>Trích dẫn [${id}] • Rerank: ${cit.cross_encoder_score ? cit.cross_encoder_score.toFixed(2) : (cit.score ? cit.score.toFixed(2) : 'N/A')}</strong>` +
                            `<span class="${statusColor} text-[9px] font-bold uppercase tracking-tighter flex items-center gap-1">` +
                                `<i class="fa-solid ${statusIcon}"></i> ${status}` +
                            `</span>` +
                        `</div>` +
                        `<div class="text-slate-200">${tooltipText}...</div>` +
                        `<span class="block mt-2 text-sky-300 text-[9px] border-t border-slate-700 pt-1">` +
                            `<i class="fa-solid fa-hand-pointer mr-1"></i> Bấm để xem toàn bộ văn bản` +
                        `</span>` +
                    `</span>` +
                `</span>`;
            }
            return `<span class="text-sky-600 bg-sky-50 px-1 rounded text-[10px] font-bold mx-0.5">[${id}]</span>`;
        });
        
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
    function renderLegalWorkspace(wsData) {
        const lwPanel = document.getElementById('legalWorkspacePanel');
        if (!lwPanel) return;

        // Auto-show ONLY if in legal mode and it was hidden
        if (currentModelMode === 'legal' && lwPanel.classList.contains('hidden')) {
            lwPanel.classList.remove('hidden');
            setTimeout(() => {
                lwPanel.classList.remove('translate-x-full', 'opacity-0');
            }, 10);
        }

        // Facts
        const factsEl = document.getElementById('lw-facts');
        if (factsEl && wsData.facts) {
            factsEl.innerHTML = wsData.facts.length ? wsData.facts.map(f => `<li>${f}</li>`).join('') : `<li class="italic text-slate-400 list-none -ml-4">Chưa có sự kiện</li>`;
        }
        
        // Assumptions
        const assumpEl = document.getElementById('lw-assumptions');
        if (assumpEl && wsData.assumptions) {
            assumpEl.innerHTML = wsData.assumptions.length ? wsData.assumptions.map(a => `<li>${a}</li>`).join('') : `<li class="italic text-slate-400 list-none -ml-4">Chưa có giả định</li>`;
        }

        // Open Questions
        const qEl = document.getElementById('lw-open-questions');
        if (qEl && wsData.open_questions) {
            qEl.innerHTML = wsData.open_questions.length ? wsData.open_questions.map(q => `<li>${q}</li>`).join('') : `<li class="italic text-slate-400 list-none -ml-4">Chưa có câu hỏi</li>`;
        }

        // Verifications
        const verEl = document.getElementById('lw-verifications');
        if (verEl && wsData.verifications) {
            if (!wsData.verifications.length) {
                verEl.innerHTML = `<div class="italic text-slate-400">Chưa có xác minh</div>`;
            } else {
                verEl.innerHTML = wsData.verifications.map(v => `
                    <div class="bg-slate-50 border border-slate-200 rounded p-2 text-[11px]">
                        <div class="font-bold text-slate-700">${v.claim || 'Nhận định'}</div>
                        <div class="mt-1 flex items-center gap-1 ${v.is_verified ? 'text-emerald-600' : 'text-amber-600'}">
                            <i class="fa-solid ${v.is_verified ? 'fa-check-circle' : 'fa-circle-exclamation'}"></i> 
                            ${v.is_verified ? 'Đã xác minh' : 'Chưa xác minh'}
                        </div>
                    </div>
                `).join('');
            }
        }

        // Escalations
        const escCont = document.getElementById('lw-escalations-container');
        const escEl = document.getElementById('lw-escalations');
        if (escCont && escEl && wsData.escalations) {
            if (wsData.escalations.length) {
                escCont.classList.remove('hidden');
                escEl.innerHTML = wsData.escalations.map(e => `<li><i class="fa-solid fa-triangle-exclamation mr-1"></i> ${e}</li>`).join('');
            } else {
                escCont.classList.add('hidden');
            }
        }

        // Cited Documents List
        const docsEl = document.getElementById('lw-documents');
        const citations = window.currentCitations || [];
        if (docsEl && citations.length) {
            docsEl.innerHTML = citations.map((c, idx) => {
                const status = (c.effective_status || 'chưa xác định').toLowerCase();
                const isEffective = status.includes('còn hiệu lực') || status.includes('hiệu lực') && !status.includes('hết');
                const isExpired = status.includes('hết hiệu lực');
                const statusColor = isEffective ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : (isExpired ? 'bg-red-50 text-red-600 border-red-200' : 'bg-slate-50 text-slate-500 border-slate-200');
                const statusIcon = isEffective ? 'fa-check-circle' : (isExpired ? 'fa-circle-xmark' : 'fa-circle-question');

                return `
                    <div class="p-2.5 rounded-lg border bg-white shadow-sm hover:shadow-md transition-shadow group cursor-pointer border-slate-200 hover:border-sky-300">
                        <div class="flex justify-between items-start mb-1">
                            <span class="w-5 h-5 flex items-center justify-center rounded bg-sky-100 text-sky-700 font-bold text-[9px]">[${idx+1}]</span>
                            <span class="text-[9px] px-1.5 py-0.5 rounded border ${statusColor} font-bold uppercase tracking-tighter flex items-center gap-1">
                                <i class="fa-solid ${statusIcon} text-[8px]"></i> ${status}
                            </span>
                        </div>
                        <div class="text-[11px] font-bold text-slate-700 leading-tight group-hover:text-sky-700 transition-colors">${c.title || c.citation_key}</div>
                        <div class="mt-1.5 flex items-center justify-between text-[9px] text-slate-400">
                            <span><i class="fa-solid fa-dna mr-1 opacity-60"></i>Tương quan: ${(c.score * 100).toFixed(0)}%</span>
                            <i class="fa-solid fa-arrow-up-right-from-square opacity-0 group-hover:opacity-100 transition-opacity"></i>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  CITATION MODAL — Show full document on click
    // ═══════════════════════════════════════════════════════════

    window._openCitationModal = function(idx) {
        const citations = window.currentCitations || [];
        const cit = citations[idx];
        if (!cit) return;

        const statusRaw = cit.effective_status || 'chưa xác định';
        const status = (typeof statusRaw === 'string' ? statusRaw : (statusRaw.status || 'chưa xác định')).toLowerCase();
        
        const isEffective = status.includes('còn hiệu lực') || status.includes('hiệu lực') && !status.includes('hết');
        const isExpired = status.includes('hết hiệu lực');
        
        const badgeClass = isEffective ? 'bg-emerald-100 text-emerald-700 border-emerald-200' : 
                          (isExpired ? 'bg-red-100 text-red-700 border-red-200' : 'bg-slate-100 text-slate-700 border-slate-200');
        const statusLabel = isEffective ? 'Còn hiệu lực' : (isExpired ? 'Hết hiệu lực' : 'Chưa xác định');

        // Extract metadata if available
        let docType = cit.doc_type || 'Văn bản pháp luật';
        let authority = cit.official_letter_scope?.agency || 'Bộ Tài chính > Tổng cục Thuế';
        if (cit.authority_path && cit.authority_path.length > 0) {
            authority = cit.authority_path.join(' > ');
        }
        let docNumber = cit.citation_key || '';
        if (docNumber.includes(':cite')) docNumber = docNumber.split(':cite')[0];

        // Format body text
        // Use full_text if available, else fallback to text
        let rawBody = cit.full_text || cit.text || cit.chunk_text || 'Không có nội dung chi tiết.';
        let targetText = cit.text || cit.chunk_text || '';
        
        // Escape HTML
        rawBody = rawBody.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        targetText = targetText.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Highlight the specific chunk within the full text
        let bodyHtml = rawBody;
        if (targetText && targetText.length > 20 && rawBody !== targetText) {
            // Find the snippet in the full text and wrap it
            // We use a simple indexOf after stripping some whitespace variations
            const snippet = targetText.substring(0, 100); 
            const idx = rawBody.indexOf(snippet);
            if (idx !== -1) {
                const endIdx = rawBody.indexOf(targetText.substring(targetText.length - 100), idx) + 100;
                if (endIdx > 100 && endIdx <= rawBody.length) {
                    const before = rawBody.substring(0, idx);
                    const match = rawBody.substring(idx, endIdx);
                    const after = rawBody.substring(endIdx);
                    bodyHtml = before + `<span class="bg-blue-100 text-blue-900 px-1 py-0.5 rounded shadow-sm inline-block my-1 border border-blue-200">` + match + `</span>` + after;
                }
            }
        } else if (rawBody === targetText) {
            // If the whole thing is the chunk, just highlight it all or don't
            bodyHtml = `<span class="bg-blue-100 text-blue-900 px-1 py-0.5 rounded shadow-sm block my-1 border border-blue-200">` + rawBody + `</span>`;
        }

        // Convert newlines to paragraphs
        bodyHtml = bodyHtml.split('\n').map(p => p.trim() ? `<p class="mb-3 leading-relaxed text-justify">${p}</p>` : '').join('');

        // Get user's last query for keyword highlighting within the text
        const lastUserMsg = document.querySelector('.user-message:last-of-type');
        const queryText = lastUserMsg ? lastUserMsg.textContent.trim() : '';
        const keywords = queryText.split(/[\s,]+/).filter(w => w.length > 2 && !/^(cho|của|với|các|những|không|được|bao|nhiêu|thì|vậy|phải|hay|nào|ạ|dạ|anh|chị|em|ơi|tôi|mà)$/i.test(w));
        keywords.forEach(kw => {
            const regex = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
            bodyHtml = bodyHtml.replace(regex, '<mark class="bg-yellow-200 text-yellow-900 font-medium px-0.5 rounded">$1</mark>');
        });

        const score = cit.cross_encoder_score ? cit.cross_encoder_score.toFixed(2) : (cit.score ? cit.score.toFixed(2) : 'N/A');

        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[999] flex items-center justify-center p-4 sm:p-6 opacity-0 transition-opacity duration-300';
        overlay.onclick = (e) => { if (e.target === overlay) close(); };

        const modalHtml = `
            <div class="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col transform scale-95 transition-transform duration-300 overflow-hidden relative">
                <!-- Header Actions -->
                <div class="absolute top-4 right-4 flex items-center gap-2 z-10">
                    <button class="w-8 h-8 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 transition-colors" onclick="this.closest('.fixed').remove()">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>

                <!-- Document Header -->
                <div class="px-8 pt-8 pb-4 border-b border-slate-100 bg-slate-50/50">
                    <div class="flex flex-wrap items-center gap-2 mb-6 text-[11px] font-medium font-mono">
                        <span class="px-2.5 py-1 rounded border ${badgeClass}">
                            <i class="fa-solid ${isEffective ? 'fa-check-circle' : (isExpired ? 'fa-xmark-circle' : 'fa-question-circle')} mr-1"></i>${statusLabel}
                        </span>
                        <span class="px-2.5 py-1 rounded border bg-sky-50 text-sky-700 border-sky-200">
                            Phạm vi: Toàn quốc
                        </span>
                        <span class="px-2.5 py-1 rounded border bg-slate-100 text-slate-600 border-slate-200">
                            ${authority}
                        </span>
                        <div class="flex-1"></div>
                        <span class="text-slate-400">
                            <i class="fa-solid fa-scale-balanced mr-1"></i> Điểm Rerank: ${score}
                        </span>
                    </div>

                    <div class="flex justify-between items-start text-center mb-6">
                        <div class="w-1/3">
                            <div class="font-bold text-slate-800 uppercase mb-1">Bộ Tài Chính</div>
                            <div class="w-12 h-[1px] bg-slate-800 mx-auto mb-2"></div>
                            <div class="text-sm text-slate-600">Số: ${docNumber.split('/').pop()}</div>
                        </div>
                        <div class="w-2/3">
                            <div class="font-bold text-slate-800 uppercase mb-1">Cộng Hòa Xã Hội Chủ Nghĩa Việt Nam</div>
                            <div class="font-bold text-slate-800 mb-1">Độc lập - Tự do - Hạnh phúc</div>
                            <div class="w-32 h-[1px] bg-slate-800 mx-auto"></div>
                        </div>
                    </div>

                    <div class="text-center mb-4">
                        <h2 class="text-xl font-bold text-slate-900 uppercase tracking-wide mb-2">${docType}</h2>
                        <h3 class="text-base font-bold text-slate-700 max-w-2xl mx-auto leading-snug">${cit.title || 'Hướng dẫn thực hiện pháp luật thuế'}</h3>
                    </div>
                </div>

                <!-- Document Body -->
                <div class="p-8 overflow-y-auto flex-1 bg-white text-slate-800 font-serif text-[15px]">
                    <div class="max-w-3xl mx-auto">
                        ${bodyHtml}
                    </div>
                </div>
            </div>`;
        
        overlay.innerHTML = modalHtml;
        document.body.appendChild(overlay);
        
        // Animate in
        requestAnimationFrame(() => {
            overlay.classList.remove('opacity-0');
            overlay.querySelector('div').classList.remove('scale-95');
        });

        // Close on Escape
        const close = () => {
            overlay.classList.add('opacity-0');
            overlay.querySelector('div').classList.add('scale-95');
            setTimeout(() => overlay.remove(), 300);
            document.removeEventListener('keydown', esc);
        };
        const esc = (e) => { if (e.key === 'Escape') close(); };
        document.addEventListener('keydown', esc);
    };

    // ═══════════════════════════════════════════════════════════
    //  GRAPHRAG FULLSCREEN — Expand graph to full viewport
    // ═══════════════════════════════════════════════════════════

    window._openKgFullscreen = function() {
        const kg = window._lastKgData;
        if (!kg) return;

        const overlay = document.createElement('div');
        overlay.className = 'kg-fullscreen-overlay';
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

        const graphId = 'kg-fullscreen-svg';
        overlay.innerHTML = `
            <div class="kg-fullscreen-modal" onclick="event.stopPropagation()">
                <div class="kg-fullscreen-header">
                    <div>
                        <i class="fa-solid fa-diagram-project mr-2"></i> Đồ Thị Tri Thức Pháp Luật
                        <span style="font-size:11px; color:#a5b4fc; margin-left:12px;">${kg.total_entities} Nodes • ${kg.total_relations} Edges</span>
                    </div>
                    <button class="kg-fullscreen-close" onclick="this.closest('.kg-fullscreen-overlay').remove()">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>
                <div class="kg-fullscreen-body">
                    <svg id="${graphId}" width="100%" height="100%" viewBox="-500 -350 1000 700"></svg>
                    <div class="kg-legend">
                        <div style="font-weight:700; margin-bottom:8px; font-size:11px;"><i class="fa-solid fa-palette mr-1"></i> Chú giải</div>
                        <div class="kg-legend-item"><div class="kg-legend-dot" style="border-color:#ec4899; background:rgba(236,72,153,0.2)"></div> Luật</div>
                        <div class="kg-legend-item"><div class="kg-legend-dot" style="border-color:#8b5cf6; background:rgba(139,92,246,0.2)"></div> Nghị định</div>
                        <div class="kg-legend-item"><div class="kg-legend-dot" style="border-color:#3b82f6; background:rgba(59,130,246,0.2)"></div> Thông tư</div>
                        <div class="kg-legend-item"><div class="kg-legend-dot" style="border-color:#0ea5e9; background:rgba(14,165,233,0.2)"></div> Quyết định</div>
                        <div class="kg-legend-item"><div class="kg-legend-dot" style="border-color:#94a3b8; background:rgba(148,163,184,0.2)"></div> Điều khoản</div>
                        <div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.15);">
                            <div class="kg-legend-item"><div style="width:20px;height:2px;background:#6366f1;border-radius:2px;flex-shrink:0"></div> Tham chiếu</div>
                            <div class="kg-legend-item"><div style="width:20px;height:2px;background:#f43f5e;border-radius:2px;flex-shrink:0"></div> Sửa đổi/Thay thế</div>
                        </div>
                    </div>
                </div>
                <div class="kg-fullscreen-footer">
                    <span><i class="fa-solid fa-bolt text-amber-400 mr-1"></i> Engine: ${kg.retrieval_tier || 'GraphRAG'}</span>
                    <span><i class="fa-solid fa-clock mr-1"></i> ${kg.latency_ms}ms</span>
                </div>
            </div>`;
        document.body.appendChild(overlay);

        // Render the graph at larger scale
        setTimeout(() => {
            const svg = document.getElementById(graphId);
            if (svg) renderKgFullscreen(svg, kg);
        }, 50);

        const esc = (e) => { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', esc); } };
        document.addEventListener('keydown', esc);
    };

    function renderKgFullscreen(svg, data) {
        const nodes = data.nodes || [];
        const edges = data.edges || [];
        if (!nodes.length) return;

        nodes.forEach((n, i) => {
            if (i === 0) {
                n.x = 0; n.y = 0;
            } else {
                const angle = i * 2.39996; // Golden angle
                const radius = 120 + Math.pow(i, 0.7) * 45; // Significantly increased spacing
                n.x = Math.cos(angle) * radius;
                n.y = Math.sin(angle) * radius * 0.75;
            }
        });

        const nodeMap = {};
        nodes.forEach(n => nodeMap[n.id] = n);

        const typeColors = { 'law':'#ec4899','decree':'#8b5cf6','circular':'#3b82f6','decision':'#0ea5e9','article':'#10b981','default':'#94a3b8' };

        let h = `<defs>
            <filter id="kg-glow2" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>`;

        edges.forEach((e, i) => {
            const s = nodeMap[e.source], t = nodeMap[e.target];
            if (s && t) {
                const isA = e.relation==='amends'||e.relation==='replaces'||e.relation==='supplements';
                h += `<g opacity="0"><line x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" stroke="${isA?'#f43f5e':'#6366f1'}" stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="${isA?'6 3':'4'}"><animate attributeName="stroke-dashoffset" from="100" to="0" dur="1.2s" fill="freeze"/></line><animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="${i*0.03}s" fill="freeze"/></g>`;
            }
        });

        nodes.forEach((n, i) => {
            const c = typeColors[n.type]||typeColors.default;
            const r = n.is_anchor ? 32 : 18;
            const f = n.is_anchor ? 'filter="url(#kg-glow2)"' : '';
            const label = n.label.length > 50 ? n.label.substring(0, 47) + '...' : n.label;
            h += `<g opacity="0"><circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${c}" fill-opacity="0.25" stroke="${c}" stroke-width="3" ${f}/>`;
            if (n.is_anchor) h += `<circle cx="${n.x}" cy="${n.y}" r="${r+8}" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="4 6" opacity="0.6"><animateTransform attributeName="transform" type="rotate" from="0 ${n.x} ${n.y}" to="360 ${n.x} ${n.y}" dur="12s" repeatCount="indefinite"/></circle>`;
            h += `<text x="${n.x}" y="${n.y+r+18}" text-anchor="middle" font-size="12" fill="#e2e8f0" font-weight="700"><title>${n.label}</title>${label}</text>`;
            h += `<text x="${n.x}" y="${n.y+r+32}" text-anchor="middle" font-size="9" fill="${c}" style="text-transform:uppercase;font-weight:600;opacity:0.9">${n.type||'unknown'}</text>`;
            h += `<animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="${i*0.04}s" fill="freeze"/></g>`;
        });

        svg.innerHTML = h;
    }

});
