/**
 * login.js – Authentication Logic (Security Hardened + 2FA Signature)
 * =====================================================================
 * Features:
 *   1. Cookie-based auth via secureFetch()
 *   2. Face/CCCD login with 2FA signature step when required
 *   3. Canvas-based signature drawing for login verification
 */

let authToastTimer = null;
let registerSuccessTimer = null;
let currentTempToken = null; // Stores temp token between 2FA steps

const ROLE_LABELS = {
    viewer: 'Chuyên viên',
    analyst: 'Phân tích',
    inspector: 'Thanh tra',
    admin: 'Quản trị',
};

function showAuthToast(message, type = 'error') {
    const toast = document.getElementById('auth-toast');
    if (!toast) { alert(message); return; }
    if (authToastTimer) clearTimeout(authToastTimer);
    toast.textContent = message;
    toast.classList.remove('auth-toast-success', 'auth-toast-error', 'show');
    toast.classList.add(type === 'success' ? 'auth-toast-success' : 'auth-toast-error');
    requestAnimationFrame(() => toast.classList.add('show'));
    authToastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}

function showRegisterSuccessPanel({ badgeId, role }) {
    const panel = document.getElementById('register-success-panel');
    const text = document.getElementById('register-success-text');
    const roleNode = document.getElementById('register-success-role');
    if (!panel || !text || !roleNode) return;
    if (registerSuccessTimer) clearTimeout(registerSuccessTimer);
    text.textContent = `Mã cán bộ ${badgeId} đã được tạo. Đang chuyển về tab Đăng nhập...`;
    roleNode.textContent = ROLE_LABELS[role] || role || 'Cán bộ';
    panel.classList.remove('hidden');
    requestAnimationFrame(() => panel.classList.add('show'));
    registerSuccessTimer = setTimeout(() => {
        panel.classList.remove('show');
        setTimeout(() => panel.classList.add('hidden'), 220);
    }, 3600);
}

function showApiOfflineMessage() {
    showAuthToast('API backend chưa chạy trên cổng 8000. Hãy chạy: uvicorn app.main:app --reload --port 8000');
}

function openForgotModal() {
    const modal = document.getElementById('forgot-modal');
    const input = document.getElementById('forgot-email-input');
    const errorEl = document.getElementById('forgot-error');
    if (!modal) return;
    modal.classList.add('active');
    if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.add('hidden');
    }
    if (input) {
        input.value = '';
        setTimeout(() => input.focus(), 200);
    }
}

function closeForgotModal() {
    const modal = document.getElementById('forgot-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

async function submitForgotPassword() {
    const input = document.getElementById('forgot-email-input');
    const errorEl = document.getElementById('forgot-error');
    const btn = document.getElementById('forgot-submit-btn');
    if (!input || !errorEl || !btn) return;

    const email = input.value.trim().toLowerCase();
    if (!email || !email.includes('@')) {
        errorEl.textContent = 'Vui lòng nhập email công vụ hợp lệ.';
        errorEl.classList.remove('hidden');
        return;
    }

    errorEl.classList.add('hidden');
    const restore = setButtonLoading(btn, 'Đang gửi...');

    try {
        const response = await secureFetch(`${API_BASE}/auth/forgot-password`, {
            method: 'POST',
            body: JSON.stringify({ email }),
        });

        if (!response.ok) {
            errorEl.textContent = await parseApiError(response, 'Không thể gửi yêu cầu khôi phục lúc này.');
            errorEl.classList.remove('hidden');
            return;
        }

        const data = await response.json().catch(() => ({ message: 'Yêu cầu đã được ghi nhận.' }));
        closeForgotModal();
        showAuthToast(data.message || 'Yêu cầu khôi phục đã được ghi nhận.', 'success');
    } catch (err) {
        console.error('[ForgotPassword]', err);
        showApiOfflineMessage();
    } finally {
        restore();
    }
}

function enhanceAuthControls() {
    const controls = document.querySelectorAll('#forms-wrapper input, #forms-wrapper select');
    controls.forEach((control) => {
        control.classList.add('auth-control');
        const syncValueState = () => {
            const value = typeof control.value === 'string' ? control.value.trim() : control.value;
            control.classList.toggle('has-value', Boolean(value));
        };
        syncValueState();
        control.addEventListener('input', () => {
            syncValueState();
            control.classList.add('is-typing');
            if (control._typingTimer) clearTimeout(control._typingTimer);
            control._typingTimer = setTimeout(() => control.classList.remove('is-typing'), 220);
        });
        control.addEventListener('blur', () => { control.classList.remove('is-typing'); syncValueState(); });
    });
}

function setButtonLoading(button, loadingText) {
    const orig = button.innerHTML;
    button.innerHTML = `<div class="loader" style="width:20px;height:20px;border-width:2px;border-top-color:white;border-left-color:transparent;border-bottom-color:transparent;border-right-color:transparent;display:inline-block;"></div> ${loadingText}`;
    button.disabled = true;
    return () => { button.innerHTML = orig; button.disabled = false; };
}

async function parseApiError(response, fallback) {
    try { const d = await response.json(); return d.detail || fallback; } catch { return fallback; }
}

document.addEventListener('DOMContentLoaded', () => {
    enhanceAuthControls();
    initLoginSigCanvas();

    // ---- LOGIN LOGIC ----
    const loginForm = document.getElementById('form-login');
    const loginBtn = document.getElementById('login-btn');
    if (loginForm && loginBtn) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const badgeId = document.getElementById('login-badge').value.trim();
            const password = document.getElementById('login-pwd').value;
            if (!badgeId || !password) { showAuthToast('Vui lòng nhập đầy đủ Mã cán bộ và Mật khẩu.'); return; }
            const restore = setButtonLoading(loginBtn, 'Đang xác thực...');
            try {
                const r = await secureFetch(`${API_BASE}/auth/login`, {
                    method: 'POST', body: JSON.stringify({ badge_id: badgeId, password }),
                });
                if (!r.ok) { showAuthToast(await parseApiError(r, 'Đăng nhập thất bại.')); return; }
                showAuthToast('Đăng nhập thành công, đang chuyển hướng...', 'success');
                setTimeout(() => { window.location.href = 'dashboard.html'; }, 380);
            } catch { showApiOfflineMessage(); }
            finally { restore(); }
        });
    }

    // ---- REGISTER LOGIC ----
    const registerForm = document.getElementById('form-register');
    const registerBtn = document.getElementById('register-btn');
    if (registerForm && registerBtn) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fullName = document.getElementById('reg-fullname').value.trim();
            const badgeId = document.getElementById('reg-badge').value.trim();
            const email = document.getElementById('reg-email').value.trim().toLowerCase();
            const dept = document.getElementById('reg-dept').value;
            const role = document.getElementById('reg-role').value;
            const pwd = document.getElementById('reg-pwd').value;
            const pwd2 = document.getElementById('reg-pwd2').value;
            if (!fullName || !badgeId || !email || !dept || !pwd) { showAuthToast('Vui lòng điền đầy đủ các trường thông tin bắt buộc.'); return; }
            if (pwd.length < 8) { showAuthToast('Mật khẩu phải có ít nhất 8 ký tự.'); return; }
            if (pwd !== pwd2) { showAuthToast('Mật khẩu xác nhận không khớp.'); return; }
            if (!email.endsWith('@gdt.gov.vn')) { showAuthToast('Email phải là email công vụ (@gdt.gov.vn).'); return; }
            const restore = setButtonLoading(registerBtn, 'Đang khởi tạo...');
            try {
                const r = await secureFetch(`${API_BASE}/auth/register`, {
                    method: 'POST', body: JSON.stringify({ badge_id: badgeId, full_name: fullName, department: dept, email, password: pwd, role }),
                });
                if (!r.ok) { showAuthToast(await parseApiError(r, 'Lỗi tạo tài khoản.')); return; }
                showRegisterSuccessPanel({ badgeId, role });
                registerForm.reset();
                const loginBadge = document.getElementById('login-badge');
                if (loginBadge) { loginBadge.value = badgeId; loginBadge.dispatchEvent(new Event('input')); }
                setTimeout(() => {
                    if (typeof window.switchTab === 'function') window.switchTab('login');
                    const loginPwd = document.getElementById('login-pwd');
                    if (loginPwd) loginPwd.focus();
                    showAuthToast('Đăng ký thành công. Vui lòng đăng nhập.', 'success');
                }, 1150);
            } catch { showApiOfflineMessage(); }
            finally { restore(); }
        });
    }
});


// =============================================================================
// BIOMETRIC LOGIN: FACE
// =============================================================================

let bioStream = null;
let faceApiModelsLoaded = false;
const FACE_API_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/';

async function loadFaceApiModels() {
    if (faceApiModelsLoaded) return;
    await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(FACE_API_MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODEL_URL);
    faceApiModelsLoaded = true;
}

async function startCamera() {
    const video = document.getElementById('bio-video');
    try {
        bioStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
        video.srcObject = bioStream;
        await video.play();
        return true;
    } catch { return false; }
}

function stopCamera() {
    if (bioStream) { bioStream.getTracks().forEach(t => t.stop()); bioStream = null; }
    const v = document.getElementById('bio-video');
    if (v) v.srcObject = null;
}

async function openFaceLogin() {
    const overlay = document.getElementById('bio-overlay');
    const statusEl = document.getElementById('bio-status');
    const captureBtn = document.getElementById('bio-capture-btn');
    overlay.classList.add('active');
    statusEl.textContent = 'Đang xin quyền truy cập camera...';
    captureBtn.disabled = true;
    if (!(await startCamera())) { statusEl.textContent = '❌ Không thể truy cập camera.'; return; }
    statusEl.textContent = 'Đang tải mô hình AI nhận diện...';
    try { await loadFaceApiModels(); } catch (err) { statusEl.textContent = '❌ ' + err.message; return; }
    statusEl.textContent = '✓ Sẵn sàng. Đưa khuôn mặt vào khung hình và nhấn Xác nhận.';
    captureBtn.disabled = false;
}

function closeBioOverlay() {
    stopCamera();
    document.getElementById('bio-overlay').classList.remove('active');
}

async function captureFaceLogin() {
    const statusEl = document.getElementById('bio-status');
    const captureBtn = document.getElementById('bio-capture-btn');
    const video = document.getElementById('bio-video');
    captureBtn.disabled = true;
    statusEl.textContent = '🔍 Đang phân tích khuôn mặt...';
    try {
        const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks(true).withFaceDescriptor();
        if (!detection) { statusEl.textContent = '❌ Không phát hiện khuôn mặt.'; captureBtn.disabled = false; return; }
        statusEl.textContent = '🔐 Đang xác thực với hệ thống...';
        const scanLine = document.querySelector('.scan-line');
        if (scanLine) scanLine.style.animationDuration = '0.6s';

        const response = await secureFetch(`${API_BASE}/auth/login-face`, {
            method: 'POST', body: JSON.stringify({ descriptor: Array.from(detection.descriptor) }),
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            statusEl.textContent = '❌ ' + (data.detail || 'Xác thực khuôn mặt thất bại.');
            captureBtn.disabled = false;
            if (scanLine) scanLine.style.animationDuration = '';
            return;
        }

        const data = await response.json();

        // Check if 2FA signature is required
        if (data.require_signature) {
            currentTempToken = data.temp_token;
            statusEl.textContent = '✅ Khuôn mặt xác thực. Chuyển sang bước ký tên...';
            if (scanLine) { scanLine.style.background = 'linear-gradient(90deg, transparent, #fbbf24, transparent)'; }
            setTimeout(() => {
                closeBioOverlay();
                if (scanLine) { scanLine.style.background = ''; scanLine.style.animationDuration = ''; }
                openLoginSigModal();
            }, 1200);
            return;
        }

        // No 2FA → login success
        statusEl.textContent = '✅ Xác thực thành công! Đang chuyển hướng...';
        if (scanLine) { scanLine.style.background = 'linear-gradient(90deg, transparent, #4ade80, transparent)'; scanLine.style.boxShadow = '0 0 30px rgba(74,222,128,0.7)'; }
        setTimeout(() => { closeBioOverlay(); window.location.href = 'dashboard.html'; }, 1200);
    } catch (err) {
        console.error('[FaceLogin]', err);
        statusEl.textContent = '❌ Lỗi kết nối API.';
        captureBtn.disabled = false;
    }
}


// =============================================================================
// BIOMETRIC LOGIN: CCCD
// =============================================================================

function openCccdLogin() {
    document.getElementById('cccd-modal').classList.add('active');
    document.getElementById('cccd-login-error').classList.add('hidden');
    document.getElementById('cccd-login-input').value = '';
    setTimeout(() => document.getElementById('cccd-login-input').focus(), 300);
}

function closeCccdModal() {
    document.getElementById('cccd-modal').classList.remove('active');
}

async function submitCccdLogin() {
    const cccd = document.getElementById('cccd-login-input').value.trim().replace(/\s/g, '');
    const errorEl = document.getElementById('cccd-login-error');
    const btn = document.getElementById('cccd-login-btn');
    if (!cccd || cccd.length < 9) { errorEl.textContent = 'Nhập số CCCD hợp lệ (9-12 ký tự).'; errorEl.classList.remove('hidden'); return; }
    errorEl.classList.add('hidden');
    btn.disabled = true;
    btn.innerHTML = '<div class="loader" style="width:18px;height:18px;border-width:2px;border-top-color:white;display:inline-block;"></div> Đang xác thực...';

    try {
        const response = await secureFetch(`${API_BASE}/auth/login-cccd`, {
            method: 'POST', body: JSON.stringify({ cccd_number: cccd }),
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            errorEl.textContent = data.detail || 'Số CCCD không tìm thấy.';
            errorEl.classList.remove('hidden');
            return;
        }

        const data = await response.json();

        // Check if 2FA signature is required
        if (data.require_signature) {
            currentTempToken = data.temp_token;
            closeCccdModal();
            showAuthToast('CCCD xác thực thành công. Vui lòng ký tên xác nhận...', 'success');
            setTimeout(() => openLoginSigModal(), 500);
            return;
        }

        // No 2FA → login success
        errorEl.classList.add('hidden');
        showAuthToast('Đăng nhập CCCD thành công!', 'success');
        setTimeout(() => { closeCccdModal(); window.location.href = 'dashboard.html'; }, 800);
    } catch (err) {
        console.error('[CCCD Login]', err);
        errorEl.textContent = 'Lỗi kết nối API backend.';
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined text-base">login</span> Xác nhận đăng nhập';
    }
}


// =============================================================================
// 2FA: LOGIN SIGNATURE CANVAS
// =============================================================================

let loginSigCanvas, loginSigCtx, loginSigDrawing = false, loginSigHasContent = false;

function initLoginSigCanvas() {
    loginSigCanvas = document.getElementById('login-sig-canvas');
    if (!loginSigCanvas) return;
    loginSigCtx = loginSigCanvas.getContext('2d');
    loginSigCtx.strokeStyle = '#1a1c1e';
    loginSigCtx.lineWidth = 2.5;
    loginSigCtx.lineCap = 'round';
    loginSigCtx.lineJoin = 'round';

    loginSigCanvas.addEventListener('mousedown', loginSigStart);
    loginSigCanvas.addEventListener('mousemove', loginSigMove);
    loginSigCanvas.addEventListener('mouseup', loginSigEnd);
    loginSigCanvas.addEventListener('mouseleave', loginSigEnd);
    loginSigCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); loginSigStart(e.touches[0]); });
    loginSigCanvas.addEventListener('touchmove', (e) => { e.preventDefault(); loginSigMove(e.touches[0]); });
    loginSigCanvas.addEventListener('touchend', loginSigEnd);
}

function loginSigStart(e) {
    loginSigDrawing = true;
    const rect = loginSigCanvas.getBoundingClientRect();
    loginSigCtx.beginPath();
    loginSigCtx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function loginSigMove(e) {
    if (!loginSigDrawing) return;
    const rect = loginSigCanvas.getBoundingClientRect();
    loginSigCtx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    loginSigCtx.stroke();
    loginSigHasContent = true;
}

function loginSigEnd() { loginSigDrawing = false; }

function clearLoginSigCanvas() {
    if (!loginSigCtx) return;
    loginSigCtx.clearRect(0, 0, loginSigCanvas.width, loginSigCanvas.height);
    loginSigHasContent = false;
}

function openLoginSigModal() {
    document.getElementById('login-sig-modal').classList.add('active');
    clearLoginSigCanvas();
    document.getElementById('login-sig-error').classList.add('hidden');
}

function closeLoginSigModal() {
    document.getElementById('login-sig-modal').classList.remove('active');
    currentTempToken = null;
}

async function submitLoginSignature() {
    const errorEl = document.getElementById('login-sig-error');
    const btn = document.getElementById('login-sig-submit-btn');

    if (!loginSigHasContent) {
        errorEl.textContent = 'Vui lòng ký tên trước khi xác nhận.';
        errorEl.classList.remove('hidden');
        return;
    }

    if (!currentTempToken) {
        errorEl.textContent = 'Phiên xác thực đã hết hạn. Vui lòng thử lại từ đầu.';
        errorEl.classList.remove('hidden');
        return;
    }

    errorEl.classList.add('hidden');
    btn.disabled = true;
    const orig = btn.innerHTML;
    btn.innerHTML = '<div class="loader" style="width:18px;height:18px;border-width:2px;border-top-color:white;display:inline-block;"></div> Đang xác thực...';

    try {
        const sigImage = loginSigCanvas.toDataURL('image/png');
        const response = await secureFetch(`${API_BASE}/auth/login-signature`, {
            method: 'POST',
            body: JSON.stringify({
                temp_token: currentTempToken,
                signature_image: sigImage,
            }),
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            errorEl.textContent = data.detail || 'Chữ ký không khớp.';
            errorEl.classList.remove('hidden');
            return;
        }

        // 2FA complete → login success!
        showAuthToast('Xác thực 2 bước hoàn tất! Đang chuyển hướng...', 'success');
        setTimeout(() => {
            closeLoginSigModal();
            window.location.href = 'dashboard.html';
        }, 800);
    } catch (err) {
        console.error('[SigLogin]', err);
        errorEl.textContent = 'Lỗi kết nối API.';
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = orig;
    }
}
