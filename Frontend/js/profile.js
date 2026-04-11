/**
 * profile.js  –  Trang Tài khoản Cán bộ (Security Hardened + Signature 2FA)
 * ===========================================================================
 * Features:
 *   1. Cookie-based auth via secureFetch()
 *   2. Face / CCCD / Signature setup & reset
 *   3. Phone number inline edit
 *   4. Canvas-based signature drawing
 */

const ROLE_MAP = {
    viewer:    { label: 'Chuyên viên',  color: 'bg-blue-100 text-blue-700',    dot: 'bg-blue-500' },
    analyst:   { label: 'Phân tích',    color: 'bg-indigo-100 text-indigo-700', dot: 'bg-indigo-500' },
    inspector: { label: 'Thanh tra',    color: 'bg-amber-100 text-amber-700',   dot: 'bg-amber-500' },
    admin:     { label: 'Quản trị viên',color: 'bg-emerald-100 text-emerald-700', dot: 'bg-emerald-500' },
};

function redirectToLogin() {
    window.location.href = 'login.html';
}

async function loadProfile() {
    try {
        const res = await secureFetch(`${API_BASE}/auth/me`);
        if (res.status === 401) { redirectToLogin(); return; }
        if (!res.ok) throw new Error('Không thể tải thông tin tài khoản.');
        const user = await res.json();
        renderProfile(user);
    } catch (err) {
        console.error('[profile]', err);
        const errorEl = document.getElementById('profile-error');
        if (errorEl) errorEl.classList.remove('hidden');
    }
}

function renderProfile(user) {
    const role = ROLE_MAP[user.role] || ROLE_MAP.viewer;
    const initials = (user.full_name || '').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase();

    document.getElementById('profile-avatar-initials').textContent = initials;
    document.getElementById('profile-full-name').textContent = user.full_name;
    document.getElementById('profile-badge-id').textContent = user.badge_id;

    const roleBadge = document.getElementById('profile-role-badge');
    roleBadge.textContent = role.label;
    roleBadge.className = `inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-bold ${role.color}`;

    document.getElementById('info-badge').textContent = user.badge_id;
    document.getElementById('info-name').textContent = user.full_name;
    document.getElementById('info-email').textContent = user.email;
    document.getElementById('info-department').textContent = user.department;
    document.getElementById('info-phone').textContent = user.phone || 'Chưa cập nhật';
    document.getElementById('info-role').textContent = role.label;

    const createdAt = new Date(user.created_at);
    document.getElementById('info-created').textContent = createdAt.toLocaleDateString('vi-VN', {
        year: 'numeric', month: 'long', day: 'numeric',
    });

    const sidebarName = document.getElementById('user-full-name');
    const sidebarRole = document.getElementById('user-current-role');
    if (sidebarName) sidebarName.textContent = user.full_name;
    if (sidebarRole) sidebarRole.textContent = role.label;

    updateFaceCard(user.face_verified);
    updateCccdCard(user.cccd_verified);
    updateSignatureCard(user.signature_verified);
}

function updateFaceCard(verified) {
    document.getElementById('face-unverified').classList.toggle('hidden', verified);
    document.getElementById('face-verified').classList.toggle('hidden', !verified);
}

function updateCccdCard(verified) {
    document.getElementById('cccd-unverified').classList.toggle('hidden', verified);
    document.getElementById('cccd-verified').classList.toggle('hidden', !verified);
}

function updateSignatureCard(verified) {
    document.getElementById('sig-unverified').classList.toggle('hidden', verified);
    document.getElementById('sig-verified').classList.toggle('hidden', !verified);
}

function handleLogout() {
    logout();
}

document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    initSigCanvas();

    const logoutBtn = document.getElementById('btn-logout');
    if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);
});


// =============================================================================
// PHONE EDIT
// =============================================================================

function togglePhoneEdit() {
    const phoneSpan = document.getElementById('info-phone');
    const phoneInput = document.getElementById('phone-edit-input');
    const btnEdit = document.getElementById('btn-edit-phone');
    const btnSave = document.getElementById('btn-save-phone');
    const btnCancel = document.getElementById('btn-cancel-phone');

    phoneInput.value = phoneSpan.textContent === 'Chưa cập nhật' ? '' : phoneSpan.textContent;
    phoneSpan.classList.add('hidden');
    phoneInput.classList.remove('hidden');
    btnEdit.classList.add('hidden');
    btnSave.classList.remove('hidden');
    btnCancel.classList.remove('hidden');
    phoneInput.focus();
}

function cancelPhoneEdit() {
    document.getElementById('info-phone').classList.remove('hidden');
    document.getElementById('phone-edit-input').classList.add('hidden');
    document.getElementById('btn-edit-phone').classList.remove('hidden');
    document.getElementById('btn-save-phone').classList.add('hidden');
    document.getElementById('btn-cancel-phone').classList.add('hidden');
}

async function savePhone() {
    const phoneInput = document.getElementById('phone-edit-input');
    const phone = phoneInput.value.trim();

    if (!phone || phone.length < 10) {
        alert('Vui lòng nhập số điện thoại hợp lệ (10-11 chữ số).');
        return;
    }

    try {
        const res = await secureFetch(`${API_BASE}/auth/update-phone`, {
            method: 'PUT',
            body: JSON.stringify({ phone }),
        });

        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            alert(data.detail || 'Cập nhật thất bại.');
            return;
        }

        const data = await res.json();
        document.getElementById('info-phone').textContent = data.phone;
        cancelPhoneEdit();
    } catch (err) {
        console.error('[Phone]', err);
        alert('Lỗi kết nối. Vui lòng thử lại.');
    }
}


// =============================================================================
// FACE SETUP (Profile Page)
// =============================================================================

let profileBioStream = null;
let profileFaceApiLoaded = false;
const FACE_API_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/';

async function loadProfileFaceApi() {
    if (profileFaceApiLoaded) return;
    await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(FACE_API_MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODEL_URL);
    profileFaceApiLoaded = true;
}

async function startProfileCamera() {
    const video = document.getElementById('profile-bio-video');
    try {
        profileBioStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
        });
        video.srcObject = profileBioStream;
        await video.play();
        return true;
    } catch { return false; }
}

function stopProfileCamera() {
    if (profileBioStream) { profileBioStream.getTracks().forEach(t => t.stop()); profileBioStream = null; }
    const v = document.getElementById('profile-bio-video');
    if (v) v.srcObject = null;
}

async function openFaceSetup() {
    const overlay = document.getElementById('profile-bio-overlay');
    const statusEl = document.getElementById('profile-bio-status');
    const captureBtn = document.getElementById('profile-bio-capture-btn');
    overlay.classList.add('active');
    statusEl.textContent = 'Đang xin quyền truy cập camera...';
    captureBtn.disabled = true;
    if (!(await startProfileCamera())) { statusEl.textContent = '❌ Không thể truy cập camera.'; return; }
    statusEl.textContent = 'Đang tải mô hình AI nhận diện...';
    try { await loadProfileFaceApi(); } catch { statusEl.textContent = '❌ Không thể tải mô hình AI.'; return; }
    statusEl.textContent = '✓ Sẵn sàng. Đưa khuôn mặt vào khung hình rồi nhấn Xác nhận.';
    captureBtn.disabled = false;
}

function closeProfileBioOverlay() {
    stopProfileCamera();
    document.getElementById('profile-bio-overlay').classList.remove('active');
}

async function captureFaceSetup() {
    const statusEl = document.getElementById('profile-bio-status');
    const captureBtn = document.getElementById('profile-bio-capture-btn');
    const video = document.getElementById('profile-bio-video');
    captureBtn.disabled = true;
    statusEl.textContent = '🔍 Đang phân tích khuôn mặt...';
    try {
        const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks(true).withFaceDescriptor();
        if (!detection) { statusEl.textContent = '❌ Không phát hiện khuôn mặt.'; captureBtn.disabled = false; return; }
        statusEl.textContent = '🔐 Đang lưu vào hệ thống...';
        const response = await secureFetch(`${API_BASE}/auth/setup-face`, {
            method: 'POST', body: JSON.stringify({ descriptor: Array.from(detection.descriptor) }),
        });
        if (!response.ok) { const d = await response.json().catch(() => ({})); statusEl.textContent = '❌ ' + (d.detail || 'Thất bại.'); captureBtn.disabled = false; return; }
        statusEl.textContent = '✅ Thiết lập khuôn mặt thành công!';
        setTimeout(() => { closeProfileBioOverlay(); updateFaceCard(true); }, 1500);
    } catch (err) { statusEl.textContent = '❌ Lỗi kết nối Backend.'; captureBtn.disabled = false; }
}

async function resetFace() {
    if (!confirm('Xóa dữ liệu xác minh khuôn mặt?')) return;
    try { const r = await secureFetch(`${API_BASE}/auth/reset-face`, { method: 'DELETE' }); if (r.ok) updateFaceCard(false); } catch {}
}


// =============================================================================
// CCCD SETUP
// =============================================================================

function openCccdSetup() {
    document.getElementById('profile-cccd-modal').classList.add('active');
    document.getElementById('profile-cccd-error').classList.add('hidden');
    document.getElementById('profile-cccd-input').value = '';
    setTimeout(() => document.getElementById('profile-cccd-input').focus(), 300);
}

function closeProfileCccdModal() {
    document.getElementById('profile-cccd-modal').classList.remove('active');
}

async function submitCccdSetup() {
    const cccd = document.getElementById('profile-cccd-input').value.trim().replace(/\s/g, '');
    const errorEl = document.getElementById('profile-cccd-error');
    const btn = document.getElementById('profile-cccd-btn');
    if (!cccd || cccd.length < 9) { errorEl.textContent = 'Nhập số CCCD hợp lệ (9-12 ký tự).'; errorEl.classList.remove('hidden'); return; }
    errorEl.classList.add('hidden');
    btn.disabled = true;
    const orig = btn.innerHTML;
    btn.innerHTML = '<div class="loader" style="width:18px;height:18px;border-width:2px;border-top-color:white;display:inline-block;"></div> Đang xác minh...';
    try {
        const r = await secureFetch(`${API_BASE}/auth/setup-cccd`, { method: 'POST', body: JSON.stringify({ cccd_number: cccd }) });
        if (!r.ok) { const d = await r.json().catch(() => ({})); errorEl.textContent = d.detail || 'Thất bại.'; errorEl.classList.remove('hidden'); return; }
        closeProfileCccdModal(); updateCccdCard(true);
    } catch { errorEl.textContent = 'Lỗi kết nối.'; errorEl.classList.remove('hidden'); }
    finally { btn.disabled = false; btn.innerHTML = orig; }
}

async function resetCccd() {
    if (!confirm('Xóa dữ liệu xác minh CCCD?')) return;
    try { const r = await secureFetch(`${API_BASE}/auth/reset-cccd`, { method: 'DELETE' }); if (r.ok) updateCccdCard(false); } catch {}
}


// =============================================================================
// SIGNATURE SETUP (Canvas Drawing)
// =============================================================================

let sigCanvas, sigCtx, sigDrawing = false, sigHasContent = false;

function initSigCanvas() {
    sigCanvas = document.getElementById('sig-canvas');
    if (!sigCanvas) return;
    sigCtx = sigCanvas.getContext('2d');
    sigCtx.strokeStyle = '#1a1c1e';
    sigCtx.lineWidth = 2.5;
    sigCtx.lineCap = 'round';
    sigCtx.lineJoin = 'round';

    // Mouse events
    sigCanvas.addEventListener('mousedown', sigStart);
    sigCanvas.addEventListener('mousemove', sigMove);
    sigCanvas.addEventListener('mouseup', sigEnd);
    sigCanvas.addEventListener('mouseleave', sigEnd);

    // Touch events
    sigCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); sigStart(e.touches[0]); });
    sigCanvas.addEventListener('touchmove', (e) => { e.preventDefault(); sigMove(e.touches[0]); });
    sigCanvas.addEventListener('touchend', sigEnd);
}

function sigStart(e) {
    sigDrawing = true;
    const rect = sigCanvas.getBoundingClientRect();
    sigCtx.beginPath();
    sigCtx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function sigMove(e) {
    if (!sigDrawing) return;
    const rect = sigCanvas.getBoundingClientRect();
    sigCtx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    sigCtx.stroke();
    sigHasContent = true;
}

function sigEnd() {
    sigDrawing = false;
}

function clearSigCanvas() {
    if (!sigCtx) return;
    sigCtx.clearRect(0, 0, sigCanvas.width, sigCanvas.height);
    sigHasContent = false;
}

function getSigCanvasBase64() {
    return sigCanvas.toDataURL('image/png');
}

function openSignatureSetup() {
    const modal = document.getElementById('profile-sig-modal');
    modal.classList.add('active');
    clearSigCanvas();
    document.getElementById('sig-error').classList.add('hidden');
}

function closeSignatureModal() {
    document.getElementById('profile-sig-modal').classList.remove('active');
}

async function submitSignatureSetup() {
    const errorEl = document.getElementById('sig-error');
    const btn = document.getElementById('sig-submit-btn');

    if (!sigHasContent) {
        errorEl.textContent = 'Vui lòng ký tên trước khi lưu.';
        errorEl.classList.remove('hidden');
        return;
    }

    errorEl.classList.add('hidden');
    btn.disabled = true;
    const orig = btn.innerHTML;
    btn.innerHTML = '<div class="loader" style="width:18px;height:18px;border-width:2px;border-top-color:white;display:inline-block;"></div> Đang lưu...';

    try {
        const sigImage = getSigCanvasBase64();
        const r = await secureFetch(`${API_BASE}/auth/setup-signature`, {
            method: 'POST',
            body: JSON.stringify({ signature_image: sigImage }),
        });

        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            errorEl.textContent = d.detail || 'Thiết lập chữ ký thất bại.';
            errorEl.classList.remove('hidden');
            return;
        }

        closeSignatureModal();
        updateSignatureCard(true);
    } catch {
        errorEl.textContent = 'Lỗi kết nối API.';
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = orig;
    }
}

async function resetSignature() {
    if (!confirm('Xóa dữ liệu chữ ký số? Đăng nhập bằng khuôn mặt/CCCD sẽ không yêu cầu ký tên nữa.')) return;
    try {
        const r = await secureFetch(`${API_BASE}/auth/reset-signature`, { method: 'DELETE' });
        if (r.ok) updateSignatureCard(false);
    } catch {}
}
