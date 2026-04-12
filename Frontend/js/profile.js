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

let profileToastTimer = null;
const AVATAR_MAX_BYTES = 5 * 1024 * 1024;
const AVATAR_ALLOWED_TYPES = ['image/png', 'image/jpeg'];

let avatarCameraStream = null;
let avatarPendingData = null;
let avatarUploadInProgress = false;

function showProfileToast(message, type = 'error') {
    const toast = document.getElementById('profile-toast');
    if (!toast) {
        alert(message);
        return;
    }

    if (profileToastTimer) {
        clearTimeout(profileToastTimer);
    }

    toast.textContent = message;
    toast.classList.remove('profile-toast-success', 'profile-toast-error', 'show');
    toast.classList.add(type === 'success' ? 'profile-toast-success' : 'profile-toast-error');

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    profileToastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3400);
}

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

    setProfileAvatar(user.avatar_data, initials);
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

    if (typeof applySidebarIdentity === 'function') {
        applySidebarIdentity(user);
    }

    updateFaceCard(user.face_verified);
    updateCccdCard(user.cccd_verified);
    updateSignatureCard(user.signature_verified);
}

function setProfileAvatar(avatarData, initials) {
    const avatarImg = document.getElementById('profile-avatar-image');
    const avatarText = document.getElementById('profile-avatar-initials');
    if (!avatarImg || !avatarText) return;

    if (avatarData) {
        avatarImg.src = avatarData;
        avatarImg.classList.remove('hidden');
        avatarText.classList.add('hidden');
        return;
    }

    avatarImg.removeAttribute('src');
    avatarImg.classList.add('hidden');
    avatarText.classList.remove('hidden');
    avatarText.textContent = initials || '--';
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


// =============================================================================
// AVATAR UPDATE (camera or local file)
// =============================================================================

function openAvatarPickerModal() {
    if (avatarUploadInProgress) return;
    const modal = document.getElementById('avatar-chooser-modal');
    if (modal) modal.classList.add('active');
}

function closeAvatarPickerModal() {
    const modal = document.getElementById('avatar-chooser-modal');
    if (modal) modal.classList.remove('active');
}

function triggerAvatarFileInput() {
    const input = document.getElementById('avatar-file-input');
    if (!input) return;
    input.click();
}

function handleAvatarFileSelect(event) {
    const file = event?.target?.files?.[0];
    if (!file) return;

    if (!AVATAR_ALLOWED_TYPES.includes(file.type)) {
        showProfileToast('Chỉ hỗ trợ ảnh PNG hoặc JPEG.');
        return;
    }

    if (file.size > AVATAR_MAX_BYTES) {
        showProfileToast('Dung lượng ảnh vượt quá giới hạn 5MB.');
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        const result = String(reader.result || '');
        if (!result.startsWith('data:image/')) {
            showProfileToast('Ảnh tải lên không hợp lệ.');
            return;
        }

        avatarPendingData = result;
        closeAvatarPickerModal();
        openAvatarPreviewModal(result);
    };
    reader.onerror = () => {
        showProfileToast('Không thể đọc ảnh từ máy tính.');
    };
    reader.readAsDataURL(file);
}

function openAvatarPreviewModal(dataUrl) {
    const modal = document.getElementById('avatar-preview-modal');
    const image = document.getElementById('avatar-preview-image');
    if (!modal || !image) return;

    image.src = dataUrl;
    modal.classList.add('active');
}

function closeAvatarPreviewModal(clearPending = true) {
    const modal = document.getElementById('avatar-preview-modal');
    if (modal) modal.classList.remove('active');
    if (clearPending) avatarPendingData = null;
}

async function confirmAvatarFileUpload() {
    if (!avatarPendingData) {
        showProfileToast('Vui lòng chọn ảnh trước khi lưu.');
        return;
    }

    const saved = await submitAvatarUpdate(avatarPendingData);
    if (saved) {
        closeAvatarPreviewModal();
    }
}

function setAvatarCameraStatus(message) {
    const statusEl = document.getElementById('avatar-camera-status');
    if (statusEl) statusEl.textContent = message;
}

function resetAvatarCameraControls() {
    const video = document.getElementById('avatar-camera-video');
    const preview = document.getElementById('avatar-camera-preview');
    const captureBtn = document.getElementById('avatar-capture-btn');
    const retakeBtn = document.getElementById('avatar-retake-btn');
    const confirmBtn = document.getElementById('avatar-camera-confirm-btn');

    if (video) video.classList.remove('hidden');
    if (preview) {
        preview.classList.add('hidden');
        preview.removeAttribute('src');
    }
    if (captureBtn) captureBtn.classList.remove('hidden');
    if (retakeBtn) retakeBtn.classList.add('hidden');
    if (confirmBtn) confirmBtn.classList.add('hidden');
}

async function startAvatarCamera() {
    const video = document.getElementById('avatar-camera-video');
    if (!video || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return false;
    }

    try {
        avatarCameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 640, facingMode: 'user' },
            audio: false,
        });
        video.srcObject = avatarCameraStream;
        await video.play();
        return true;
    } catch {
        return false;
    }
}

function stopAvatarCamera() {
    if (avatarCameraStream) {
        avatarCameraStream.getTracks().forEach(track => track.stop());
        avatarCameraStream = null;
    }

    const video = document.getElementById('avatar-camera-video');
    if (video) video.srcObject = null;
}

async function openAvatarCameraModal() {
    closeAvatarPickerModal();
    avatarPendingData = null;

    const modal = document.getElementById('avatar-camera-modal');
    if (!modal) return;

    resetAvatarCameraControls();
    modal.classList.add('active');
    setAvatarCameraStatus('Đang xin quyền truy cập camera...');

    const started = await startAvatarCamera();
    if (!started) {
        setAvatarCameraStatus('Không thể truy cập camera. Vui lòng chọn ảnh từ máy tính.');
        showProfileToast('Không thể truy cập camera.');
        return;
    }

    setAvatarCameraStatus('Camera đã sẵn sàng. Hãy nhấn Chụp ảnh.');
}

function closeAvatarCameraModal() {
    stopAvatarCamera();
    avatarPendingData = null;
    const modal = document.getElementById('avatar-camera-modal');
    if (modal) modal.classList.remove('active');
    resetAvatarCameraControls();
}

function captureAvatarFromCamera() {
    const video = document.getElementById('avatar-camera-video');
    const preview = document.getElementById('avatar-camera-preview');
    const captureBtn = document.getElementById('avatar-capture-btn');
    const retakeBtn = document.getElementById('avatar-retake-btn');
    const confirmBtn = document.getElementById('avatar-camera-confirm-btn');

    if (!video || video.readyState < 2) {
        showProfileToast('Camera chưa sẵn sàng. Vui lòng thử lại.');
        return;
    }

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const side = Math.min(sourceWidth, sourceHeight);
    const sx = (sourceWidth - side) / 2;
    const sy = (sourceHeight - side) / 2;

    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        showProfileToast('Không thể xử lý ảnh từ camera.');
        return;
    }

    ctx.drawImage(video, sx, sy, side, side, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    avatarPendingData = dataUrl;
    stopAvatarCamera();

    if (preview) {
        preview.src = dataUrl;
        preview.classList.remove('hidden');
    }
    video.classList.add('hidden');
    if (captureBtn) captureBtn.classList.add('hidden');
    if (retakeBtn) retakeBtn.classList.remove('hidden');
    if (confirmBtn) confirmBtn.classList.remove('hidden');

    setAvatarCameraStatus('Ảnh đã chụp. Bạn có thể chụp lại hoặc xác nhận lưu.');
}

async function retakeAvatarCamera() {
    avatarPendingData = null;
    resetAvatarCameraControls();

    setAvatarCameraStatus('Đang khởi động lại camera...');
    const started = await startAvatarCamera();
    if (!started) {
        setAvatarCameraStatus('Không thể truy cập camera. Vui lòng chọn ảnh từ máy tính.');
        showProfileToast('Không thể truy cập camera.');
        return;
    }

    setAvatarCameraStatus('Camera đã sẵn sàng. Hãy nhấn Chụp ảnh.');
}

async function confirmAvatarCameraUpload() {
    if (!avatarPendingData) {
        showProfileToast('Vui lòng chụp ảnh trước khi lưu.');
        return;
    }

    const saved = await submitAvatarUpdate(avatarPendingData);
    if (saved) {
        closeAvatarCameraModal();
    }
}

function setAvatarControlsDisabled(disabled) {
    const ids = [
        'avatar-capture-btn',
        'avatar-retake-btn',
        'avatar-camera-confirm-btn',
        'avatar-preview-confirm-btn',
    ];

    ids.forEach((id) => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = disabled;
    });
}

async function submitAvatarUpdate(avatarImage) {
    if (!avatarImage || avatarUploadInProgress) {
        return false;
    }

    avatarUploadInProgress = true;
    setAvatarControlsDisabled(true);

    try {
        const response = await secureFetch(`${API_BASE}/auth/update-avatar`, {
            method: 'PUT',
            body: JSON.stringify({ avatar_image: avatarImage }),
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            showProfileToast(data.detail || 'Không thể cập nhật ảnh đại diện.');
            return false;
        }

        showProfileToast(data.message || 'Cập nhật ảnh đại diện thành công.', 'success');
        await loadProfile();
        if (typeof hydrateSidebarIdentity === 'function') {
            hydrateSidebarIdentity({ forceRefresh: true });
        }
        return true;
    } catch (err) {
        console.error('[Avatar]', err);
        showProfileToast('Lỗi kết nối API. Vui lòng thử lại.');
        return false;
    } finally {
        avatarUploadInProgress = false;
        setAvatarControlsDisabled(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    initSigCanvas();

    const logoutBtn = document.getElementById('btn-logout');
    if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);

    const avatarFileInput = document.getElementById('avatar-file-input');
    if (avatarFileInput) {
        avatarFileInput.addEventListener('click', () => {
            avatarFileInput.value = '';
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        closeAvatarPickerModal();
        closeAvatarPreviewModal();
        closeAvatarCameraModal();
    });
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
// CHANGE PASSWORD
// =============================================================================

async function submitChangePassword() {
    const currentPassword = document.getElementById('current-password');
    const newPassword = document.getElementById('new-password');
    const confirmPassword = document.getElementById('confirm-password');
    const submitBtn = document.getElementById('change-password-btn');

    if (!currentPassword || !newPassword || !confirmPassword || !submitBtn) {
        return;
    }

    const currentValue = currentPassword.value;
    const newValue = newPassword.value;
    const confirmValue = confirmPassword.value;

    if (!currentValue || !newValue || !confirmValue) {
        showProfileToast('Vui lòng điền đầy đủ thông tin đổi mật khẩu.');
        return;
    }

    if (newValue.length < 8) {
        showProfileToast('Mật khẩu mới phải có ít nhất 8 ký tự.');
        return;
    }

    if (newValue !== confirmValue) {
        showProfileToast('Mật khẩu xác nhận không khớp.');
        return;
    }

    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<div class="loader" style="width:18px;height:18px;border-width:2px;border-top-color:white;display:inline-block;"></div> Đang cập nhật...';

    try {
        const response = await secureFetch(`${API_BASE}/auth/change-password`, {
            method: 'POST',
            body: JSON.stringify({
                current_password: currentValue,
                new_password: newValue,
                confirm_password: confirmValue,
            }),
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            showProfileToast(data.detail || 'Không thể đổi mật khẩu lúc này.');
            return;
        }

        currentPassword.value = '';
        newPassword.value = '';
        confirmPassword.value = '';
        showProfileToast('Đổi mật khẩu thành công.', 'success');
    } catch (err) {
        console.error('[ChangePassword]', err);
        showProfileToast('Lỗi kết nối API. Vui lòng thử lại.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
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
