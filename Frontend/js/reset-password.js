let resetToastTimer = null;

function showResetToast(message, type = 'error') {
    const toast = document.getElementById('reset-toast');
    if (!toast) {
        alert(message);
        return;
    }

    if (resetToastTimer) {
        clearTimeout(resetToastTimer);
    }

    toast.textContent = message;
    toast.classList.remove('reset-toast-success', 'reset-toast-error', 'show');
    toast.classList.add(type === 'success' ? 'reset-toast-success' : 'reset-toast-error');

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    resetToastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3600);
}

async function parseApiError(response, fallback) {
    try {
        const data = await response.json();
        return data.detail || fallback;
    } catch {
        return fallback;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token') || '';

    const warning = document.getElementById('token-warning');
    const form = document.getElementById('reset-form');
    const submitBtn = document.getElementById('reset-submit-btn');

    if (!token) {
        if (warning) {
            warning.classList.remove('hidden');
        }
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
        return;
    }

    if (!form || !submitBtn) {
        return;
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('confirm-password').value;

        if (!newPassword || newPassword.length < 8) {
            showResetToast('Mật khẩu mới phải có ít nhất 8 ký tự.');
            return;
        }

        if (newPassword !== confirmPassword) {
            showResetToast('Mật khẩu xác nhận không khớp.');
            return;
        }

        const originalText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = 'Đang cập nhật...';

        try {
            const response = await fetch(`${API_BASE}/auth/reset-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    token,
                    new_password: newPassword,
                    confirm_password: confirmPassword,
                }),
            });

            if (!response.ok) {
                showResetToast(await parseApiError(response, 'Không thể đặt lại mật khẩu.'));
                return;
            }

            showResetToast('Đặt lại mật khẩu thành công. Đang chuyển về đăng nhập...', 'success');
            form.reset();
            setTimeout(() => {
                window.location.href = 'login.html';
            }, 1200);
        } catch (err) {
            console.error('[ResetPassword]', err);
            showResetToast('Không kết nối được máy chủ API (http://localhost:8000).');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    });
});
