let authToastTimer = null;
let registerSuccessTimer = null;

const ROLE_LABELS = {
    viewer: 'Chuyên viên',
    analyst: 'Phân tích',
    inspector: 'Thanh tra',
    admin: 'Quản trị',
};

function showAuthToast(message, type = 'error') {
    const toast = document.getElementById('auth-toast');
    if (!toast) {
        alert(message);
        return;
    }

    if (authToastTimer) {
        clearTimeout(authToastTimer);
    }

    toast.textContent = message;
    toast.classList.remove('auth-toast-success', 'auth-toast-error', 'show');
    toast.classList.add(type === 'success' ? 'auth-toast-success' : 'auth-toast-error');

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    authToastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3200);
}

function showRegisterSuccessPanel({ badgeId, role }) {
    const panel = document.getElementById('register-success-panel');
    const text = document.getElementById('register-success-text');
    const roleNode = document.getElementById('register-success-role');

    if (!panel || !text || !roleNode) {
        return;
    }

    if (registerSuccessTimer) {
        clearTimeout(registerSuccessTimer);
    }

    const safeRole = ROLE_LABELS[role] || role || 'Cán bộ';
    text.textContent = `Mã cán bộ ${badgeId} đã được tạo. Đang chuyển về tab Đăng nhập...`;
    roleNode.textContent = safeRole;

    panel.classList.remove('hidden');
    requestAnimationFrame(() => {
        panel.classList.add('show');
    });

    registerSuccessTimer = setTimeout(() => {
        panel.classList.remove('show');
        setTimeout(() => {
            panel.classList.add('hidden');
        }, 220);
    }, 3600);
}

function showApiOfflineMessage() {
    showAuthToast('Backend API chưa chạy trên cổng 8000. Hãy chạy: uvicorn app.main:app --reload --port 8000');
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

            if (control._typingTimer) {
                clearTimeout(control._typingTimer);
            }

            control._typingTimer = setTimeout(() => {
                control.classList.remove('is-typing');
            }, 220);
        });

        control.addEventListener('blur', () => {
            control.classList.remove('is-typing');
            syncValueState();
        });
    });
}

function setButtonLoading(button, loadingText) {
    const originalText = button.innerHTML;
    button.innerHTML = `<div class="loader" style="width: 20px; height: 20px; border-width: 2px; border-top-color: white; border-left-color: transparent; border-bottom-color: transparent; border-right-color: transparent; display: inline-block;"></div> ${loadingText}`;
    button.disabled = true;
    return () => {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

async function parseApiError(response, fallbackMessage) {
    try {
        const data = await response.json();
        return data.detail || fallbackMessage;
    } catch {
        return fallbackMessage;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    enhanceAuthControls();

    // ---- LOGIN LOGIC ----
    const loginForm = document.getElementById('form-login');
    const loginBtn = document.getElementById('login-btn');
    if (loginForm && loginBtn) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const badgeId = document.getElementById('login-badge').value.trim();
            const password = document.getElementById('login-pwd').value;

            if (!badgeId || !password) {
                showAuthToast('Vui lòng nhập đầy đủ Mã cán bộ và Mật khẩu.');
                return;
            }

            const restoreButton = setButtonLoading(loginBtn, 'Đang xác thực...');

            try {
                const response = await fetch(`${API_BASE}/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        badge_id: badgeId,
                        password: password,
                    }),
                });

                if (!response.ok) {
                    const message = await parseApiError(response, 'Đăng nhập thất bại.');
                    showAuthToast(message);
                    return;
                }

                const data = await response.json();
                localStorage.setItem('tax_token', data.access_token);
                showAuthToast('Đăng nhập thành công, đang chuyển hướng...', 'success');
                setTimeout(() => {
                    window.location.href = 'dashboard.html';
                }, 380);
            } catch (error) {
                console.error(error);
                showApiOfflineMessage();
            } finally {
                restoreButton();
            }
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

            if (!fullName || !badgeId || !email || !dept || !pwd) {
                showAuthToast('Vui lòng điền đầy đủ các trường thông tin bắt buộc.');
                return;
            }
            if (pwd.length < 8) {
                showAuthToast('Mật khẩu phải có ít nhất 8 ký tự.');
                return;
            }
            if (pwd !== pwd2) {
                showAuthToast('Mật khẩu xác nhận không khớp.');
                return;
            }
            if (!email.endsWith('@gdt.gov.vn')) {
                showAuthToast('Email đăng ký bắt buộc phải là email công vụ (@gdt.gov.vn).');
                return;
            }

            const restoreButton = setButtonLoading(registerBtn, 'Đang khởi tạo...');

            try {
                const response = await fetch(`${API_BASE}/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        badge_id: badgeId,
                        full_name: fullName,
                        department: dept,
                        email: email,
                        password: pwd,
                        role: role,
                    }),
                });

                if (!response.ok) {
                    const message = await parseApiError(response, 'Có lỗi xảy ra khi tạo tài khoản.');
                    showAuthToast(message);
                    return;
                }

                showRegisterSuccessPanel({ badgeId, role });
                registerForm.reset();

                const loginBadge = document.getElementById('login-badge');
                if (loginBadge) {
                    loginBadge.value = badgeId;
                    loginBadge.dispatchEvent(new Event('input'));
                }

                setTimeout(() => {
                    if (typeof window.switchTab === 'function') {
                        window.switchTab('login');
                    }

                    const loginPwd = document.getElementById('login-pwd');
                    if (loginPwd) {
                        loginPwd.focus();
                    }

                    showAuthToast('Đăng ký thành công. Vui lòng đăng nhập để tiếp tục.', 'success');
                }, 1150);
            } catch (error) {
                console.error(error);
                showApiOfflineMessage();
            } finally {
                restoreButton();
            }
        });
    }
});
