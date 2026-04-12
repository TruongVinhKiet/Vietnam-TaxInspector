/**
 * api.js – Centralized API Configuration (Security Hardened)
 * ===========================================================
 * Changes:
 *   1. All fetch calls MUST use secureFetch() instead of raw fetch()
 *   2. secureFetch() automatically includes credentials (HttpOnly cookies)
 *   3. Handles 401 responses globally → redirects to login
 */

const API_BASE = "http://localhost:8000/api";
const SIDEBAR_IDENTITY_CACHE_KEY = "taxinspector_sidebar_identity_v1";

const SIDEBAR_ROLE_MAP = {
    viewer: "Chuyên viên",
    analyst: "Phân tích",
    inspector: "Thanh tra",
    admin: "Quản trị viên",
};

/**
 * Secure fetch wrapper that automatically includes HttpOnly cookie credentials
 * and handles session expiration globally.
 * 
 * @param {string} url - The URL to fetch
 * @param {object} options - Standard fetch options (method, headers, body, etc.)
 * @returns {Promise<Response>} - The fetch response
 */
async function secureFetch(url, options = {}) {
    const defaultOptions = {
        credentials: "include",   // CRITICAL: sends HttpOnly cookies automatically
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    };

    // Merge options (don't override content-type if body is FormData)
    const merged = { ...defaultOptions, ...options, headers: defaultOptions.headers };
    if (options.body instanceof FormData) {
        delete merged.headers["Content-Type"];
    }

    const response = await fetch(url, merged);

    // Global 401 handler: session expired → redirect to login
    if (response.status === 401) {
        const currentPage = window.location.pathname.split("/").pop();
        if (currentPage !== "login.html") {
            window.location.href = "login.html";
            return response;
        }
    }

    return response;
}


/**
 * Check if user is authenticated by calling /api/auth/me.
 * Redirects to login if not authenticated.
 * Returns user data if authenticated.
 */
async function checkAuth() {
    try {
        const res = await secureFetch(`${API_BASE}/auth/me`);
        if (!res.ok) {
            window.location.href = "login.html";
            return null;
        }
        return await res.json();
    } catch {
        window.location.href = "login.html";
        return null;
    }
}


/**
 * Logout: call /api/auth/logout to clear HttpOnly cookie on server side.
 */
async function logout() {
    try {
        await secureFetch(`${API_BASE}/auth/logout`, { method: "POST" });
    } catch {
        // Continue regardless
    }
    try {
        sessionStorage.removeItem(SIDEBAR_IDENTITY_CACHE_KEY);
    } catch {
        // Ignore storage errors
    }
    window.location.href = "login.html";
}


function getUserInitials(fullName) {
    if (!fullName) return "--";
    return fullName
        .split(" ")
        .filter(Boolean)
        .slice(0, 2)
        .map((word) => word[0])
        .join("")
        .toUpperCase();
}


function getRoleLabel(role) {
    return SIDEBAR_ROLE_MAP[role] || SIDEBAR_ROLE_MAP.viewer;
}


function applySidebarIdentity(user) {
    if (!user || typeof user !== "object") return;

    const sidebarName = document.getElementById("user-full-name");
    const sidebarRole = document.getElementById("user-current-role");
    const avatarImg = document.getElementById("user-avatar-image");
    const avatarFallback = document.getElementById("user-avatar-fallback");

    if (sidebarName && user.full_name) {
        sidebarName.textContent = user.full_name;
    }

    if (sidebarRole) {
        sidebarRole.textContent = getRoleLabel(user.role);
    }

    if (avatarImg && avatarFallback) {
        if (user.avatar_data) {
            avatarImg.src = user.avatar_data;
            avatarImg.classList.remove("hidden");
            avatarFallback.classList.add("hidden");
        } else {
            avatarImg.removeAttribute("src");
            avatarImg.classList.add("hidden");
            avatarFallback.classList.remove("hidden");
            avatarFallback.textContent = getUserInitials(user.full_name);
        }
    }
}


async function hydrateSidebarIdentity(options = {}) {
    const { forceRefresh = false } = options;
    if (!document.getElementById("user-profile-badge")) return null;

    if (!forceRefresh) {
        try {
            const raw = sessionStorage.getItem(SIDEBAR_IDENTITY_CACHE_KEY);
            if (raw) {
                const cached = JSON.parse(raw);
                applySidebarIdentity(cached);
            }
        } catch {
            // Ignore cache parse issues
        }
    }

    try {
        const res = await secureFetch(`${API_BASE}/auth/me`);
        if (!res.ok) return null;

        const user = await res.json();
        applySidebarIdentity(user);

        try {
            const cachedPayload = {
                full_name: user.full_name || "",
                role: user.role || "viewer",
                avatar_data: user.avatar_data || null,
            };
            sessionStorage.setItem(SIDEBAR_IDENTITY_CACHE_KEY, JSON.stringify(cachedPayload));
        } catch {
            // Ignore storage errors
        }

        return user;
    } catch {
        return null;
    }
}


document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("user-profile-badge")) {
        hydrateSidebarIdentity();
    }
});


function formatCurrencyCode(amount) {
    if (amount >= 1e9) {
        return (amount / 1e9).toFixed(1) + " Ty";
    }
    if (amount >= 1e6) {
        return (amount / 1e6).toFixed(1) + " Tr";
    }
    return amount.toLocaleString();
}
