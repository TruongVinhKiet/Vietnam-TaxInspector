/**
 * api.js – Centralized API Configuration (Security Hardened)
 * ===========================================================
 * Changes:
 *   1. All fetch calls MUST use secureFetch() instead of raw fetch()
 *   2. secureFetch() automatically includes credentials (HttpOnly cookies)
 *   3. Handles 401 responses globally → redirects to login
 */

const API_BASE = "http://localhost:8000/api";

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
    window.location.href = "login.html";
}


function formatCurrencyCode(amount) {
    if (amount >= 1e9) {
        return (amount / 1e9).toFixed(1) + " Ty";
    }
    if (amount >= 1e6) {
        return (amount / 1e6).toFixed(1) + " Tr";
    }
    return amount.toLocaleString();
}
