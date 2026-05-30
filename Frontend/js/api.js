/**
 * api.js – Centralized API Configuration (Security Hardened + Role-Based Access Control)
 * =======================================================================================
 * Security Features:
 *   1. All fetch calls MUST use secureFetch() instead of raw fetch()
 *   2. secureFetch() automatically includes credentials (HttpOnly cookies)
 *   3. Handles 401 responses globally → redirects to login
 *   4. Role-based route guard: officers cannot access business_* pages, taxpayers cannot access officer pages
 *   5. 403 responses are silently absorbed to prevent UI crashes on cross-role API calls
 *   6. 'enterprise' role is treated identically to 'taxpayer' (merged role)
 */

const API_BASE = window.API_BASE_URL || "http://localhost:8000/api";

// ─── Role Constants ──────────────────────────────────────────────────────────
const OFFICER_ROLES  = ["viewer", "analyst", "inspector", "admin"];
const TAXPAYER_ROLES = ["taxpayer", "enterprise"];   // enterprise is legacy alias

function _isTaxpayerRole(role) { return TAXPAYER_ROLES.includes(role); }
function _isOfficerRole(role)  { return OFFICER_ROLES.includes(role); }

// ─── Public pages that skip auth checks ──────────────────────────────────────
const PUBLIC_PAGES = ["login.html", "reset-password.html", ""];


// ─── Global fetch interceptor ────────────────────────────────────────────────
// Automatically injects credentials for cross-origin API calls (port 3000 → 8000).
const originalFetch = window.fetch;
window.fetch = function (input, init = {}) {
    let url = typeof input === "string" ? input : (input instanceof Request ? input.url : "");
    const apiBaseUrl = window.API_BASE_URL || "http://localhost:8000/api";
    
    if (url.includes("/api/") && !url.includes("8000")) {
        const idx = url.indexOf("/api/");
        url = apiBaseUrl + url.substring(idx + 4);
        if (typeof input === "string") {
            input = url;
        } else if (input instanceof Request) {
            try {
                input = new Request(url, {
                    method: input.method,
                    headers: input.headers,
                    body: input.body,
                    mode: input.mode,
                    credentials: input.credentials || "include",
                    cache: input.cache,
                    redirect: input.redirect,
                    referrer: input.referrer,
                    integrity: input.integrity,
                    keepalive: input.keepalive,
                    signal: input.signal
                });
            } catch (e) {
                input = url;
            }
        }
    }
    
    if (url.startsWith(apiBaseUrl) || url.startsWith("http://localhost:8000")) {
        if (!init.credentials) {
            init.credentials = "include";
        }
        if (input instanceof Request && !input.credentials) {
            try { input = new Request(input, { credentials: "include" }); } catch (e) {}
        }
    }
    return originalFetch(input, init).then(response => {
        const currentPage = window.location.pathname.split("/").pop();
        if (PUBLIC_PAGES.includes(currentPage)) return response;

        // 401 → session expired, redirect to login
        if (response.status === 401) {
            try { sessionStorage.removeItem(SIDEBAR_IDENTITY_CACHE_KEY); } catch (e) {}
            window.location.href = "login.html";
        }
        return response;
    });
};


// ─── Session Cache ───────────────────────────────────────────────────────────
const SIDEBAR_IDENTITY_CACHE_KEY = "taxinspector_sidebar_identity_v1";

const SIDEBAR_ROLE_MAP = {
    viewer: "Chuyên viên",
    analyst: "Phân tích",
    inspector: "Thanh tra",
    admin: "Quản trị viên",
};


// ─── secureFetch ─────────────────────────────────────────────────────────────
/**
 * Secure fetch wrapper that automatically includes HttpOnly cookie credentials
 * and handles session expiration globally.
 */
async function secureFetch(url, options = {}) {
    const defaultOptions = {
        credentials: "include",
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    };

    const merged = { ...defaultOptions, ...options, headers: defaultOptions.headers };
    if (options.body instanceof FormData) {
        delete merged.headers["Content-Type"];
    }

    const response = await fetch(url, merged);

    // 401 → redirect to login (only on protected pages)
    if (response.status === 401) {
        const currentPage = window.location.pathname.split("/").pop();
        if (currentPage !== "login.html") {
            window.location.href = "login.html";
            return response;
        }
    }

    return response;
}


// ─── Auth Utilities ──────────────────────────────────────────────────────────

/**
 * Check if user is authenticated by calling /api/auth/me.
 * Redirects to login if not authenticated.
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


// ─── Sidebar Identity Helpers ────────────────────────────────────────────────

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


function applyTaxpayerIdentity(user) {
    if (!user || typeof user !== "object") return;
    
    // Find sidebar name (Nguyễn Văn Thuận is default)
    const sidebarName = Array.from(document.querySelectorAll("aside p.text-slate-200")).find(
        p => p.textContent.includes("Nguyễn Văn Thuận") || p.classList.contains("user-full-name-placeholder")
    ) || document.querySelector("aside p.text-slate-200");
    if (sidebarName) {
        sidebarName.textContent = user.full_name;
        sidebarName.classList.add("user-full-name-placeholder");
    }

    // Find sidebar MST
    const sidebarMst = Array.from(document.querySelectorAll("aside p.font-mono")).find(
        p => p.textContent.includes("MST:") || p.classList.contains("user-mst-placeholder")
    ) || document.querySelector("aside p.font-mono");
    if (sidebarMst) {
        const prefix = (user.badge_id && user.badge_id.length === 12) ? "CCCD: " : "MST: ";
        sidebarMst.textContent = prefix + user.badge_id;
        sidebarMst.classList.add("user-mst-placeholder");
    }

    // Find header business name
    const headerName = Array.from(document.querySelectorAll("header p.text-slate-800")).find(
        p => p.textContent.includes("Cửa hàng Gia dụng Thuận Phát") || p.classList.contains("header-name-placeholder")
    ) || document.querySelector("header p.text-slate-800");
    if (headerName) {
        headerName.textContent = user.full_name;
        headerName.classList.add("header-name-placeholder");
    }

    // Find header MST
    const headerMst = Array.from(document.querySelectorAll("header p.text-\\[9px\\]")).find(
        p => p.textContent.includes("Mã số thuế:") || p.textContent.includes("MST:") || p.classList.contains("header-mst-placeholder")
    ) || document.querySelector("header p.text-\\[9px\\]");
    if (headerMst) {
        const prefix = (user.badge_id && user.badge_id.length === 12) ? "Số CCCD: " : "Mã số thuế: ";
        headerMst.textContent = prefix + user.badge_id;
        headerMst.classList.add("header-mst-placeholder");
    }

    // User initials bubbles
    const userInitialsBubbles = document.querySelectorAll("header .w-8.h-8.rounded-full.bg-slate-100");
    userInitialsBubbles.forEach(bubble => {
        bubble.textContent = getUserInitials(user.full_name);
    });

    const sidebarRoleEl = document.querySelector("aside div.mt-2\\.5 span.text-\\[9px\\]");
    if (sidebarRoleEl) {
        sidebarRoleEl.textContent = "Hộ Kinh Doanh / Cá Nhân / DN";
    }

    const avatarImg = document.getElementById("user-avatar-image");
    const avatarFallback = document.getElementById("user-avatar-fallback");
    if (avatarImg && avatarFallback) {
        if (user.avatar_data) {
            avatarImg.src = user.avatar_data;
            avatarImg.classList.remove("hidden");
            avatarFallback.classList.add("hidden");
        } else {
            avatarImg.removeAttribute("src");
            avatarImg.classList.add("hidden");
            avatarFallback.classList.remove("hidden");
            avatarFallback.textContent = "NNT";
        }
    } else {
        const sidebarInitialsBubble = document.querySelector("aside .rounded-full.bg-emerald-600");
        if (sidebarInitialsBubble) {
            sidebarInitialsBubble.textContent = "NNT";
        }
    }
}


function applySidebarIdentity(user) {
    if (!user || typeof user !== "object") return;

    if (_isTaxpayerRole(user.role)) {
        try { applyTaxpayerIdentity(user); } catch (e) {
            console.warn("[api.js] applyTaxpayerIdentity error (non-fatal):", e);
        }
        return;
    }

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
        try { applySidebarIdentity(user); } catch (e) {
            console.warn("[api.js] applySidebarIdentity error (non-fatal):", e);
        }

        try {
            const cachedPayload = {
                full_name: user.full_name || "",
                role: user.role || "viewer",
                badge_id: user.badge_id || "",
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


// ─── ROLE-BASED ROUTE GUARD (Government-Grade) ──────────────────────────────
// This runs on every protected page load. It enforces two invariants:
//   1. Taxpayer users can ONLY access business_*.html pages
//   2. Officer users can ONLY access non-business_*.html pages
//   3. Unauthenticated users are sent to login.html
//
// Implementation uses a two-phase check:
//   Phase 1 (synchronous): Check sessionStorage cache for instant redirect
//   Phase 2 (async):       Verify with /api/auth/me and re-check

function _enforceRouteAccess(role, currentPage) {
    const isTaxpayerPage = currentPage.startsWith("business_");
    const isTaxpayer = _isTaxpayerRole(role);

    if (isTaxpayerPage && !isTaxpayer) {
        // Officer tried to access taxpayer page → send to officer dashboard
        window.location.href = "dashboard.html";
        return true;
    }
    if (!isTaxpayerPage && isTaxpayer) {
        // Taxpayer tried to access officer page → send to taxpayer dashboard
        window.location.href = "business_dashboard.html";
        return true;
    }
    return false; // access allowed
}

// Run Phase 1 immediately to abort unauthorized pages as fast as possible (even before DOMContentLoaded)
(function() {
    const currentPage = window.location.pathname.split("/").pop();
    if (PUBLIC_PAGES.includes(currentPage)) return;

    try {
        const raw = sessionStorage.getItem(SIDEBAR_IDENTITY_CACHE_KEY);
        if (raw) {
            const cached = JSON.parse(raw);
            if (cached && cached.role) {
                if (_enforceRouteAccess(cached.role, currentPage)) return;
            }
        }
    } catch (e) {}
})();

document.addEventListener("DOMContentLoaded", () => {
    const currentPage = window.location.pathname.split("/").pop();
    if (PUBLIC_PAGES.includes(currentPage)) return;

    // Phase 2: Async server verification
    hydrateSidebarIdentity().then(user => {
        if (user) {
            _enforceRouteAccess(user.role, currentPage);
        } else {
            window.location.href = "login.html";
        }
    });
});


// ─── Utility ─────────────────────────────────────────────────────────────────
function formatCurrencyCode(amount) {
    if (amount >= 1e9) {
        return (amount / 1e9).toFixed(1) + " Ty";
    }
    if (amount >= 1e6) {
        return (amount / 1e6).toFixed(1) + " Tr";
    }
    return amount.toLocaleString();
}
