(function () {
    const UI = window.TaxpayerUI;
    const Registry = window.TaxpayerAIRegistry;
    if (!UI || !Registry) return;

    function normalizeError(error) {
        return {
            status: "error",
            message: error?.message || "Không thể tải insight AI.",
            confidence: "low",
            model: { confidence: "low" },
        };
    }

    async function requestCapability(key, payload) {
        const capability = Registry.getCapability(key);
        if (!capability) {
            throw new Error(`Capability không tồn tại: ${key}`);
        }
        const body = payload ?? capability.body ?? {};
        if ((capability.method || "GET").toUpperCase() === "POST") {
            return UI.post(capability.endpoint, body);
        }
        return UI.get(capability.endpoint);
    }

    async function safeRequestCapability(key, payload) {
        try {
            return await requestCapability(key, payload);
        } catch (error) {
            return normalizeError(error);
        }
    }

    async function requestMany(keys) {
        const pairs = await Promise.all(keys.map(async (key) => [key, await safeRequestCapability(key)]));
        return Object.fromEntries(pairs);
    }

    window.TaxpayerAIClient = {
        requestCapability,
        safeRequestCapability,
        requestMany,
        normalizeError,
    };
})();
