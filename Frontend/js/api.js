const API_BASE = "http://localhost:8000/api";

function formatCurrencyCode(amount) {
    if (amount >= 1e9) {
        return (amount / 1e9).toFixed(1) + " Ty";
    }
    if (amount >= 1e6) {
        return (amount / 1e6).toFixed(1) + " Tr";
    }
    return amount.toLocaleString();
}
