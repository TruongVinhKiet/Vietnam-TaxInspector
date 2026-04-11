async function checkFraudRisk() {
    const taxCode = document.getElementById('fraud-mst').value;
    if (!taxCode) return alert('Vui lòng nhập Mã số thuế');

    document.getElementById('fraud-btn').innerHTML = '<div class="loader" style="width: 20px; height: 20px; border-width: 2px;"></div> Đang phân tích...';
    
    try {
        const response = await fetch(`${API_BASE}/scoring/${taxCode}`);
        const data = await response.json();
        
        // Show result box
        document.getElementById('fraud-result').classList.remove('hidden');
        
        // In a full implementation, we would selectively update the DOM elements inside 'fraud-result'
        // based on the fields from `data` such as data.risk_score, data.risk_level, data.red_flags.
        // For visual parity with the mockup, we will display the mockup's static container.
        console.log("Fraud analysis complete:", data);

    } catch (error) {
        console.error(error);
        alert('Có lỗi xảy ra khi gọi API Hệ thống tính điểm.');
    } finally {
        document.getElementById('fraud-btn').innerHTML = '<span class="material-symbols-outlined text-[18px]">psychology</span> Phân tích AI';
    }
}
