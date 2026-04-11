async function checkFraudRisk() {
    const taxCode = document.getElementById('fraud-mst').value;
    if (!taxCode) return alert('Vui lòng nhập Mã số thuế');

    document.getElementById('fraud-btn').innerHTML = '<div class="loader" style="width: 20px; height: 20px; border-width: 2px;"></div> Đang phân tích...';
    
    try {
        const response = await fetch(`${API_BASE}/scoring/${taxCode}`);
        const data = await response.json();
        
        // --- Animation Logic ---
        const resultDiv = document.getElementById('fraud-result');
        resultDiv.classList.remove('hidden');
        
        // 1. Fade slide in
        resultDiv.animate([
            { opacity: 0, transform: 'translateY(40px)' },
            { opacity: 1, transform: 'translateY(0)' }
        ], {
            duration: 800,
            easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
            fill: 'both'
        });

        // 2. Animate Numbers
        const scoreEl = document.getElementById('risk-score');
        const circleEl = document.getElementById('risk-circle');
        const confEl = document.getElementById('anim-confidence');
        
        const targetScore = 85; // Hardcoded static mock per UI
        const targetConf = 94.2; 
        
        // reset to 0
        scoreEl.textContent = '0';
        confEl.textContent = '0';
        // max circumference is 552.92
        circleEl.style.strokeDashoffset = '552.92';
        
        // trigger reflow
        void circleEl.offsetWidth;

        // Start animations
        const durationMs = 2000;
        const startTime = performance.now();
        
        // Stroke offset animation
        const targetOffset = 552.92 - (552.92 * (targetScore / 100));
        circleEl.style.strokeDashoffset = targetOffset;
        
        // Count up numbers
        requestAnimationFrame(function animateNumbers(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / durationMs, 1);
            
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            
            scoreEl.textContent = Math.round(easeOut * targetScore);
            confEl.textContent = (easeOut * targetConf).toFixed(1);
            
            if (progress < 1) {
                requestAnimationFrame(animateNumbers);
            } else {
                scoreEl.textContent = targetScore;
                confEl.textContent = targetConf;
            }
        });

        console.log("Fraud analysis complete:", data);

    } catch (error) {
        console.error(error);
        alert('Có lỗi xảy ra khi gọi API Hệ thống tính điểm.');
    } finally {
        document.getElementById('fraud-btn').innerHTML = '<span class="material-symbols-outlined text-[18px]">psychology</span> Phân tích AI';
    }
}
