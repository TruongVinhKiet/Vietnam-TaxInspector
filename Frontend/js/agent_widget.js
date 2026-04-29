// e:\TaxInspector\Frontend\js\agent_widget.js
document.addEventListener("DOMContentLoaded", () => {
    // 1. Inject Floating Widget CSS
    const style = document.createElement('style');
    style.innerHTML = `
        .tax-agent-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #1E293B;
            box-shadow: 0 10px 25px rgba(30, 41, 59, 0.4), 0 0 0 4px rgba(30, 41, 59, 0.1);
            cursor: pointer;
            z-index: 9999;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .tax-agent-widget:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 35px rgba(30, 41, 59, 0.5), 0 0 0 6px rgba(14, 165, 233, 0.3);
        }
        .tax-agent-widget img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .tax-agent-widget::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border-radius: 50%;
            box-shadow: inset 0 0 15px rgba(0,0,0,0.5);
            pointer-events: none;
        }
        .tax-agent-pulse {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(14, 165, 233, 0.5);
            z-index: 9998;
            animation: agent-pulse 2s infinite cubic-bezier(0.4, 0, 0.2, 1);
        }
        @keyframes agent-pulse {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(1.8); opacity: 0; }
        }
        
        .tax-agent-tooltip {
            position: fixed;
            bottom: 45px;
            right: 105px;
            background: #0F172A;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s ease;
            transform: translateX(10px);
            z-index: 9999;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            white-space: nowrap;
        }
        .tax-agent-tooltip::after {
            content: '';
            position: absolute;
            top: 50%;
            right: -6px;
            transform: translateY(-50%);
            border-width: 6px 0 6px 6px;
            border-style: solid;
            border-color: transparent transparent transparent #0F172A;
        }
        .tax-agent-widget:hover + .tax-agent-tooltip {
            opacity: 1;
            transform: translateX(0);
        }
        
        /* Transition Overlay for page change */
        .tax-agent-transition-overlay {
            position: fixed;
            bottom: 60px;
            right: 60px;
            width: 0;
            height: 0;
            background: #F8FAFC;
            border-radius: 50%;
            z-index: 10000;
            transition: all 0.6s cubic-bezier(0.85, 0, 0.15, 1);
            pointer-events: none;
        }
        .tax-agent-transition-overlay.active {
            width: 300vw;
            height: 300vw;
            bottom: -150vw;
            right: -150vw;
        }
    `;
    document.head.appendChild(style);

    // 2. Inject HTML
    const pulse = document.createElement('div');
    pulse.className = 'tax-agent-pulse';
    
    const defaultAvatar = 'ai_avatar.png';
    const savedTheme = localStorage.getItem('taxAgentTheme') || defaultAvatar;

    const widget = document.createElement('div');
    widget.className = 'tax-agent-widget';
    widget.innerHTML = `<img src="../assets/img/${savedTheme}" alt="AI Agent">`;

    
    const tooltip = document.createElement('div');
    tooltip.className = 'tax-agent-tooltip';
    tooltip.textContent = 'Trợ lý AI Đa Năng';
    
    const overlay = document.createElement('div');
    overlay.className = 'tax-agent-transition-overlay';

    document.body.appendChild(pulse);
    document.body.appendChild(widget);
    document.body.appendChild(tooltip);
    document.body.appendChild(overlay);

    // 3. Navigation Logic
    widget.addEventListener('click', () => {
        overlay.classList.add('active');
        widget.style.transform = 'scale(0)';
        pulse.style.display = 'none';
        tooltip.style.display = 'none';
        
        setTimeout(() => {
            window.location.href = 'agent.html';
        }, 550); // Matches the CSS transition duration
    });

    // Handle browser back button (bfcache restoration)
    window.addEventListener('pageshow', (event) => {
        // Always reset the transition overlay just in case
        overlay.classList.remove('active');
        widget.style.transform = '';
        pulse.style.display = '';
    });
});
