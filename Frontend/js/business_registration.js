/**
 * business_registration.js
 * Client-side script for Nhóm 1 Nhận diện & Đăng ký Thuế.
 * Integrates Leaflet map click reverse-geocoding, cascading selects,
 * anti-fraud MST validation, 3-step registration wizard, and Bank reporting.
 */

let mapInstance = null;
let currentMarker = null;
let officesData = [];
let userSession = null;

// E-Signature Canvas variables
let canvas = null;
let ctx = null;
let drawing = false;

document.addEventListener("DOMContentLoaded", async function() {
    await fetchUserInfo();
    initLeafletMap();
    await loadDistricts();
    await loadAllTaxOffices();
    initSignatureCanvas();
});

/**
 * 1. Fetch logged in user identity & handle identity bars
 */
async function fetchUserInfo() {
    try {
        // Attempt to fetch current user session from local/session storage or /api/auth/me
        const stored = sessionStorage.getItem("user");
        if (stored) {
            userSession = JSON.parse(stored);
            updateUIWithUser(userSession);
        } else {
            // Fallback request
            const res = await secureFetch("/api/auth/me");
            if (res.ok) {
                userSession = await res.json();
                sessionStorage.setItem("user", JSON.stringify(userSession));
                updateUIWithUser(userSession);
            } else {
                console.warn("Could not retrieve active session, running in sandbox/guest mode.");
                // Default guest identity for sandbox
                userSession = {
                    full_name: "Cửa hàng Gia dụng Thuận Phát",
                    tax_code: "8092471928",
                    role: "taxpayer"
                };
                updateUIWithUser(userSession);
            }
        }
    } catch (err) {
        console.error("fetchUserInfo error:", err);
    }
}

function updateUIWithUser(user) {
    const fullNameEl = document.getElementById("header-user-name");
    const mstEl = document.getElementById("header-user-mst");
    const sidebarNameEl = document.getElementById("user-full-name");
    const sidebarRoleEl = document.getElementById("user-current-role");

    if (fullNameEl) fullNameEl.textContent = user.full_name || "Hộ kinh doanh Thuận Phát";
    if (mstEl) {
        mstEl.textContent = user.tax_code ? `MST: ${user.tax_code}` : "MST: Chưa liên kết";
    }
    if (sidebarNameEl) sidebarNameEl.textContent = user.full_name || "Người nộp thuế";
    if (sidebarRoleEl) {
        sidebarRoleEl.textContent = user.tax_code ? `MST: ${user.tax_code}` : "Chưa liên kết MST";
    }

    // Trigger bank accounts and compliance warning lists if MST is available
    if (user.tax_code) {
        fetchReportedBankAccounts(user.tax_code);
        fetchComplianceWarnings(user.tax_code);
    }
}

/**
 * 2. Leaflet Map Initialization & Markers
 */
function initLeafletMap() {
    try {
        // Center around HCMC (post-2025 districts)
        mapInstance = L.map('leaflet-map').setView([10.7769, 106.7009], 11);

        // Dark or Elegant tile styling from CartoDB or standard OSM
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(mapInstance);

        // Listen for maps clicks to perform reverse geocoding
        mapInstance.on('click', async function(e) {
            const { lat, lng } = e.latlng;
            await handleMapClickReverseGeocode(lat, lng);
        });

        console.log("[MAP] Leaflet engine initialized successfully.");
    } catch (err) {
        console.error("Leaflet initialization failed:", err);
    }
}

/**
 * 3. HCMC geographical cascading dropdowns
 */
async function loadDistricts() {
    try {
        const select = document.getElementById("lookup-district");
        if (!select) return;

        // Fetch districts from backend
        const res = await secureFetch("/api/registration/districts");
        if (res.ok) {
            const data = await res.json();
            // Clear existing except first
            select.innerHTML = '<option value="">-- Chọn Quận/Huyện --</option>';
            data.districts.forEach(d => {
                const opt = document.createElement("option");
                opt.value = d;
                opt.textContent = d;
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("loadDistricts failed:", err);
    }
}

async function onDistrictChange() {
    const districtSelect = document.getElementById("lookup-district");
    const wardSelect = document.getElementById("lookup-ward");
    if (!districtSelect || !wardSelect) return;

    const district = districtSelect.value;
    if (!district) {
        wardSelect.innerHTML = '<option value="">-- Chọn Phường/Xã --</option>';
        wardSelect.disabled = true;
        return;
    }

    try {
        const res = await secureFetch(`/api/registration/wards?district=${encodeURIComponent(district)}`);
        if (res.ok) {
            const data = await res.json();
            wardSelect.innerHTML = '<option value="">-- Chọn Phường/Xã --</option>';
            data.wards.forEach(w => {
                const opt = document.createElement("option");
                opt.value = w.ward_name;
                opt.textContent = `Phường/Xã ${w.ward_name}`;
                // Attach coordinates & managing tax office as attributes
                opt.setAttribute("data-office", w.tax_office_code);
                opt.setAttribute("data-lat", w.lat);
                opt.setAttribute("data-lng", w.lng);
                wardSelect.appendChild(opt);
            });
            wardSelect.disabled = false;
        }
    } catch (err) {
        console.error("onDistrictChange failed:", err);
    }
}

async function onWardChange() {
    const districtSelect = document.getElementById("lookup-district");
    const wardSelect = document.getElementById("lookup-ward");
    if (!districtSelect || !wardSelect) return;

    const district = districtSelect.value;
    const ward = wardSelect.value;
    if (!district || !ward) return;

    const opt = wardSelect.options[wardSelect.selectedIndex];
    const officeCode = opt.getAttribute("data-office");
    const lat = parseFloat(opt.getAttribute("data-lat"));
    const lng = parseFloat(opt.getAttribute("data-lng"));

    if (lat && lng) {
        // Center map around selected ward coordinates
        mapInstance.setView([lat, lng], 14);
        
        // Add temporary marker
        if (currentMarker) {
            mapInstance.removeLayer(currentMarker);
        }
        currentMarker = L.marker([lat, lng]).addTo(mapInstance)
            .bindPopup(`<b>Vị trí được chọn</b><br>Phường ${ward}, ${district}`)
            .openPopup();
    }

    // Call API to fetch correct office details
    try {
        const response = await secureFetch("/api/registration/lookup-tax-office", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ district, ward })
        });
        if (response.ok) {
            const payload = await response.json();
            renderTaxOfficeCard(payload.office);
        }
    } catch (err) {
        console.error("onWardChange lookup failed:", err);
    }
}

/**
 * 4. Map click reverse-geocoding lookup
 */
async function handleMapClickReverseGeocode(lat, lng) {
    try {
        // Add temporary marker to click location
        if (currentMarker) {
            mapInstance.removeLayer(currentMarker);
        }
        currentMarker = L.marker([lat, lng]).addTo(mapInstance)
            .bindPopup("Đang giải mã vị trí...")
            .openPopup();

        const res = await secureFetch("/api/registration/reverse-geocode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat, lng })
        });

        if (res.ok) {
            const data = await res.json();
            
            // Set values in selects and trigger update
            const districtSelect = document.getElementById("lookup-district");
            const wardSelect = document.getElementById("lookup-ward");

            if (districtSelect && wardSelect) {
                districtSelect.value = data.ward.district_name;
                await onDistrictChange(); // reload wards list
                wardSelect.value = data.ward.ward_name;
            }

            // Update marker popup
            currentMarker.bindPopup(`<b>${data.ward.district_name}</b><br>Phường/Xã ${data.ward.ward_name}`).openPopup();
            mapInstance.panTo([lat, lng]);

            // Render office card details
            renderTaxOfficeCard(data.office);
        } else {
            currentMarker.bindPopup("Không thuộc địa phận hành chính HCMC.").openPopup();
        }
    } catch (err) {
        console.error("handleMapClickReverseGeocode error:", err);
    }
}

/**
 * 5. Plot all 29 tax offices as pins on Leaflet
 */
async function loadAllTaxOffices() {
    try {
        const res = await secureFetch("/api/registration/tax-offices");
        if (res.ok) {
            const data = await res.json();
            officesData = data.offices;

            // Create custom icon
            const taxIcon = L.icon({
                iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
                shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            });

            // Plot markers
            officesData.forEach(office => {
                if (office.lat && office.lng) {
                    L.marker([office.lat, office.lng], { icon: taxIcon }).addTo(mapInstance)
                        .bindPopup(`<b>${office.office_name}</b><br>${office.address}<br><button onclick="selectOfficeByCode('${office.office_code}')" class="mt-2 px-2 py-1 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded text-[9px]">Xem chi tiết</button>`);
                }
            });
        }
    } catch (err) {
        console.error("loadAllTaxOffices failed:", err);
    }
}

function selectOfficeByCode(code) {
    const office = officesData.find(o => o.office_code === code);
    if (office) {
        renderTaxOfficeCard(office);
        mapInstance.setView([office.lat, office.lng], 13);
    }
}

function renderTaxOfficeCard(office) {
    const box = document.getElementById("office-result");
    const nameEl = document.getElementById("res-office-name");
    const badgeEl = document.getElementById("res-office-badge");
    const addrEl = document.getElementById("res-office-addr");
    const phoneEl = document.getElementById("res-office-phone");
    const hoursEl = document.getElementById("res-office-hours");

    if (!box) return;

    nameEl.textContent = office.full_name || office.office_name;
    badgeEl.textContent = office.office_code;
    addrEl.textContent = office.address || "Chưa cập nhật";
    phoneEl.textContent = office.phone || "Chưa cập nhật";
    hoursEl.textContent = office.working_hours || "Thứ 2 - Thứ 6 (07:30 - 11:30, 13:00 - 17:00)";

    box.classList.remove("hidden");
}

/**
 * 6. Household Group Classifier
 */
async function classifyHousehold() {
    const field = document.getElementById("class-field").value;
    const revenue = parseFloat(document.getElementById("class-revenue").value) || 0;
    const labor = parseInt(document.getElementById("class-labor").value) || 0;

    const resultBox = document.getElementById("class-result");
    const badgeEl = document.getElementById("res-group-badge");
    const nameEl = document.getElementById("res-group-name");
    const descEl = document.getElementById("res-group-desc");
    const obligationsList = document.getElementById("res-group-obligations");
    const taxSectorEl = document.getElementById("res-tax-sector");
    const taxRateEl = document.getElementById("res-tax-rate");
    const taxEstimateEl = document.getElementById("res-tax-estimate");

    try {
        const res = await secureFetch("/api/registration/classify-household", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ revenue, labor, field })
        });

        if (res.ok) {
            const data = await res.json();
            const classif = data.classification;
            const policy = data.tax_policy;

            // Render classifications
            badgeEl.textContent = `NHÓM ${classif.group}`;
            badgeEl.className = `px-2 py-0.5 font-bold rounded text-[9px] ${
                classif.group === 3 ? 'bg-rose-600 text-white' : classif.group === 2 ? 'bg-amber-500 text-white' : 'bg-emerald-600 text-white'
            }`;
            
            nameEl.textContent = classif.group_name;
            descEl.textContent = classif.description;

            // Obligations list
            obligationsList.innerHTML = "";
            classif.obligations.forEach(ob => {
                const li = document.createElement("li");
                li.textContent = ob;
                obligationsList.appendChild(li);
            });

            // Tax estimates
            taxSectorEl.textContent = policy.sector_name;
            taxRateEl.textContent = `${policy.total_rate}% (GTGT: ${policy.gtgt_rate}%, TNCN: ${policy.tncn_rate}%)`;
            
            if (policy.is_taxable) {
                taxEstimateEl.textContent = `${policy.estimated_annual_tax_million.toLocaleString("vi-VN")} Triệu VNĐ / Năm`;
            } else {
                taxEstimateEl.textContent = "Miễn nộp thuế (Doanh thu <= 500 Tr/năm)";
            }

            resultBox.classList.remove("hidden");
        }
    } catch (err) {
        console.error("classifyHousehold failed:", err);
    }
}

/**
 * 7. MST self-lookup & partner Shell company validation
 */
async function lookupMst() {
    const query = document.getElementById("lookup-mst-input").value.trim();
    const resultBox = document.getElementById("mst-result");
    const nameEl = document.getElementById("mst-res-name");
    const statusEl = document.getElementById("mst-res-status");
    const codeEl = document.getElementById("mst-res-code");
    const industryEl = document.getElementById("mst-res-industry");
    const addrEl = document.getElementById("mst-res-addr");
    const typeEl = document.getElementById("mst-res-type");
    const dateEl = document.getElementById("mst-res-date");
    const warningEl = document.getElementById("mst-warning-card");

    if (!query) {
        alert("Vui lòng nhập MST hoặc CCCD cần kiểm tra.");
        return;
    }

    try {
        const res = await secureFetch("/api/registration/lookup-mst", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        if (res.ok) {
            const data = await res.json();
            
            if (data.found) {
                nameEl.textContent = data.name;
                codeEl.textContent = data.mst || "CCCD";
                industryEl.textContent = data.industry || "Bán lẻ tổng hợp";
                addrEl.textContent = data.address || "Chưa cập nhật";
                typeEl.textContent = data.type === "business" ? "Doanh nghiệp / Hộ kinh doanh" : "Cá nhân đóng thuế";
                dateEl.textContent = data.registration_date || "Chưa cập nhật";
                
                // Status mapping & colors
                statusEl.textContent = data.status;
                if (data.status_code === "00") {
                    statusEl.className = "px-2 py-0.5 bg-emerald-100 text-emerald-800 font-bold rounded text-[9px]";
                    warningEl.classList.add("hidden");
                    resultBox.className = "p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-3 text-xs fade-in";
                } else if (data.status_code === "01") {
                    statusEl.className = "px-2 py-0.5 bg-amber-100 text-amber-800 font-bold rounded text-[9px]";
                    warningEl.classList.add("hidden");
                    resultBox.className = "p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-3 text-xs fade-in";
                } else {
                    // Critical status 06 (shell company / inactive at address warning)
                    statusEl.className = "px-2 py-0.5 bg-rose-600 text-white font-bold rounded text-[9px]";
                    warningEl.classList.remove("hidden");
                    resultBox.className = "p-4 rounded-xl border border-rose-300 bg-rose-50/50 space-y-3 text-xs fade-in";
                }

                resultBox.classList.remove("hidden");
            } else if (data.suggestions) {
                alert(`Không tìm thấy chính xác, nhưng có gợi ý: ${data.suggestions.map(s => s.name).join(", ")}`);
            } else {
                alert("Không tìm thấy thông tin khớp với yêu cầu.");
            }
        }
    } catch (err) {
        console.error("lookupMst failed:", err);
    }
}

/**
 * 8. 3-Step Wizard & CCCD Verification
 */
let wizardData = {};

function gotoStep(step) {
    // Progress UI steps
    const steps = [1, 2, 3];
    steps.forEach(s => {
        const card = document.getElementById(`wizard-step-${s}`);
        const label = document.getElementById(`step-label-${s}`);
        if (card) {
            if (s === step) {
                card.classList.remove("hidden");
                if (label) label.className = "text-emerald-500 font-bold";
            } else {
                card.classList.add("hidden");
                if (label) {
                    if (s < step) label.className = "text-emerald-500/60 font-semibold line-through";
                    else label.className = "text-slate-400 font-semibold";
                }
            }
        }
    });

    const progress = document.getElementById("step-progress-bar");
    if (progress) {
        progress.style.width = step === 1 ? "33%" : step === 2 ? "66%" : "100%";
    }
}

function triggerVneidVerification() {
    // Mock QR scan. Automatically populates standard citizen
    const randomCccds = ["079094001234", "001096004928", "079092004918"];
    const input = document.getElementById("wizard-cccd-input");
    if (input) {
        input.value = randomCccds[Math.floor(Math.random() * randomCccds.length)];
        verifyCccdHandshake();
    }
}

async function verifyCccdHandshake() {
    const cccd = document.getElementById("wizard-cccd-input").value.trim();
    if (!cccd || cccd.length !== 12) {
        alert("Vui lòng nhập mã CCCD đủ 12 chữ số hợp lệ.");
        return;
    }

    try {
        const res = await secureFetch("/api/registration/verify-cccd", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cccd })
        });

        if (res.ok) {
            const data = await res.json();
            wizardData = { ...data.citizen };
            
            // Populates Step 2 Form
            document.getElementById("wiz-full-name").value = data.citizen.full_name;
            document.getElementById("wiz-dob").value = data.citizen.date_of_birth;
            document.getElementById("wiz-gender").value = data.citizen.gender;
            document.getElementById("wiz-permanent-address").value = data.citizen.permanent_address;

            // Move to Step 2
            gotoStep(2);
        }
    } catch (err) {
        console.error("verifyCccdHandshake failed:", err);
    }
}

/**
 * 9. Canvas drawing e-signature
 */
function initSignatureCanvas() {
    canvas = document.getElementById("signature-canvas");
    if (!canvas) return;

    ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 3;

    // Mouse handlers
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    // Touch handlers for mobile
    canvas.addEventListener("touchstart", (e) => {
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        drawing = true;
        ctx.beginPath();
        ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
    });

    canvas.addEventListener("touchmove", (e) => {
        if (!drawing) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
        ctx.stroke();
    });

    canvas.addEventListener("touchend", stopDrawing);
}

function startDrawing(e) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
}

function stopDrawing() {
    drawing = false;
}

function clearSignatureCanvas() {
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

async function submitMstApplication() {
    // Grab all inputs
    const businessName = document.getElementById("wiz-business-name").value.trim();
    const businessAddress = document.getElementById("wiz-business-address").value.trim();
    const capital = parseFloat(document.getElementById("wiz-capital").value) || 0;
    const labor = parseInt(document.getElementById("wiz-labor").value) || 0;
    const districtSelect = document.getElementById("lookup-district");
    const wardSelect = document.getElementById("lookup-ward");

    if (!businessName || !businessAddress) {
        alert("Vui lòng khai báo tên cơ sở và địa điểm kinh doanh.");
        return;
    }

    // Convert canvas signature to base64
    const signatureData = canvas.toDataURL();

    // Prepare payload
    const payload = {
        ...wizardData,
        business_name: businessName,
        business_address: businessAddress,
        capital: capital,
        employee_count: labor,
        district: districtSelect.value || "Quận 1",
        ward: wardSelect.value || "Tân Định",
        expected_revenue: capital * 1.5, // simulated revenue
        signature_data: signatureData
    };

    try {
        const res = await secureFetch("/api/registration/submit-application", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const data = await res.json();
            
            // Render success screen
            document.getElementById("wizard-step-3").classList.add("hidden");
            document.getElementById("step-label-3").className = "text-emerald-500/60 font-semibold line-through";
            
            const successDiv = document.getElementById("wizard-success");
            document.getElementById("success-submitted-at").textContent = `Nộp lúc: ${data.application.submitted_at}`;
            document.getElementById("success-mst").textContent = data.application.mst_assigned;
            document.getElementById("success-office").textContent = `Chi cục Thuế Cơ sở ${data.application.tax_office_code}`;
            document.getElementById("success-group").textContent = `Nhóm ${data.application.household_group}`;

            successDiv.classList.remove("hidden");
            
            // Refresh main taxpayer identity with the newly assigned MST!
            userSession.tax_code = data.application.mst_assigned;
            sessionStorage.setItem("user", JSON.stringify(userSession));
            updateUIWithUser(userSession);
        }
    } catch (err) {
        console.error("submitMstApplication failed:", err);
    }
}

function resetWizard() {
    document.getElementById("wizard-success").classList.add("hidden");
    document.getElementById("wizard-cccd-input").value = "";
    document.getElementById("wiz-business-name").value = "";
    document.getElementById("wiz-business-address").value = "";
    document.getElementById("wiz-capital").value = "";
    document.getElementById("wiz-labor").value = "";
    clearSignatureCanvas();
    gotoStep(1);
}

/**
 * 10. Form 01/BK-STK Bank account reporting
 */
async function reportBankAccount() {
    if (!userSession || !userSession.tax_code) {
        alert("Vui lòng hoàn tất liên kết MST hoặc đăng ký tài khoản thuế trước khi thực hiện khai báo ngân hàng.");
        return;
    }

    const bankName = document.getElementById("bank-name").value;
    const accountNum = document.getElementById("bank-account-num").value.trim();
    const holder = document.getElementById("bank-account-holder").value.trim();

    if (!accountNum) {
        alert("Vui lòng nhập số tài khoản ngân hàng.");
        return;
    }

    try {
        const res = await secureFetch("/api/registration/report-bank-account", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                tax_code: userSession.tax_code,
                bank_name: bankName,
                account_number: accountNum,
                account_holder: holder
            })
        });

        if (res.ok) {
            const data = await res.json();
            alert(data.message);
            
            // Reset input & reload list
            document.getElementById("bank-account-num").value = "";
            document.getElementById("bank-account-holder").value = "";
            await fetchReportedBankAccounts(userSession.tax_code);
            await fetchComplianceWarnings(userSession.tax_code);
        }
    } catch (err) {
        console.error("reportBankAccount failed:", err);
    }
}

async function fetchReportedBankAccounts(mst) {
    const container = document.getElementById("reported-bank-accounts-list");
    if (!container) return;

    try {
        const res = await secureFetch(`/api/registration/bank-accounts/${mst}`);
        if (res.ok) {
            const data = await res.json();
            if (data.accounts.length === 0) {
                container.innerHTML = '<p class="text-[10px] text-slate-400 italic">Chưa khai báo tài khoản nào.</p>';
                return;
            }

            container.innerHTML = "";
            data.accounts.forEach(acc => {
                const div = document.createElement("div");
                div.className = "flex justify-between items-center p-2 bg-slate-50 border border-slate-200 rounded-lg text-[10px]";
                div.innerHTML = `
                    <div>
                        <p class="font-bold text-slate-800">${acc.bank_name}</p>
                        <p class="font-mono text-slate-500">${acc.account_number} • ${acc.account_holder}</p>
                    </div>
                    <span class="text-[9px] text-slate-400 font-semibold">${acc.reported_date}</span>
                `;
                container.appendChild(div);
            });
        }
    } catch (err) {
        console.error("fetchReportedBankAccounts failed:", err);
    }
}

/**
 * 11. Compliance warning alerts
 */
async function fetchComplianceWarnings(mst) {
    const banner = document.getElementById("compliance-banner-container");
    if (!banner) return;

    try {
        const res = await secureFetch(`/api/registration/compliance-warnings/${mst}`);
        if (res.ok) {
            const data = await res.json();
            if (data.warnings.length === 0) {
                banner.classList.add("hidden");
                return;
            }

            banner.innerHTML = "";
            data.warnings.forEach(w => {
                const card = document.createElement("div");
                card.className = "p-4 bg-amber-50 border-l-4 border-amber-500 text-slate-800 rounded-r-xl shadow-sm text-xs flex gap-3 fade-in mb-4";
                card.innerHTML = `
                    <span class="material-symbols-outlined text-amber-600 shrink-0">warning</span>
                    <div class="space-y-1">
                        <p class="font-extrabold text-amber-950">${w.title}</p>
                        <p class="text-[11px] leading-relaxed text-slate-600">${w.message}</p>
                        <p class="text-[10px] text-amber-800 font-bold">${w.deadline}</p>
                    </div>
                `;
                banner.appendChild(card);
            });
            banner.classList.remove("hidden");
        }
    } catch (err) {
        console.error("fetchComplianceWarnings failed:", err);
    }
}
