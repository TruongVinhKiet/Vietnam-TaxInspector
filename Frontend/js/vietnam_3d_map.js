// vietnam_3d_map.js - Three.js extruded 34-unit Vietnam macro risk map.

const Vietnam3DMap = (() => {
    let scene, camera, renderer, raycaster, pointer, controls;
    let meshes = [];
    let containerId = 'vietnam-3d-map';
    let geoBounds = null;
    let animationStarted = false;
    let infoOverlay = null;

    async function init(id = 'vietnam-3d-map') {
        containerId = id;
        const container = document.getElementById(containerId);
        if (!container || typeof THREE === 'undefined') return;
        if (!window.MacroMapData) {
            container.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-400">Không tải được dữ liệu bản đồ.</div>';
            return;
        }

        const mapState = await window.MacroMapData.loadState();
        geoBounds = computeGeoBounds(mapState.geojson);

        container.innerHTML = '';
        container.style.position = 'relative';
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x08111f);
        camera = new THREE.PerspectiveCamera(42, container.clientWidth / Math.max(container.clientHeight, 1), 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.06;
            controls.enablePan = true;
            controls.enableZoom = true;
            controls.maxPolarAngle = Math.PI / 2.04;
            controls.minDistance = 8;
            controls.maxDistance = 42;
        }

        raycaster = new THREE.Raycaster();
        pointer = new THREE.Vector2();

        scene.add(new THREE.AmbientLight(0xffffff, 0.62));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.82);
        keyLight.position.set(8, -18, 28);
        scene.add(keyLight);
        const rimLight = new THREE.DirectionalLight(0x8fb8ff, 0.35);
        rimLight.position.set(-12, 18, 18);
        scene.add(rimLight);

        // Create info overlay element
        createInfoOverlay(container);

        buildProvinceMeshes(mapState.geojson);
        frameCountry();
        renderer.domElement.addEventListener('click', onClick);
        window.addEventListener('resize', resize);
        if (!animationStarted) {
            animationStarted = true;
            animate();
        }
    }

    function createInfoOverlay(container) {
        if (infoOverlay) infoOverlay.remove();
        infoOverlay = document.createElement('div');
        infoOverlay.id = 'three-map-info-overlay';
        infoOverlay.style.cssText = `
            position:absolute; top:12px; right:12px; z-index:10;
            max-width:320px; min-width:260px;
            background:rgba(0,33,71,0.94); backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.15); border-radius:12px;
            padding:16px; color:#fff; font-family:Inter,system-ui,sans-serif;
            display:none; box-shadow:0 8px 32px rgba(0,0,0,0.4);
            transition:opacity 0.2s ease; opacity:0;
        `;
        container.appendChild(infoOverlay);
    }

    function showInfoCard(province) {
        if (!infoOverlay || !province) return;
        const esc = window.MacroMapData?.escapeHtml || ((s) => String(s ?? ''));
        const riskScore = Number(province.risk_score || 0);
        const riskColor = riskScore >= 65 ? '#ef4444' : riskScore >= 35 ? '#fbbf24' : '#34d399';
        const riskLabel = riskScore >= 65 ? 'Cao' : riskScore >= 35 ? 'TB' : 'Thấp';
        const gdp = Number(province.gdp_billion_vnd || 0);
        const tax = Number(province.tax_revenue_billion_vnd || 0);
        const enterprises = Number(province.num_enterprises || 0);
        const compliance = Number(province.compliance_rate || 0);
        const population = Number(province.population || 0);
        const taxEfficiency = gdp > 0 ? ((tax / gdp) * 100).toFixed(1) : '—';
        const gdpPerCapita = population > 0 ? (gdp * 1e9 / population / 1e6).toFixed(1) : '—';

        infoOverlay.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                <span style="font-size:15px;font-weight:800;letter-spacing:-0.02em">${esc(province.province_name || 'Tỉnh/TP')}</span>
                <span style="margin-left:auto;background:${riskColor};color:#000;font-size:10px;font-weight:800;padding:2px 10px;border-radius:20px">${riskLabel}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;font-weight:700;letter-spacing:0.05em">Điểm rủi ro</div>
                    <div style="font-size:20px;font-weight:900;color:${riskColor};margin-top:2px">${riskScore.toFixed(1)}</div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;font-weight:700;letter-spacing:0.05em">Doanh nghiệp</div>
                    <div style="font-size:20px;font-weight:900;margin-top:2px">${enterprises.toLocaleString('vi-VN')}</div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;font-weight:700;letter-spacing:0.05em">GDP/GRDP</div>
                    <div style="font-size:16px;font-weight:800;margin-top:2px">${gdp.toLocaleString('vi-VN')} <span style="font-size:10px;color:#94a3b8">tỷ</span></div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;font-weight:700;letter-spacing:0.05em">Thu thuế</div>
                    <div style="font-size:16px;font-weight:800;margin-top:2px">${tax.toLocaleString('vi-VN')} <span style="font-size:10px;color:#94a3b8">tỷ</span></div>
                </div>
            </div>
            <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:11px">
                <div style="text-align:center">
                    <div style="color:#94a3b8;font-size:9px;font-weight:600">Tuân thủ</div>
                    <div style="font-weight:800">${(compliance * 100).toFixed(1)}%</div>
                </div>
                <div style="text-align:center">
                    <div style="color:#94a3b8;font-size:9px;font-weight:600">Hiệu suất thuế</div>
                    <div style="font-weight:800">${taxEfficiency}%</div>
                </div>
                <div style="text-align:center">
                    <div style="color:#94a3b8;font-size:9px;font-weight:600">${population > 0 ? 'GDP/người' : 'Mã vùng'}</div>
                    <div style="font-weight:800">${gdpPerCapita !== '—' ? gdpPerCapita + ' tr' : esc(province.province_code)}</div>
                </div>
            </div>
            <div style="margin-top:10px;text-align:center;font-size:10px;color:rgba(255,255,255,0.4)">Kịch bản chi tiết đã tải ở panel bên phải →</div>
        `;
        infoOverlay.style.display = 'block';
        requestAnimationFrame(() => { infoOverlay.style.opacity = '1'; });
    }

    function hideInfoCard() {
        if (!infoOverlay) return;
        infoOverlay.style.opacity = '0';
        setTimeout(() => { if (infoOverlay) infoOverlay.style.display = 'none'; }, 200);
    }

    function buildProvinceMeshes(geojson) {
        meshes.forEach((mesh) => scene.remove(mesh));
        meshes = [];
        if (!geojson?.features?.length) return;

        geojson.features.forEach((feature) => {
            const province = window.MacroMapData.provinceForFeature(feature);
            const score = Number(province.risk_score || 0) / 100;
            const height = 0.08 + score * 1.65;
            const material = new THREE.MeshStandardMaterial({
                color: riskColor(score),
                roughness: 0.58,
                metalness: 0.08,
                side: THREE.DoubleSide,
            });
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0x06101f, transparent: true, opacity: 0.55 });
            const coordsList = feature.geometry?.type === 'Polygon'
                ? [feature.geometry.coordinates]
                : (feature.geometry?.type === 'MultiPolygon' ? feature.geometry.coordinates : []);

            coordsList.forEach((polygon) => {
                const shape = polygonToShape(polygon);
                if (!shape) return;
                const geometry = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false });
                const mesh = new THREE.Mesh(geometry, material);
                const edges = new THREE.EdgesGeometry(geometry);
                mesh.add(new THREE.LineSegments(edges, lineMaterial));
                mesh.userData = province;
                scene.add(mesh);
                meshes.push(mesh);
            });
        });
    }

    function polygonToShape(polygon) {
        if (!polygon?.[0]?.length) return null;
        const shape = new THREE.Shape();
        polygon[0].forEach((coord, idx) => {
            const pt = projectVietnam(coord[0], coord[1]);
            if (idx === 0) shape.moveTo(pt.x, pt.y);
            else shape.lineTo(pt.x, pt.y);
        });
        for (let i = 1; i < polygon.length; i += 1) {
            const hole = new THREE.Path();
            polygon[i].forEach((coord, idx) => {
                const pt = projectVietnam(coord[0], coord[1]);
                if (idx === 0) hole.moveTo(pt.x, pt.y);
                else hole.lineTo(pt.x, pt.y);
            });
            shape.holes.push(hole);
        }
        return shape;
    }

    function computeGeoBounds(geojson) {
        const xs = [];
        const ys = [];
        const walk = (coords) => {
            if (Array.isArray(coords) && coords.length && typeof coords[0] === 'number') {
                xs.push(coords[0]);
                ys.push(coords[1]);
            } else if (Array.isArray(coords)) {
                coords.forEach(walk);
            }
        };
        (geojson?.features || []).forEach((feature) => walk(feature.geometry?.coordinates));
        const minLng = Math.min(...xs);
        const maxLng = Math.max(...xs);
        const minLat = Math.min(...ys);
        const maxLat = Math.max(...ys);
        return {
            minLng, maxLng, minLat, maxLat,
            centerLng: (minLng + maxLng) / 2,
            centerLat: (minLat + maxLat) / 2,
            span: Math.max(maxLng - minLng, maxLat - minLat) || 1,
        };
    }

    function projectVietnam(lng, lat) {
        const bounds = geoBounds || { centerLng: 106.5, centerLat: 16.2, span: 15 };
        const scale = 16 / bounds.span;
        return {
            x: (lng - bounds.centerLng) * scale,
            y: (lat - bounds.centerLat) * scale,
        };
    }

    function frameCountry() {
        camera.position.set(0, -20, 21);
        camera.lookAt(0, 0, 0);
        if (controls) {
            controls.target.set(0, 0, 0);
            controls.update();
        }
        resize();
    }

    function riskColor(score) {
        if (score >= 0.65) return 0xef4444;
        if (score >= 0.35) return 0xfacc15;
        return 0x22c55e;
    }

    function applyMacroParams(params = {}) {
        if (!scene || !window.MacroMapData) return;
        window.MacroMapData.applyMacroParams(params);
        window.MacroMapData.loadState().then((mapState) => buildProvinceMeshes(mapState.geojson));
    }

    function applyProvinceImpacts(impacts = []) {
        if (!scene || !window.MacroMapData) return;
        window.MacroMapData.applyProvinceImpacts(impacts);
        window.MacroMapData.loadState().then((mapState) => buildProvinceMeshes(mapState.geojson));
    }

    function onClick(event) {
        if (!renderer || !camera || !raycaster) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const intersections = raycaster.intersectObjects(meshes, true);
        
        if (intersections.length === 0) {
            hideInfoCard();
            return;
        }
        
        const hit = intersections[0];
        let province = hit?.object?.userData;
        // Walk up to parent if userData is on parent mesh (LineSegments children)
        if ((!province || !province.province_code) && hit?.object?.parent) {
            province = hit.object.parent.userData;
        }
        
        if (province?.province_code) {
            showInfoCard(province);
            if (window.loadProvinceScenario) {
                window.loadProvinceScenario(province.province_code, province.province_name);
            }
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        if (!renderer || !scene || !camera) return;
        if (controls) controls.update();
        renderer.render(scene, camera);
    }

    function resize() {
        const container = document.getElementById(containerId);
        if (!container || !renderer || !camera) return;
        const width = Math.max(container.clientWidth, 1);
        const height = Math.max(container.clientHeight, 1);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function onVisible() {
        resize();
        frameCountry();
    }

    return { init, applyMacroParams, applyProvinceImpacts, onVisible };
})();

window.Vietnam3DMap = Vietnam3DMap;

document.addEventListener('DOMContentLoaded', () => {
    const leaf = document.getElementById('vietnam-map');
    const three = document.getElementById('vietnam-3d-map');
    const echartsContainer = document.getElementById('vietnam-echarts-map');

    const leafBtn = document.getElementById('map-tab-leaflet');
    const threeBtn = document.getElementById('map-tab-3d');
    const echartsBtn = document.getElementById('map-tab-echarts');

    if (!leaf || !three || !leafBtn || !threeBtn) return;

    let threeInitialized = false;
    let echartsInitialized = false;

    function resetTabs() {
        if (leaf) leaf.classList.add('hidden');
        if (three) three.classList.add('hidden');
        if (echartsContainer) echartsContainer.classList.add('hidden');
        [leafBtn, threeBtn, echartsBtn].forEach((btn) => {
            if (!btn) return;
            btn.classList.remove('bg-white', 'text-primary-container', 'border', 'border-slate-200', 'shadow-sm');
        });
    }

    function activate(btn) {
        btn.classList.add('bg-white', 'text-primary-container', 'border', 'border-slate-200', 'shadow-sm');
    }

    leafBtn.addEventListener('click', () => {
        resetTabs();
        leaf.classList.remove('hidden');
        activate(leafBtn);
        setTimeout(() => window.VietnamMap?.onVisible?.(), 80);
        setTimeout(() => window.VietnamMap?.onVisible?.(), 280);
    });

    threeBtn.addEventListener('click', async () => {
        resetTabs();
        three.classList.remove('hidden');
        activate(threeBtn);
        if (!threeInitialized) {
            threeInitialized = true;
            await Vietnam3DMap.init('vietnam-3d-map');
        }
        setTimeout(() => Vietnam3DMap.onVisible(), 80);
        setTimeout(() => Vietnam3DMap.onVisible(), 280);
    });

    if (echartsBtn && echartsContainer) {
        echartsBtn.addEventListener('click', async () => {
            resetTabs();
            echartsContainer.classList.remove('hidden');
            activate(echartsBtn);
            if (!echartsInitialized && window.VietnamEChartsMap) {
                echartsInitialized = true;
                await window.VietnamEChartsMap.init('vietnam-echarts-map');
            }
            setTimeout(() => window.VietnamEChartsMap?.onVisible?.(), 80);
            setTimeout(() => window.VietnamEChartsMap?.onVisible?.(), 280);
        });
    }
});
