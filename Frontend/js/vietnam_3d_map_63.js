// vietnam_3d_map.js - Lightweight Three.js macro risk map.
const Vietnam3DMap = (() => {

    let scene, camera, renderer, raycaster, pointer, controls;
    let provinces = [];
    let geojsonData = null;
    let meshes = [];
    let containerId = 'vietnam-3d-map';
    let macroParams = { gdp_delta_pct: 0, compliance_delta: 0, unemployment_delta: 0 };

    async function init(id = 'vietnam-3d-map') {
        containerId = id;
        const container = document.getElementById(containerId);
        if (!container || typeof THREE === 'undefined') return;

        const boundaryVersion = window.MACRO_BOUNDARY_VERSION || 'vn_34_2025';
        const [data, geojson] = await Promise.all([
            fetchJson(`${getApiBase()}/simulation/provinces?boundary_version=${encodeURIComponent(boundaryVersion)}`),
            fetchJson('../json/vietnam.json')
        ]).catch(e => { console.error('Fetch error:', e); return [{}, null]; });

        provinces = Array.isArray(data.provinces) ? data.provinces : [];
        geojsonData = geojson;

        container.innerHTML = '';
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x08111f);
        camera = new THREE.PerspectiveCamera(45, container.clientWidth / Math.max(container.clientHeight, 1), 0.1, 1000);
        camera.position.set(0, -25, 25);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2.1;
        }

        raycaster = new THREE.Raycaster();
        pointer = new THREE.Vector2();

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.8);
        light.position.set(10, -20, 30);
        scene.add(light);

        const light2 = new THREE.DirectionalLight(0xaaccff, 0.3);
        light2.position.set(-10, 20, 20);
        scene.add(light2);

        buildProvinceMeshes();
        renderer.domElement.addEventListener('click', onClick);
        window.addEventListener('resize', resize);
        animate();
    }



    function buildProvinceMeshes() {
        meshes.forEach((mesh) => scene.remove(mesh));
        meshes = [];

        if (!geojsonData || !geojsonData.features) return;

        // Map province data by name for quick lookup
        const provMap = new Map();
        provinces.forEach(p => provMap.set(p.province_name, p));

        const PROVINCE_NAME_MAP = {
            'TP.HCM': 'Hồ Chí Minh city',
            'Hồ Chí Minh': 'Hồ Chí Minh city',
            'Bà Rịa-VT': 'Bà Rịa - Vũng Tàu',
            'Bà Rịa - Vũng Tàu': 'Bà Rịa - Vũng Tàu',
            'Thừa Thiên Huế': 'Thừa Thiên - Huế',
            'Đắk Lắk': 'Đắk Lắk',
            'Đăk Nông': 'Đăk Nông',
            'Hà Nội': 'Hà Nội',
            'Đà Nẵng': 'Đà Nẵng',
            'Hải Phòng': 'Hải Phòng',
            'Cần Thơ': 'Cần Thơ'
        };

        geojsonData.features.forEach(feature => {
            let name = feature.properties.name || feature.properties.NAME_1 || feature.properties.Name || '';
            // Try to find the matching province
            let provData = null;
            for (let [apiName, p] of provMap) {
                if (PROVINCE_NAME_MAP[apiName] === name || apiName === name) {
                    provData = p;
                    break;
                }
            }
            if (!provData) provData = { province_name: name, risk_level: 'low' };

            const risk = scenarioRisk(provData);
            const height = 0.1 + risk * 2.0;

            const material = new THREE.MeshStandardMaterial({
                color: riskColor(risk),
                roughness: 0.6,
                metalness: 0.1,
                side: THREE.DoubleSide
            });
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0x112233, linewidth: 1, transparent: true, opacity: 0.5 });

            const geoms = feature.geometry;
            const coordsList = geoms.type === 'Polygon' ? [geoms.coordinates] : (geoms.type === 'MultiPolygon' ? geoms.coordinates : []);

            coordsList.forEach(polygon => {
                const shape = new THREE.Shape();
                polygon[0].forEach((coord, idx) => {
                    const pt = projectVietnam(coord[0], coord[1]);
                    if (idx === 0) shape.moveTo(pt.x, pt.y);
                    else shape.lineTo(pt.x, pt.y);
                });

                // create holes if any
                if (polygon.length > 1) {
                    for (let i = 1; i < polygon.length; i++) {
                        const hole = new THREE.Path();
                        polygon[i].forEach((coord, idx) => {
                            const pt = projectVietnam(coord[0], coord[1]);
                            if (idx === 0) hole.moveTo(pt.x, pt.y);
                            else hole.lineTo(pt.x, pt.y);
                        });
                        shape.holes.push(hole);
                    }
                }

                const extrudeSettings = { depth: height, bevelEnabled: false };
                const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                const mesh = new THREE.Mesh(geometry, material);

                // Outline
                const edges = new THREE.EdgesGeometry(geometry);
                const line = new THREE.LineSegments(edges, lineMaterial);
                mesh.add(line);

                mesh.userData = provData;
                scene.add(mesh);
                meshes.push(mesh);
            });
        });

        // Center the camera
        camera.position.set(2, -15, 20);
        if (controls) controls.target.set(2, 4, 0);
    }

    function projectVietnam(lng, lat) {
        const x = (lng - 106.5) * 0.82;
        const y = (lat - 16.2) * 0.88;
        return { x, y };
    }

    function scenarioRisk(province) {
        let score = province.risk_level === 'high' ? 0.72 : province.risk_level === 'medium' ? 0.42 : 0.18;
        score += Math.max(0, -Number(macroParams.gdp_delta_pct || 0)) * 0.018;
        score += Math.max(0, Number(macroParams.unemployment_delta || 0)) * 0.055;
        const rawCompDelta = Number(macroParams.compliance_delta || 0);
        const compDelta = Math.abs(rawCompDelta) > 1 ? rawCompDelta / 100.0 : rawCompDelta;
        score += Math.max(0, -compDelta) * 0.75;
        return Math.max(0.05, Math.min(1, score));
    }

    function riskColor(score) {
        if (score >= 0.62) return 0xef4444;
        if (score >= 0.35) return 0xf59e0b;
        return 0x22c55e;
    }

    function applyMacroParams(params = {}) {
        macroParams = { ...macroParams, ...params };
        if (scene) buildProvinceMeshes();
    }

    function onClick(event) {
        if (!renderer || !camera || !raycaster) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(meshes)[0];
        if (hit?.object?.userData && window.loadProvinceScenario) {
            const province = hit.object.userData;
            window.loadProvinceScenario(province.province_code, province.province_name);
        }
    }


    function animate() {
        if (!renderer || !scene || !camera) return;
        if (controls) controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
    }

    function resize() {
        const container = document.getElementById(containerId);
        if (!container || !renderer || !camera) return;
        camera.aspect = container.clientWidth / Math.max(container.clientHeight, 1);
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    async function fetchJson(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    }

    function getApiBase() {
        return window.API_BASE || 'http://localhost:8000/api';
    }

    return { init, applyMacroParams };
})();

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

        if (leafBtn) leafBtn.classList.remove('bg-white', 'text-primary-container', 'border', 'border-slate-200');
        if (threeBtn) threeBtn.classList.remove('bg-white', 'text-primary-container', 'border', 'border-slate-200');
        if (echartsBtn) echartsBtn.classList.remove('bg-white', 'text-primary-container', 'border', 'border-slate-200');
    }

    if (leafBtn) leafBtn.addEventListener('click', () => {
        resetTabs();
        leaf.classList.remove('hidden');
        leafBtn.classList.add('bg-white', 'text-primary-container', 'border', 'border-slate-200');
        // Trigger resize observer naturally or explicitly dispatch resize
        setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 150); setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 450);
    });

    if (threeBtn) threeBtn.addEventListener('click', async () => {
        resetTabs();
        three.classList.remove('hidden');
        threeBtn.classList.add('bg-white', 'text-primary-container', 'border', 'border-slate-200');
        if (!threeInitialized) {
            threeInitialized = true;
            await Vietnam3DMap.init('vietnam-3d-map');
        }
        setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 150); setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 450);
    });

    if (echartsBtn) echartsBtn.addEventListener('click', async () => {
        resetTabs();
        if (echartsContainer) echartsContainer.classList.remove('hidden');
        echartsBtn.classList.add('bg-white', 'text-primary-container', 'border', 'border-slate-200');
        if (!echartsInitialized && window.VietnamEChartsMap) {
            echartsInitialized = true;
            await window.VietnamEChartsMap.init('vietnam-echarts-map');
        }
        setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 150); setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 450);
    });
});