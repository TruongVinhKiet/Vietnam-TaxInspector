// vietnam_3d_map.js - Three.js extruded 34-unit Vietnam macro risk map.
// Includes desaturated 3D neighboring countries, hover lift, central economic towers, and quick camera presets.

const Vietnam3DMap = (() => {
    let scene, camera, renderer, raycaster, pointer, controls;
    let meshes = [];
    let pillars = [];
    let containerId = 'vietnam-3d-map';
    let geoBounds = null;
    let animationStarted = false;
    let infoOverlay = null;

    let currentHoveredProvinceCode = null;
    let selectedProvinceCode = null;

    // Camera preset animation targets
    let targetCameraPos = null;
    let targetControlsTarget = null;

    // Pointer events variables for drag/click detection
    let clickStartX = 0;
    let clickStartY = 0;
    let clickStartTime = 0;

    async function init(id = 'vietnam-3d-map') {
        containerId = id;
        const container = document.getElementById(containerId);
        if (!container || typeof THREE === 'undefined') return;
        if (!window.MacroMapData) {
            container.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-400">Không tải được dữ liệu bản đồ.</div>';
            return;
        }

        const mapState = await window.MacroMapData.loadState({ force: true });
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
            
            // Constrain angles to keep the map oriented cleanly and prevent going "under" or "behind"
            controls.maxPolarAngle = Math.PI / 2.15; // Prevent going below the horizontal plane
            controls.minPolarAngle = Math.PI / 6;    // Maintain a minimum tilted angle (prevent looking straight down)
            
            // Lock horizontal rotation to +/- 45 degrees so they can't flip it to the back
            controls.maxAzimuthAngle = Math.PI / 4;
            controls.minAzimuthAngle = -Math.PI / 4;

            controls.minDistance = 6;
            controls.maxDistance = 35;

            // Mapping: Left click drags the map (Pan like 2D), Right click rotates/tilts the 3D angle
            controls.mouseButtons = {
                LEFT: THREE.MOUSE.PAN,
                MIDDLE: THREE.MOUSE.DOLLY,
                RIGHT: THREE.MOUSE.ROTATE
            };
        }

        raycaster = new THREE.Raycaster();
        pointer = new THREE.Vector2();

        scene.add(new THREE.AmbientLight(0xffffff, 0.58));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
        keyLight.position.set(8, -18, 28);
        scene.add(keyLight);
        const rimLight = new THREE.DirectionalLight(0x8fb8ff, 0.38);
        rimLight.position.set(-12, 18, 18);
        scene.add(rimLight);

        // Create info overlay element
        createInfoOverlay(container);

        // Build main interactive Vietnam province meshes
        buildProvinceMeshes(mapState.geojson);

        // Load neighboring countries (basemap)
        loadNeighbors();

        // Create camera presets float menu
        createCameraPresets(container);

        frameCountry();

        // Pointer event listeners (Immune to OrbitControls preventDefault)
        const canvas = renderer.domElement;
        canvas.addEventListener('pointerdown', onPointerDown);
        canvas.addEventListener('pointerup', onPointerUp);
        canvas.addEventListener('pointermove', onPointerMove);

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

    function createCameraPresets(container) {
        const old = document.getElementById('three-camera-presets');
        if (old) old.remove();

        const menu = document.createElement('div');
        menu.id = 'three-camera-presets';
        menu.style.cssText = `
            position: absolute; bottom: 12px; left: 12px; z-index: 10;
            display: flex; gap: 8px; background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px); padding: 4px 8px; border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-family: Inter, system-ui, sans-serif; font-size: 11px;
        `;

        const btn3D = createPresetBtn('3D Nghiêng', () => {
            targetCameraPos = new THREE.Vector3(0, -17, 20);
            targetControlsTarget = new THREE.Vector3(0, 0, 0);
        });

        const btn2D = createPresetBtn('2D Phẳng', () => {
            targetCameraPos = new THREE.Vector3(0, -0.01, 26); // Slightly offset from 0 to preserve controls orientation
            targetControlsTarget = new THREE.Vector3(0, 0, 0);
        });

        const btnReset = createPresetBtn('Mặc định', () => {
            targetCameraPos = new THREE.Vector3(0, -20, 21);
            targetControlsTarget = new THREE.Vector3(0, 0, 0);
        });

        const btnDualPillar = createPresetBtn('So sánh GDP', () => {
            toggleDualPillars();
        });

        menu.appendChild(btn3D);
        menu.appendChild(btn2D);
        menu.appendChild(btnReset);
        menu.appendChild(btnDualPillar);
        container.appendChild(menu);
    }

    function createPresetBtn(text, onClick) {
        const btn = document.createElement('button');
        btn.innerText = text;
        btn.style.cssText = `
            background: rgba(255, 255, 255, 0.15); border: none; color: #fff;
            padding: 4px 10px; border-radius: 4px; cursor: pointer;
            font-weight: 700; transition: background 0.2s;
        `;
        btn.addEventListener('mouseenter', () => btn.style.background = 'rgba(255,255,255,0.3)');
        btn.addEventListener('mouseleave', () => btn.style.background = 'rgba(255,255,255,0.15)');
        btn.addEventListener('click', onClick);
        return btn;
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
        // Cleanup old meshes & pillars
        meshes.forEach((mesh) => scene.remove(mesh));
        meshes = [];
        pillars.forEach((pillar) => scene.remove(pillar));
        pillars = [];
        
        if (!geojson?.features?.length) return;

        // Find max GDP to scale pillars
        let maxGdp = 1;
        geojson.features.forEach((feature) => {
            const province = window.MacroMapData.provinceForFeature(feature);
            const gdp = Number(province.gdp_billion_vnd || 0);
            if (gdp > maxGdp) maxGdp = gdp;
        });

        geojson.features.forEach((feature) => {
            const province = window.MacroMapData.provinceForFeature(feature);
            const score = Number(province.risk_score || 0) / 100;
            
            // Subtle extrusion height (0.04 to 0.32 max) to completely eliminate click occlusion
            const height = 0.04 + score * 0.28;
            
            const isSelected = selectedProvinceCode && String(province.province_code) === String(selectedProvinceCode);
            const material = new THREE.MeshStandardMaterial({
                color: riskColor(score),
                roughness: 0.58,
                metalness: 0.08,
                side: THREE.DoubleSide,
                emissive: isSelected ? 0x1d4ed8 : 0x000000,
                emissiveIntensity: isSelected ? 0.35 : 0
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
                
                // Add targetZ property for smooth hover lift animation
                mesh.targetZ = 0;
                
                // Centroid for placing dynamic columns
                const centroid = getPolygonCentroid(polygon);
                
                // Create central glowing cylinders representing GDP
                const gdp = Number(province.gdp_billion_vnd || 0);
                const pillarHeight = 0.1 + (gdp / maxGdp) * 2.2;
                const pillarGeo = new THREE.CylinderGeometry(0.04, 0.04, pillarHeight, 8);
                // Shift pivot to base of cylinder
                pillarGeo.translate(0, pillarHeight / 2, 0);
                
                const pillarMat = new THREE.MeshStandardMaterial({
                    color: riskColor(score),
                    emissive: riskColor(score),
                    emissiveIntensity: 0.65,
                    transparent: true,
                    opacity: 0.72,
                    roughness: 0.2,
                    metalness: 0.1
                });
                const pillarMesh = new THREE.Mesh(pillarGeo, pillarMat);
                
                // Position pillar on top of extruded shape
                pillarMesh.position.set(centroid.x, centroid.y, height);
                pillarMesh.rotation.x = Math.PI / 2;
                pillarMesh.scale.set(1, 0.01, 1); // Start flat for grow intro animation
                
                // Add pillar as child of mesh so it automatically translates/lifts with the province
                mesh.add(pillarMesh);
                pillars.push(pillarMesh);
                
                scene.add(mesh);
                meshes.push(mesh);
            });
        });
    }

    async function loadNeighbors() {
        try {
            const response = await fetch('https://cdn.jsdelivr.net/gh/codeforgermany/click_that_hood@main/public/data/southeast-asia.geojson');
            if (!response.ok) return;
            const geojson = await response.json();
            buildNeighborMeshes(geojson);
        } catch (error) {
            console.warn('[Vietnam3DMap] Could not fetch neighboring countries:', error);
        }
    }

    function buildNeighborMeshes(geojson) {
        if (!geojson?.features?.length) return;

        // Desaturated basemap style for neighboring countries
        const neighborMaterial = new THREE.MeshStandardMaterial({
            color: 0x121a24,
            roughness: 0.85,
            metalness: 0.04,
            side: THREE.DoubleSide,
        });
        const neighborLineMaterial = new THREE.LineBasicMaterial({
            color: 0x1c2736,
            transparent: true,
            opacity: 0.45
        });

        geojson.features.forEach((feature) => {
            // Skip Vietnam, as we render detailed administrative provinces for Vietnam
            if (feature.properties?.name === "Vietnam") return;

            const coordsList = feature.geometry?.type === 'Polygon'
                ? [feature.geometry.coordinates]
                : (feature.geometry?.type === 'MultiPolygon' ? feature.geometry.coordinates : []);

            coordsList.forEach((polygon) => {
                const shape = polygonToShape(polygon);
                if (!shape) return;
                
                // Flat, low-extrusion basemap background (depth = 0.005)
                const geometry = new THREE.ExtrudeGeometry(shape, { depth: 0.005, bevelEnabled: false });
                const mesh = new THREE.Mesh(geometry, neighborMaterial);
                const edges = new THREE.EdgesGeometry(geometry);
                mesh.add(new THREE.LineSegments(edges, neighborLineMaterial));
                
                // Do NOT add to meshes array so they are not raycast/clickable
                scene.add(mesh);
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

    function getPolygonCentroid(polygon) {
        if (!polygon?.[0]?.length) return { x: 0, y: 0 };
        let sumX = 0, sumY = 0, count = 0;
        polygon[0].forEach((coord) => {
            const pt = projectVietnam(coord[0], coord[1]);
            sumX += pt.x;
            sumY += pt.y;
            count += 1;
        });
        return { x: sumX / count, y: sumY / count };
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

    async function switchBoundary(boundaryVersion) {
        if (!scene || !window.MacroMapData) return;
        const normalized = window.MacroMapData.setBoundaryVersion(boundaryVersion);
        const mapState = await window.MacroMapData.loadState({ boundaryVersion: normalized, force: true });
        geoBounds = computeGeoBounds(mapState.geojson);
        buildProvinceMeshes(mapState.geojson);
        meshes.forEach((mesh) => {
            mesh.scale.set(0.88, 0.88, 0.35);
            mesh.userData.__mergeTransition = true;
        });
        frameCountry();
    }

    // Pointer Event Listeners for drag vs click separation
    function onPointerDown(event) {
        clickStartX = event.clientX;
        clickStartY = event.clientY;
        clickStartTime = Date.now();

        // Kill camera transition animations immediately if user drags or interacts
        targetCameraPos = null;
        targetControlsTarget = null;
    }

    function onPointerUp(event) {
        const dx = event.clientX - clickStartX;
        const dy = event.clientY - clickStartY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const duration = Date.now() - clickStartTime;

        // Clean click validation: less than 8px drag distance and click duration
        if (dist < 8) {
            onClick(event);
        }
    }

    function onPointerMove(event) {
        if (!renderer || !camera || !raycaster) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        raycaster.setFromCamera(pointer, camera);
        const intersections = raycaster.intersectObjects(meshes, false);
        
        let hoveredProvinceCode = null;
        if (intersections.length > 0) {
            const hit = intersections[0];
            let obj = hit.object;
            // Walk up to parent if needed
            if (!obj.userData?.province_code && obj.parent?.userData?.province_code) {
                obj = obj.parent;
            }
            hoveredProvinceCode = obj.userData?.province_code || null;
        }

        // Check if hover state changed
        if (hoveredProvinceCode !== currentHoveredProvinceCode) {
            currentHoveredProvinceCode = hoveredProvinceCode;
            
            // Apply hover animation targets
            meshes.forEach((mesh) => {
                const isHovered = currentHoveredProvinceCode && mesh.userData?.province_code === currentHoveredProvinceCode;
                setMeshHoverState(mesh, isHovered);
            });
        }
    }

    function setMeshHoverState(mesh, isHovered) {
        if (!mesh) return;
        mesh.targetZ = isHovered ? 0.15 : 0;
        
        if (mesh.material) {
            const isSelected = selectedProvinceCode && String(mesh.userData?.province_code) === String(selectedProvinceCode);
            if (isHovered) {
                mesh.material.emissive.setHex(0x1d4ed8); // Bright blue hover glow
                mesh.material.emissiveIntensity = 0.45;
            } else if (isSelected) {
                mesh.material.emissive.setHex(0x1d4ed8); // Keep selected glow
                mesh.material.emissiveIntensity = 0.35;
            } else {
                mesh.material.emissive.setHex(0x000000);
                mesh.material.emissiveIntensity = 0;
            }
        }
    }

    function refresh3DSelection() {
        meshes.forEach((mesh) => {
            if (!mesh.material) return;
            const isSelected = selectedProvinceCode && String(mesh.userData?.province_code) === String(selectedProvinceCode);
            if (isSelected) {
                mesh.material.emissive.setHex(0x1d4ed8);
                mesh.material.emissiveIntensity = 0.35;
            } else {
                mesh.material.emissive.setHex(0x000000);
                mesh.material.emissiveIntensity = 0;
            }
        });
    }

    function onClick(event) {
        if (!renderer || !camera || !raycaster) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const intersections = raycaster.intersectObjects(meshes, false);
        
        if (intersections.length === 0) {
            hideInfoCard();
            return;
        }
        
        const hit = intersections[0];
        let province = hit?.object?.userData;
        if ((!province || !province.province_code) && hit?.object?.parent) {
            province = hit.object.parent.userData;
        }
        
        if (province?.province_code) {
            selectedProvinceCode = province.province_code;
            refresh3DSelection();
            showInfoCard(province);
            if (window.loadProvinceScenario) {
                window.loadProvinceScenario(province.province_code, province.province_name);
            }
            window.dispatchEvent(new CustomEvent('macro:province-selected', {
                detail: { provinceCode: province.province_code, provinceName: province.province_name },
            }));
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        if (!renderer || !scene || !camera) return;
        
        // Interpolate camera to preset targets
        if (targetCameraPos) {
            camera.position.lerp(targetCameraPos, 0.08);
            if (camera.position.distanceTo(targetCameraPos) < 0.01) {
                targetCameraPos = null;
            }
        }
        if (targetControlsTarget && controls) {
            controls.target.lerp(targetControlsTarget, 0.08);
            if (controls.target.distanceTo(targetControlsTarget) < 0.01) {
                targetControlsTarget = null;
            }
        }

        // Animate hover lift on meshes
        meshes.forEach((mesh) => {
            const tZ = mesh.targetZ || 0;
            mesh.position.z += (tZ - mesh.position.z) * 0.15;
        });

        // Animate central pillars growth (intro scale)
        pillars.forEach((pillar) => {
            pillar.scale.y += (1.0 - pillar.scale.y) * 0.08;
        });

        // Smooth boundary-switch merge/rebuild transition.
        meshes.forEach((mesh) => {
            if (!mesh.userData?.__mergeTransition) return;
            mesh.scale.x += (1 - mesh.scale.x) * 0.1;
            mesh.scale.y += (1 - mesh.scale.y) * 0.1;
            mesh.scale.z += (1 - mesh.scale.z) * 0.1;
            if (Math.abs(mesh.scale.x - 1) < 0.01 && Math.abs(mesh.scale.z - 1) < 0.01) {
                mesh.scale.set(1, 1, 1);
                mesh.userData.__mergeTransition = false;
            }
        });

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

    // #3 Dual GDP Pillar Comparison Mode
    let dualPillarsActive = false;
    let dualPillarMeshes = [];

    function toggleDualPillars() {
        if (dualPillarsActive) {
            dualPillarMeshes.forEach((m) => scene.remove(m));
            dualPillarMeshes = [];
            dualPillarsActive = false;
            return;
        }
        dualPillarsActive = true;
        const provinces = window.MacroMapData?.previewProvinces?.() || [];
        let maxGdp = 1;
        provinces.forEach((p) => { if (Number(p.gdp_billion_vnd || 0) > maxGdp) maxGdp = Number(p.gdp_billion_vnd); });

        provinces.forEach((province) => {
            if (!province.lat || !province.lng) return;
            const pt = projectVietnam(province.lng, province.lat);
            const gdp2024 = Number(province.gdp_billion_vnd || 0);
            // Estimate 2019 GDP (roughly 75-85% of current based on national growth)
            const ts = province.time_series_preview || [];
            const row2019 = ts.find((r) => Number(r.year) === 2019);
            const gdp2019 = row2019 ? Number(row2019.grdp_billion_vnd_est || gdp2024 * 0.78) : gdp2024 * 0.78;

            const h2019 = 0.1 + (gdp2019 / maxGdp) * 2.2;
            const h2024 = 0.1 + (gdp2024 / maxGdp) * 2.2;
            const offset = 0.08;

            // 2019 pillar (blue)
            const geo19 = new THREE.CylinderGeometry(0.03, 0.03, h2019, 8);
            geo19.translate(0, h2019 / 2, 0);
            const mat19 = new THREE.MeshStandardMaterial({
                color: 0x3b82f6, emissive: 0x3b82f6, emissiveIntensity: 0.5,
                transparent: true, opacity: 0.75, roughness: 0.3,
            });
            const mesh19 = new THREE.Mesh(geo19, mat19);
            mesh19.position.set(pt.x - offset, pt.y, 0.3);
            mesh19.rotation.x = Math.PI / 2;
            mesh19.scale.set(1, 0.01, 1);
            scene.add(mesh19);
            dualPillarMeshes.push(mesh19);

            // 2024 pillar (green)
            const geo24 = new THREE.CylinderGeometry(0.03, 0.03, h2024, 8);
            geo24.translate(0, h2024 / 2, 0);
            const mat24 = new THREE.MeshStandardMaterial({
                color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 0.5,
                transparent: true, opacity: 0.75, roughness: 0.3,
            });
            const mesh24 = new THREE.Mesh(geo24, mat24);
            mesh24.position.set(pt.x + offset, pt.y, 0.3);
            mesh24.rotation.x = Math.PI / 2;
            mesh24.scale.set(1, 0.01, 1);
            scene.add(mesh24);
            dualPillarMeshes.push(mesh24);
        });

        // Animate growth
        const growInterval = setInterval(() => {
            let allDone = true;
            dualPillarMeshes.forEach((m) => {
                m.scale.y += (1.0 - m.scale.y) * 0.1;
                if (Math.abs(m.scale.y - 1) > 0.01) allDone = false;
            });
            if (allDone) clearInterval(growInterval);
        }, 16);
    }

    window.addEventListener('macro:boundary-change', (event) => {
        const boundaryVersion = event.detail?.boundaryVersion || 'vn_34_2025';
        // Clean up dual pillars on boundary switch
        dualPillarMeshes.forEach((m) => scene.remove(m));
        dualPillarMeshes = [];
        dualPillarsActive = false;
        switchBoundary(boundaryVersion);
    });

    window.addEventListener('macro:province-selected', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            refresh3DSelection();
        }
    });

    window.addEventListener('macro:province-resolved-code', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            refresh3DSelection();
        }
    });

    return { init, applyMacroParams, applyProvinceImpacts, switchBoundary, toggleDualPillars, onVisible };
})();

window.Vietnam3DMap = Vietnam3DMap;

document.addEventListener('DOMContentLoaded', () => {
    const leaf = document.getElementById('vietnam-map');
    const three = document.getElementById('vietnam-3d-map');
    const echartsContainer = document.getElementById('vietnam-echarts-map');

    const leafBtn = document.getElementById('map-tab-leaflet');
    const threeBtn = document.getElementById('map-tab-3d');
    const echartsBtn = document.getElementById('map-tab-echarts');

    const scatterBtn = document.getElementById('map-tab-scatter');

    if (!leaf || !three || !leafBtn || !threeBtn) return;

    let threeInitialized = false;
    let echartsInitialized = false;
    let currentTabIsScatter = false;

    function resetTabs() {
        if (leaf) leaf.classList.add('hidden');
        if (three) three.classList.add('hidden');
        if (echartsContainer) echartsContainer.classList.add('hidden');
        currentTabIsScatter = false;
        [leafBtn, threeBtn, echartsBtn, scatterBtn].forEach((btn) => {
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
            } else {
                // Re-render map mode (not scatter) when switching back
                await window.VietnamEChartsMap?.switchBoundary?.(window.MACRO_BOUNDARY_VERSION || 'vn_34_2025');
            }
            setTimeout(() => window.VietnamEChartsMap?.onVisible?.(), 80);
            setTimeout(() => window.VietnamEChartsMap?.onVisible?.(), 280);
        });
    }

    if (scatterBtn && echartsContainer) {
        scatterBtn.addEventListener('click', async () => {
            resetTabs();
            echartsContainer.classList.remove('hidden');
            activate(scatterBtn);
            currentTabIsScatter = true;
            if (!echartsInitialized && window.VietnamEChartsMap) {
                echartsInitialized = true;
                await window.VietnamEChartsMap.init('vietnam-echarts-map');
            }
            window.VietnamEChartsMap?.renderScatterGdpTax?.();
            setTimeout(() => window.VietnamEChartsMap?.onVisible?.(), 80);
        });
    }
});
