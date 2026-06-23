// Digital exhibition — main script
// 3D exhibition space with Three.js

// Ensure Three.js is loaded
if (typeof THREE === 'undefined') {
    console.error('Three.js failed to load. Check your network or CDN.');
    document.body.innerHTML = '<div style="color: white; padding: 20px; text-align: center; background: rgba(0,0,0,0.8);"><h2>Error: Three.js library not loaded</h2><p>Check your network and refresh the page.</p></div>';
    throw new Error('THREE is not defined');
}

let scene, camera, renderer;
let controls;
let exhibitionItems = [];
let gltfLoader = null;
let displayCase = null; // Display case root
let displayCaseGroup = null; // Display case group for exhibits
let currentImportedModel = null; // Scene-imported model (replaceable)
let currentDisplayCaseModel = null; // Model inside display case (replaceable)
let currentDisplayCaseModelInitialRotation = null; // Initial rotation when model loaded
let currentDisplayCaseModelInitialRotationTarget = null; // Target for initial rotation
let currentImportedModelBaseScale = 1; // Base scale for scene import
let currentDisplayCaseModelBaseScale = 1; // Base scale for case import
let currentDisplayCaseModelScaleValue = 100; // Current UI scale for case model
let currentDisplayCaseModelSourceArrayBuffer = null; // Case model source buffer (preferred)
let currentDisplayCaseModelSourceUrl = null; // Case model source URL (fallback)
let currentDisplayCaseModelSourceFileName = 'model.glb'; // Case model source file name
let displayCasePedestal = null; // Pedestal mesh
let displayCaseBase = null; // Base mesh
let displayCasePlatform = null; // Platform mesh
let displayCaseGlasses = []; // Glass panels
let displayCaseFrames = []; // Frame meshes
let displayCaseLight = null; // Interior light
let displayCaseBaseHeight = 0.3; // Base thickness
let displayCasePedestalHeight = 3.2; // Pedestal height (scale 32 → 3.2)
let displayCaseRotationDirection = 0; // Horizontal spin: -1 left, 0 stop, 1 right
let displayCaseVerticalRotationDirection = 0; // Vertical tilt: -1 down, 0 stop, 1 up
let displayCaseRotationSpeed = 0.08; // Rotation speed (matches scale 80)
let displayCaseRotationSpeedScale = 80; // Speed scale 1–100 (100 = old 0.1)
let raycaster = null; // Raycaster
let heldObject = null; // Held object
let heldObjectOriginalParent = null; // Original parent of held object
let heldObjectOriginalPosition = null; // Original position
let heldObjectOriginalRotation = null; // Original rotation
let heldObjectOriginalScale = null; // Original scale
let heldObjectDistance = -3.0; // Held distance along -Z (in front)
let pickableObjects = []; // Pickable objects
let heldObjectContainer = null; // Held-object group (follows camera)
let lastLoadedFile = null; // Last loaded file metadata
let textureLoader = null; // Texture loader
let iframeInteractionActive = (window === window.top); // In iframe: hover state from parent

function isIframeInteractionEnabled() {
    if (window === window.top) return true;
    return iframeInteractionActive;
}

function setIframeInteractionActive(active) {
    iframeInteractionActive = !!active;
    if (!iframeInteractionActive) {
        // Stop spin when leaving interaction area
        displayCaseRotationDirection = 0;
        displayCaseVerticalRotationDirection = 0;
    }
}

// initGLTFLoader
function initGLTFLoader() {
    // Try several ways to load GLTFLoader
    if (typeof THREE !== 'undefined') {
        if (THREE.GLTFLoader) {
            gltfLoader = new THREE.GLTFLoader();
            console.log('GLTFLoader initialized');
        } else {
            // Dynamic-load GLTFLoader
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/three@0.144.0/examples/js/loaders/GLTFLoader.js';
            script.onload = function() {
                if (THREE.GLTFLoader) {
                    gltfLoader = new THREE.GLTFLoader();
                    console.log('GLTFLoader loaded dynamically');
                }
            };
            document.head.appendChild(script);
            console.warn('GLTFLoader not preloaded; attempting dynamic load');
        }
    } else {
        console.warn('THREE is undefined; GLB loading unavailable');
    }
}

// init scene
function init() {
    // Scene — bright minimal look
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff); // White background
    scene.fog = new THREE.Fog(0xffffff, 50, 200); // Light fog for depth

    const container = document.getElementById('canvas-container');
    let cw = container ? container.clientWidth : 0;
    let ch = container ? container.clientHeight : 0;
    if (cw <= 0 || ch <= 0) {
        cw = window.innerWidth || 1;
        ch = window.innerHeight || 1;
    }

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        cw / ch,
        0.1,
        1000
    );
    camera.position.set(0, 6, 8); // Start behind display case

    // Renderer — high quality
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "high-performance",
        stencil: false,
        depth: true,
        preserveDrawingBuffer: true // required for canvas.toDataURL screenshots
    });
    // Pixel ratio cap for quality
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap DPR at 2
    renderer.setSize(cw, ch);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Soft shadows
    renderer.shadowMap.autoUpdate = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; // ACES tone mapping
    renderer.toneMappingExposure = 1.0;
    renderer.outputEncoding = THREE.sRGBEncoding; // sRGB output
    renderer.physicallyCorrectLights = true; // Physically correct lights
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // initGLTFLoader
    initGLTFLoader();
    
    // TextureLoader
    textureLoader = new THREE.TextureLoader();

    // Raycaster
    raycaster = new THREE.Raycaster();

    // Held-object container
    heldObjectContainer = new THREE.Group();
    scene.add(heldObjectContainer);

    // Lights
    setupLighting();

    // Room geometry
    createExhibitionSpace();

    // Exhibits
    addExhibitionItems();

    // Orbit controls
    setupControls();

    // Resize handler
    window.addEventListener('resize', onWindowResize);

    // Animation loop
    animate();

    window.exhibitionSceneReady = true;
    setupExhibitionAssistantChat();
}

/** Capture current WebGL view as JPEG data URL (current camera). */
function captureExhibitionViewDataUrl() {
    if (!renderer || !scene || !camera) return '';
    renderer.render(scene, camera);
    try {
        return renderer.domElement.toDataURL('image/jpeg', 0.88);
    } catch (e) {
        console.error('captureExhibitionViewDataUrl failed', e);
        return '';
    }
}

/**
 * Downscale JPEG data URL so VL APIs stay within context (full-screen × DPR screenshots are huge).
 * Longest edge capped at maxSide px.
 */
function downscaleJpegDataUrl(dataUrl, maxSide, quality) {
    return new Promise(function (resolve, reject) {
        if (!dataUrl || typeof Image === 'undefined') {
            resolve(dataUrl);
            return;
        }
        var img = new Image();
        img.onload = function () {
            try {
                var w = img.naturalWidth || img.width;
                var h = img.naturalHeight || img.height;
                if (!w || !h) {
                    resolve(dataUrl);
                    return;
                }
                var scale = Math.min(1, maxSide / Math.max(w, h));
                var tw = Math.max(1, Math.round(w * scale));
                var th = Math.max(1, Math.round(h * scale));
                var c = document.createElement('canvas');
                c.width = tw;
                c.height = th;
                var ctx = c.getContext('2d');
                if (!ctx) {
                    resolve(dataUrl);
                    return;
                }
                ctx.drawImage(img, 0, 0, tw, th);
                resolve(c.toDataURL('image/jpeg', quality));
            } catch (e) {
                reject(e);
            }
        };
        img.onerror = function () {
            reject(new Error('image load failed'));
        };
        img.src = dataUrl;
    });
}

function getExhibitionChatApiBase() {
    try {
        var params = new URLSearchParams(window.location.search || '');
        var q = params.get('chatApi');
        if (q) return q.replace(/\/$/, '');
    } catch (e) {}
    try {
        var stored = localStorage.getItem('exhibition-chat-api');
        if (stored) return stored.replace(/\/$/, '');
    } catch (e2) {}
    if (typeof window.EXHIBITION_CHAT_API === 'string' && window.EXHIBITION_CHAT_API) {
        return window.EXHIBITION_CHAT_API.replace(/\/$/, '');
    }
    return '';
}

function setupExhibitionAssistantChat() {
    var fab = document.getElementById('exhibitionChatFab');
    var panel = document.getElementById('exhibitionChatPanel');
    var closeBtn = document.getElementById('exhibitionChatClose');
    var sendBtn = document.getElementById('exhibitionChatSend');
    var input = document.getElementById('exhibitionChatInput');
    var messagesEl = document.getElementById('exhibitionChatMessages');
    var deepCb = document.getElementById('exhibitionDeepResearch');
    var hintEl = document.getElementById('exhibitionChatApiHint');
    if (!fab || !panel || !sendBtn || !input || !messagesEl) return;

    function refreshApiHint() {
        var base = getExhibitionChatApiBase();
        if (hintEl) {
            if (!base) {
                hintEl.textContent =
                    'Inference API not configured: add ?chatApi=http://127.0.0.1:8765 to the URL or set localStorage exhibition-chat-api';
                hintEl.style.display = 'block';
            } else {
                hintEl.textContent = 'API: ' + base;
                hintEl.style.display = 'block';
            }
        }
    }
    refreshApiHint();

    function appendBubble(role, text, extraClass) {
        var wrap = document.createElement('div');
        wrap.className = 'exhibition-chat-bubble exhibition-chat-bubble--' + role + (extraClass ? ' ' + extraClass : '');
        wrap.textContent = text;
        messagesEl.appendChild(wrap);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    /** Deep-research trajectory: prompts, assistant rounds, tool outputs (before final answer). */
    function appendResearchSteps(steps) {
        if (!steps || !steps.length) return;
        var container = document.createElement('div');
        container.className = 'exhibition-chat-trajectory';
        var head = document.createElement('div');
        head.className = 'exhibition-chat-trajectory-head';
        head.textContent = 'Research trace';
        container.appendChild(head);

        for (var i = 0; i < steps.length; i++) {
            var s = steps[i];
            var row = document.createElement('div');
            row.className =
                'exhibition-chat-step exhibition-chat-step--' + String(s.kind || 'unknown');
            var title = document.createElement('div');
            title.className = 'exhibition-chat-step-title';
            title.textContent = s.title || s.kind || '';
            row.appendChild(title);

            if (s.text) {
                var pre = document.createElement('pre');
                pre.className = 'exhibition-chat-step-body';
                pre.textContent = s.text;
                row.appendChild(pre);
            }

            if (s.tool_calls && s.tool_calls.length) {
                var tcBox = document.createElement('div');
                tcBox.className = 'exhibition-chat-toolcalls';
                for (var j = 0; j < s.tool_calls.length; j++) {
                    var tc = s.tool_calls[j];
                    var tcRow = document.createElement('div');
                    tcRow.className = 'exhibition-chat-toolcall';
                    var nm = document.createElement('span');
                    nm.className = 'exhibition-chat-toolcall-name';
                    nm.textContent = tc.name || '(tool)';
                    tcRow.appendChild(nm);
                    if (tc.arguments) {
                        var argsPre = document.createElement('pre');
                        argsPre.className = 'exhibition-chat-toolcall-args';
                        argsPre.textContent = tc.arguments;
                        tcRow.appendChild(argsPre);
                    }
                    tcBox.appendChild(tcRow);
                }
                row.appendChild(tcBox);
            }

            container.appendChild(row);
        }

        messagesEl.appendChild(container);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    /** User message with optional screenshot preview (same image sent to API). */
    function appendUserBubbleWithScreenshot(text, imageDataUrl) {
        var wrap = document.createElement('div');
        wrap.className = 'exhibition-chat-bubble exhibition-chat-bubble--user';
        if (imageDataUrl) {
            var label = document.createElement('div');
            label.className = 'exhibition-chat-thumb-label';
            label.textContent = 'View at send time';
            wrap.appendChild(label);
            var img = document.createElement('img');
            img.className = 'exhibition-chat-thumb';
            img.src = imageDataUrl;
            img.alt = 'Current view screenshot';
            img.loading = 'lazy';
            wrap.appendChild(img);
        }
        var body = document.createElement('div');
        body.className = 'exhibition-chat-user-q';
        body.textContent = text;
        wrap.appendChild(body);
        messagesEl.appendChild(wrap);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setBusy(b) {
        sendBtn.disabled = b;
        input.disabled = b;
        if (deepCb) deepCb.disabled = b;
    }

    fab.addEventListener('click', function () {
        var open = panel.classList.toggle('open');
        fab.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) refreshApiHint();
    });
    if (closeBtn) {
        closeBtn.addEventListener('click', function () {
            panel.classList.remove('open');
            fab.setAttribute('aria-expanded', 'false');
        });
    }

    function submitChat() {
        var q = (input.value || '').trim();
        if (!q) return;
        var base = getExhibitionChatApiBase();
        if (!base) {
            appendBubble('system', 'Configure chatApi first (see hint below).');
            return;
        }
        var deep = deepCb && deepCb.checked;

        var rawShot = captureExhibitionViewDataUrl();
        if (!rawShot) {
            appendBubble('system', 'Could not capture the current view. Please try again.');
            return;
        }

        /* Longest edge 256px — minimize vision tokens (tool transcripts need room in context) */
        var API_IMAGE_MAX_SIDE = 256;
        var API_JPEG_QUALITY = 0.82;

        downscaleJpegDataUrl(rawShot, API_IMAGE_MAX_SIDE, API_JPEG_QUALITY)
            .catch(function () {
                return rawShot;
            })
            .then(function (imgDataUrl) {
                appendUserBubbleWithScreenshot(q, imgDataUrl);
                input.value = '';

                appendBubble(
                    'assistant',
                    deep ? 'Deep research in progress (search & tools)…' : 'Analyzing current view…',
                    'exhibition-chat-bubble--pending'
                );
                var pending = messagesEl.lastChild;
                setBusy(true);

                var ctrl = new AbortController();
                var timer = setTimeout(function () {
                    ctrl.abort();
                }, deep ? 600000 : 180000);

                return fetch(base + '/v1/exhibition/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: q,
                        image: imgDataUrl,
                        deep_research: deep
                    }),
                    signal: ctrl.signal
                })
                    .then(function (res) {
                        return res.json().then(function (data) {
                            return { ok: res.ok, status: res.status, data: data };
                        });
                    })
                    .then(function (r) {
                        if (pending && pending.parentNode) pending.parentNode.removeChild(pending);
                        if (!r.ok) {
                            var err = (r.data && (r.data.detail || r.data.error)) || r.status;
                            appendBubble('system', 'Request failed: ' + err);
                            return;
                        }
                        var ans = (r.data && r.data.answer) || '';
                        var mode = (r.data && r.data.mode) || '';
                        if (!ans) {
                            appendBubble('assistant', '(No text returned by the model)');
                            return;
                        }
                        if (mode === 'search' && r.data.steps && r.data.steps.length) {
                            appendResearchSteps(r.data.steps);
                        }
                        var prefix =
                            mode === 'search' ? '[Research · final] ' : '[Direct] ';
                        appendBubble('assistant', prefix + ans);
                    })
                    .catch(function (e) {
                        if (pending && pending.parentNode) pending.parentNode.removeChild(pending);
                        appendBubble('system', 'Network error: ' + (e && e.message ? e.message : String(e)));
                    })
                    .finally(function () {
                        clearTimeout(timer);
                        setBusy(false);
                    });
            });
    }

    sendBtn.addEventListener('click', submitChat);
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitChat();
        }
    });
}

// Lighting — bright white
function setupLighting() {
    // Ambient — keep shadows readable
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Lower ambient for clearer shadows
    scene.add(ambientLight);

    // Key directional from above
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8); // Moderate key intensity
    mainLight.position.set(0, 20, 0);
    mainLight.castShadow = true;
    // Higher shadow map resolution
    mainLight.shadow.mapSize.width = 4096;
    mainLight.shadow.mapSize.height = 4096;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 50;
    mainLight.shadow.camera.left = -20;
    mainLight.shadow.camera.right = 20;
    mainLight.shadow.camera.top = 20;
    mainLight.shadow.camera.bottom = -20;
    mainLight.shadow.bias = -0.0001; // Shadow bias
    mainLight.shadow.normalBias = 0.02; // Normal bias
    mainLight.shadow.radius = 8; // Shadow radius
    scene.add(mainLight);

    // Fill directional
    const sideLight = new THREE.DirectionalLight(0xffffff, 0.4); // Lower fill intensity
    sideLight.position.set(15, 10, 15);
    scene.add(sideLight);

    // Point lights for exhibits
    const pointLight1 = new THREE.PointLight(0xffffff, 0.5, 30); // Lower point intensity
    pointLight1.position.set(-8, 8, -8);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xffffff, 0.5, 30); // Lower point intensity
    pointLight2.position.set(8, 8, 8);
    scene.add(pointLight2);
    
    // Ceiling spotlights
    const spotLight1 = new THREE.SpotLight(0xffffff, 0.7, 25, Math.PI / 6, 0.3); // Lower spot intensity
    spotLight1.position.set(-5, 12, 0);
    spotLight1.target.position.set(-5, 0, 0);
    spotLight1.castShadow = true;
    scene.add(spotLight1);
    scene.add(spotLight1.target);
    
    const spotLight2 = new THREE.SpotLight(0xffffff, 0.7, 25, Math.PI / 6, 0.3); // Lower spot intensity
    spotLight2.position.set(5, 12, 0);
    spotLight2.target.position.set(5, 0, 0);
    spotLight2.castShadow = true;
    scene.add(spotLight2);
    scene.add(spotLight2.target);
}

// Museum-style room
function createExhibitionSpace() {
    // Floor — light, high segment count
    const floorGeometry = new THREE.PlaneGeometry(40, 40, 40, 40); // 40×40 segments
    const floorMaterial = new THREE.MeshStandardMaterial({
        color: 0xf0f0f0, // Light gray floor vs walls
        roughness: 0.5, // Roughness for shadow read
        metalness: 0.0,
        flatShading: false // Smooth shading
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Floor grid — tile lines
    const floorGrid = new THREE.GridHelper(40, 8, 0x000000, 0x333333); // Dark grid lines
    floorGrid.position.y = 0.01;
    scene.add(floorGrid);

    // Procedural wall texture
    function createMuseumWallTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        
        // Clean white wall base
        const baseColor = '#ffffff'; // Pure white
        ctx.fillStyle = baseColor;
        ctx.fillRect(0, 0, 1024, 1024);
        
        // Subtle noise
        for (let i = 0; i < 1000; i++) {
            const x = Math.random() * 1024;
            const y = Math.random() * 1024;
            const size = Math.random() * 1.0 + 0.3;
            const brightness = Math.random() * 0.01 - 0.005; // Tiny variation
            const alpha = Math.random() * 0.015 + 0.005;
            
            ctx.fillStyle = `rgba(${255 + brightness * 255}, ${255 + brightness * 255}, ${255 + brightness * 255}, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Subtle horizontal bands
        for (let i = 0; i < 10; i++) {
            const y = (i * 1024 / 10) + (Math.random() - 0.5) * 3;
            const alpha = 0.005 + Math.random() * 0.01;
            ctx.strokeStyle = `rgba(250, 250, 250, ${alpha})`;
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(1024, y);
            ctx.stroke();
        }
        
        // Faint vertical seams
        for (let i = 0; i < 6; i++) {
            const x = (i * 1024 / 6) + (Math.random() - 0.5) * 1;
            const alpha = 0.01 + Math.random() * 0.015;
            ctx.strokeStyle = `rgba(245, 245, 245, ${alpha})`;
            ctx.lineWidth = 0.3;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, 1024);
            ctx.stroke();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.repeat.set(4, 2);
        texture.encoding = THREE.sRGBEncoding;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        return texture;
    }
    
    // Use procedural wall map
    let wallTexture = createMuseumWallTexture();
    let wallNormalTexture = null;
    let wallRoughnessTexture = null;
    
    // Walls with texture
    const wallMaterial = new THREE.MeshStandardMaterial({
        map: wallTexture, // Bright texture
        normalMap: wallNormalTexture, // Optional normal map
        roughnessMap: wallRoughnessTexture, // Optional roughness map
        color: 0xffffff, // White albedo
        roughness: 0.3, // Lower roughness — smoother wall
        metalness: 0.0, // Non-metallic
        side: THREE.DoubleSide, // DoubleSide
        flatShading: false // Smooth shading
    });

    // Back wall behind camera
    const backWall = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 15, 40, 15), // Higher segment count
        wallMaterial
    );
    backWall.position.set(0, 7.5, 20); // z = 20 behind camera
    backWall.rotation.y = Math.PI; // Face into scene (-Z)
    backWall.receiveShadow = true;
    backWall.castShadow = true; // Wall casts shadow
    scene.add(backWall);

    // Left wall
    const leftWall = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 15, 40, 15), // Higher segment count
        wallMaterial
    );
    leftWall.rotation.y = Math.PI / 2;
    leftWall.position.set(-20, 7.5, 0);
    leftWall.receiveShadow = true;
    leftWall.castShadow = true; // Wall casts shadow
    scene.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 15, 40, 15), // Higher segment count
        wallMaterial
    );
    rightWall.rotation.y = -Math.PI / 2;
    rightWall.position.set(20, 7.5, 0);
    rightWall.receiveShadow = true;
    rightWall.castShadow = true; // Wall casts shadow
    scene.add(rightWall);

    // Front wall
    const frontWall = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 15, 40, 15), // Higher segment count
        wallMaterial
    );
    frontWall.position.set(0, 7.5, -20); // In front of camera
    frontWall.rotation.y = 0; // Front wall at z=-20 faces inward
    frontWall.receiveShadow = true;
    frontWall.castShadow = true; // Wall casts shadow
    scene.add(frontWall);
    
    // Extra light for front wall
    const frontWallLight = new THREE.PointLight(0xfff5e6, 0.6, 25); // Brighter front-wall fill
    frontWallLight.position.set(0, 8, -15); // Slightly in front of front wall
    scene.add(frontWallLight);

    // Procedural ceiling texture
    function createCeilingTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // White ceiling base
        const baseColor = '#ffffff';
        ctx.fillStyle = baseColor;
        ctx.fillRect(0, 0, 512, 512);
        
        // Subtle ceiling noise
        for (let i = 0; i < 300; i++) {
            const x = Math.random() * 512;
            const y = Math.random() * 512;
            const size = Math.random() * 1.5 + 0.3;
            const alpha = Math.random() * 0.02 + 0.01;
            
            ctx.fillStyle = `rgba(${Math.random() > 0.5 ? '255' : '250'}, ${Math.random() > 0.5 ? '255' : '250'}, ${Math.random() > 0.5 ? '255' : '250'}, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Faint ceiling grid
        ctx.strokeStyle = 'rgba(245, 245, 245, 0.05)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < 12; i++) {
            const pos = (i * 512 / 12);
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, 512);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(512, pos);
            ctx.stroke();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        // ClampToEdge for procedural tex
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.repeat.set(1, 1);
        texture.encoding = THREE.sRGBEncoding;
        return texture;
    }
    
    // Use procedural ceiling
    let ceilingTexture = createCeilingTexture();
    let ceilingNormalTexture = null;
    
    // Ceiling with texture
    const ceilingMaterial = new THREE.MeshStandardMaterial({ 
        map: ceilingTexture, // Procedural map
        normalMap: ceilingNormalTexture, // Optional normal map
        color: 0xffffff, // White albedo
        roughness: 0.3, // Smoother brighter ceiling
        metalness: 0.0,
        flatShading: false
    });
    const ceiling = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 40, 40, 40), // Higher segment count
        ceilingMaterial
    );
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.y = 15;
    scene.add(ceiling);
}

// Exhibits
function addExhibitionItems() {
    // Display case
    createDisplayCase();

    // Default vase from models/default_vase.glb (silent skip)
    loadDefaultVaseInDisplayCase();
}

// Painting
function createPainting(color, width, height) {
    const frameGeometry = new THREE.BoxGeometry(width + 0.2, height + 0.2, 0.1);
    const frameMaterial = new THREE.MeshStandardMaterial({
        color: 0x8B4513,
        roughness: 0.8
    });
    const frame = new THREE.Mesh(frameGeometry, frameMaterial);

    const canvasGeometry = new THREE.PlaneGeometry(width, height);
    const canvasMaterial = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.9
    });
    const canvas = new THREE.Mesh(canvasGeometry, canvasMaterial);
    canvas.position.z = 0.06;

    const painting = new THREE.Group();
    painting.add(frame);
    painting.add(canvas);
    painting.castShadow = true;
    painting.receiveShadow = true;

    return painting;
}

// Vase lathe
function createVase(color, radius, height) {
    const vase = new THREE.Group();

    // Lathe vase profile
    const points = [];
    const segments = 20;
    
    // Profile points bottom to top
    // Narrow base, wide belly, narrow neck
    points.push(new THREE.Vector2(radius * 0.3, 0)); // Bottom
    points.push(new THREE.Vector2(radius * 0.4, height * 0.1));
    points.push(new THREE.Vector2(radius * 0.6, height * 0.2));
    points.push(new THREE.Vector2(radius * 0.9, height * 0.4)); // Widest
    points.push(new THREE.Vector2(radius * 0.95, height * 0.6));
    points.push(new THREE.Vector2(radius * 0.8, height * 0.75)); // Neck start
    points.push(new THREE.Vector2(radius * 0.6, height * 0.85));
    points.push(new THREE.Vector2(radius * 0.5, height * 0.92));
    points.push(new THREE.Vector2(radius * 0.45, height * 0.97));
    points.push(new THREE.Vector2(radius * 0.4, height)); // Rim

    const vaseGeometry = new THREE.LatheGeometry(points, segments);
    const vaseMaterial = new THREE.MeshStandardMaterial({
        color: color,
        metalness: 0.3,
        roughness: 0.4,
        side: THREE.DoubleSide
    });
    const vaseBody = new THREE.Mesh(vaseGeometry, vaseMaterial);
    vaseBody.castShadow = true;
    vaseBody.receiveShadow = true;
    vase.add(vaseBody);

    // Decor rings
    for (let i = 0; i < 2; i++) {
        const lineGeometry = new THREE.RingGeometry(
            radius * 0.85 + i * 0.05,
            radius * 0.9 + i * 0.05,
            segments
        );
        const lineMaterial = new THREE.MeshStandardMaterial({
            color: color * 0.7,
            metalness: 0.5,
            roughness: 0.3
        });
        const line = new THREE.Mesh(lineGeometry, lineMaterial);
        line.rotation.x = -Math.PI / 2;
        line.position.y = height * (0.3 + i * 0.2);
        vase.add(line);
    }

    return vase;
}

// Table
function createTable() {
    const table = new THREE.Group();

    // Tabletop
    const tableTop = new THREE.Mesh(
        new THREE.BoxGeometry(3, 0.1, 1.5),
        new THREE.MeshStandardMaterial({
            color: 0x8B4513, // Brown wood
            roughness: 0.8,
            metalness: 0.1
        })
    );
    tableTop.position.y = 1.0;
    tableTop.castShadow = true;
    tableTop.receiveShadow = true;
    table.add(tableTop);

    // Leg FL
    const leg1 = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 0.1),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.7
        })
    );
    leg1.position.set(-1.4, 0.5, 0.6);
    leg1.castShadow = true;
    leg1.receiveShadow = true;
    table.add(leg1);

    // Leg FR
    const leg2 = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 0.1),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.7
        })
    );
    leg2.position.set(1.4, 0.5, 0.6);
    leg2.castShadow = true;
    leg2.receiveShadow = true;
    table.add(leg2);

    // Leg BL
    const leg3 = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 0.1),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.7
        })
    );
    leg3.position.set(-1.4, 0.5, -0.6);
    leg3.castShadow = true;
    leg3.receiveShadow = true;
    table.add(leg3);

    // Leg BR
    const leg4 = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 0.1),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.7
        })
    );
    leg4.position.set(1.4, 0.5, -0.6);
    leg4.castShadow = true;
    leg4.receiveShadow = true;
    table.add(leg4);

    // Table edge trim
    const edge1 = new THREE.Mesh(
        new THREE.BoxGeometry(3, 0.05, 0.05),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.6
        })
    );
    edge1.position.set(0, 1.025, 0.725);
    table.add(edge1);

    const edge2 = new THREE.Mesh(
        new THREE.BoxGeometry(3, 0.05, 0.05),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.6
        })
    );
    edge2.position.set(0, 1.025, -0.725);
    table.add(edge2);

    const edge3 = new THREE.Mesh(
        new THREE.BoxGeometry(0.05, 0.05, 1.5),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.6
        })
    );
    edge3.position.set(1.475, 1.025, 0);
    table.add(edge3);

    const edge4 = new THREE.Mesh(
        new THREE.BoxGeometry(0.05, 0.05, 1.5),
        new THREE.MeshStandardMaterial({
            color: 0x654321,
            roughness: 0.6
        })
    );
    edge4.position.set(-1.475, 1.025, 0);
    table.add(edge4);

    return table;
}

// Build display case
function createDisplayCase() {
    const caseGroup = new THREE.Group();
    
    // Case dimensions
    const caseWidth = 3;
    const caseHeight = 2.5;
    const caseDepth = 1.5;
    const glassThickness = 0.05;
    displayCaseBaseHeight = 0.3;
    displayCasePedestalHeight = 3.2; // Pedestal raises case (scale 32 → 3.2)
    
    // Pedestal block
    const pedestal = new THREE.Mesh(
        new THREE.BoxGeometry(caseWidth + 0.2, displayCasePedestalHeight, caseDepth + 0.2),
        new THREE.MeshStandardMaterial({
            color: 0x3a3a3a,
            roughness: 0.6,
            metalness: 0.4
        })
    );
    pedestal.position.y = displayCasePedestalHeight / 2;
    pedestal.castShadow = true;
    pedestal.receiveShadow = true;
    caseGroup.add(pedestal);
    displayCasePedestal = pedestal; // Keep pedestal ref
    
    // Dark metal base
    const base = new THREE.Mesh(
        new THREE.BoxGeometry(caseWidth + 0.2, displayCaseBaseHeight, caseDepth + 0.2),
        new THREE.MeshStandardMaterial({
            color: 0x2c2c2c,
            roughness: 0.3,
            metalness: 0.7
        })
    );
    base.position.y = displayCasePedestalHeight + displayCaseBaseHeight / 2;
    base.castShadow = true;
    base.receiveShadow = true;
    caseGroup.add(base);
    displayCaseBase = base; // Keep base ref
    
    // White platform
    const platform = new THREE.Mesh(
        new THREE.BoxGeometry(caseWidth - 0.1, 0.05, caseDepth - 0.1),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            roughness: 0.8,
            metalness: 0.1
        })
    );
    platform.position.y = displayCasePedestalHeight + displayCaseBaseHeight + 0.025;
    platform.receiveShadow = true;
    caseGroup.add(platform);
    displayCasePlatform = platform; // Keep platform ref
    
    // Glass material
    const glassMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.1,
        roughness: 0.1,
        metalness: 0.9,
        side: THREE.DoubleSide
    });
    
    // Glass/frame height vs pedestal
    const glassBaseY = displayCasePedestalHeight + displayCaseBaseHeight + 0.05;
    
    // Reset glass/frame arrays
    displayCaseGlasses = [];
    displayCaseFrames = [];
    
    // Front glass
    const frontGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(caseWidth, caseHeight),
        glassMaterial
    );
    frontGlass.position.set(0, glassBaseY + caseHeight / 2, caseDepth / 2 + glassThickness / 2);
    frontGlass.castShadow = false;
    caseGroup.add(frontGlass);
    displayCaseGlasses.push(frontGlass);
    
    // Back glass
    const backGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(caseWidth, caseHeight),
        glassMaterial
    );
    backGlass.position.set(0, glassBaseY + caseHeight / 2, -caseDepth / 2 - glassThickness / 2);
    caseGroup.add(backGlass);
    displayCaseGlasses.push(backGlass);
    
    // Left glass
    const leftGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(caseDepth, caseHeight),
        glassMaterial
    );
    leftGlass.rotation.y = Math.PI / 2;
    leftGlass.position.set(-caseWidth / 2 - glassThickness / 2, glassBaseY + caseHeight / 2, 0);
    caseGroup.add(leftGlass);
    displayCaseGlasses.push(leftGlass);
    
    // Right glass
    const rightGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(caseDepth, caseHeight),
        glassMaterial
    );
    rightGlass.rotation.y = -Math.PI / 2;
    rightGlass.position.set(caseWidth / 2 + glassThickness / 2, glassBaseY + caseHeight / 2, 0);
    caseGroup.add(rightGlass);
    displayCaseGlasses.push(rightGlass);
    
    // Top glass
    const topGlassMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.2, // Higher opacity hint
        roughness: 0.1,
        metalness: 0.9,
        side: THREE.DoubleSide
    });
    const topGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(caseWidth, caseDepth),
        topGlassMaterial
    );
    topGlass.rotation.x = -Math.PI / 2;
    topGlass.position.set(0, glassBaseY + caseHeight, 0);
    caseGroup.add(topGlass);
    displayCaseGlasses.push(topGlass);
    
    // Top glass frame edges
    const topFrameMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a,
        roughness: 0.2,
        metalness: 0.8
    });
    const topFrameThickness = 0.05; // Frame thickness
    const topFrameHeight = 0.08; // Frame height above glass
    
    // Front top frame
    const topFrameFront = new THREE.Mesh(
        new THREE.BoxGeometry(caseWidth, topFrameThickness, topFrameHeight),
        topFrameMaterial
    );
    topFrameFront.position.set(0, glassBaseY + caseHeight + topFrameHeight / 2, caseDepth / 2);
    caseGroup.add(topFrameFront);
    displayCaseFrames.push(topFrameFront);
    
    // Back top frame
    const topFrameBack = new THREE.Mesh(
        new THREE.BoxGeometry(caseWidth, topFrameThickness, topFrameHeight),
        topFrameMaterial
    );
    topFrameBack.position.set(0, glassBaseY + caseHeight + topFrameHeight / 2, -caseDepth / 2);
    caseGroup.add(topFrameBack);
    displayCaseFrames.push(topFrameBack);
    
    // Left top frame
    const topFrameLeft = new THREE.Mesh(
        new THREE.BoxGeometry(topFrameThickness, topFrameHeight, caseDepth),
        topFrameMaterial
    );
    topFrameLeft.position.set(-caseWidth / 2, glassBaseY + caseHeight + topFrameHeight / 2, 0);
    caseGroup.add(topFrameLeft);
    displayCaseFrames.push(topFrameLeft);
    
    // Right top frame
    const topFrameRight = new THREE.Mesh(
        new THREE.BoxGeometry(topFrameThickness, topFrameHeight, caseDepth),
        topFrameMaterial
    );
    topFrameRight.position.set(caseWidth / 2, glassBaseY + caseHeight + topFrameHeight / 2, 0);
    caseGroup.add(topFrameRight);
    displayCaseFrames.push(topFrameRight);
    
    // Corner posts
    const frameMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a,
        roughness: 0.2,
        metalness: 0.8
    });
    
    const cornerWidth = 0.08;
    // Post FL
    const corner1 = new THREE.Mesh(
        new THREE.BoxGeometry(cornerWidth, caseHeight, cornerWidth),
        frameMaterial
    );
    corner1.position.set(-caseWidth / 2, glassBaseY + caseHeight / 2, caseDepth / 2);
    caseGroup.add(corner1);
    displayCaseFrames.push(corner1);
    
    // Post FR
    const corner2 = new THREE.Mesh(
        new THREE.BoxGeometry(cornerWidth, caseHeight, cornerWidth),
        frameMaterial
    );
    corner2.position.set(caseWidth / 2, glassBaseY + caseHeight / 2, caseDepth / 2);
    caseGroup.add(corner2);
    displayCaseFrames.push(corner2);
    
    // Post BL
    const corner3 = new THREE.Mesh(
        new THREE.BoxGeometry(cornerWidth, caseHeight, cornerWidth),
        frameMaterial
    );
    corner3.position.set(-caseWidth / 2, glassBaseY + caseHeight / 2, -caseDepth / 2);
    caseGroup.add(corner3);
    displayCaseFrames.push(corner3);
    
    // Post BR
    const corner4 = new THREE.Mesh(
        new THREE.BoxGeometry(cornerWidth, caseHeight, cornerWidth),
        frameMaterial
    );
    corner4.position.set(caseWidth / 2, glassBaseY + caseHeight / 2, -caseDepth / 2);
    caseGroup.add(corner4);
    displayCaseFrames.push(corner4);
    
    // Interior point light
    const caseLight = new THREE.PointLight(0xffffff, 0.8, 5);
    caseLight.position.set(0, glassBaseY + caseHeight / 2, 0);
    caseGroup.add(caseLight);
    displayCaseLight = caseLight; // Keep interior light ref
    
    // Case centered
    const casePosition = { x: 0, y: 0, z: 0 };
    caseGroup.position.set(casePosition.x, casePosition.y, casePosition.z);
    scene.add(caseGroup);
    
    // Inner exhibit group
    displayCaseGroup = new THREE.Group();
    // Inner origin from pedestal+base+platform
    displayCaseGroup.position.set(
        casePosition.x, 
        casePosition.y + displayCasePedestalHeight + displayCaseBaseHeight + 0.05, 
        casePosition.z
    );
    scene.add(displayCaseGroup);
    
    // Store case reference
    displayCase = caseGroup;
    
    exhibitionItems.push({ object: caseGroup, name: 'Artifact display case' });
    
    console.log('Display case created at position:', caseGroup.position);
    console.log('Display case inner group position:', displayCaseGroup.position);
    console.log('Display case pedestal height:', displayCasePedestalHeight);
    
    
    // Sync pedestal UI scale
    setTimeout(function() {
        const pedestalInput = document.getElementById('pedestal-input');
        if (pedestalInput) {
            // Scale = height / 0.1
            // e.g. 3.2→32, 10→100, 0.5→5
            const scaleValue = Math.round(displayCasePedestalHeight / 0.1);
            pedestalInput.value = scaleValue;
        }
    }, 1500); // Delay 1.5s for UI
}

// setPedestalHeight scale 5–100
function adjustDisplayCasePedestalHeight(scaleValue) {
    if (!displayCase || !displayCasePedestal) {
        console.warn('Display case or pedestal not created');
        return;
    }
    
    // Clamp scale 5–100
    scaleValue = Math.max(5, Math.min(100, scaleValue));
    
    // Map 5–100 to 0.5–10.0 height
    // height = 0.1 * scale
    // Check 5→0.5, 100→10
    const actualHeight = 0.1 * scaleValue;
    displayCasePedestalHeight = actualHeight;
    
    const caseWidth = 3;
    const caseHeight = 2.5;
    const caseDepth = 1.5;
    const glassThickness = 0.05;
    
    // Rebuild pedestal geometry
    displayCasePedestal.geometry.dispose();
    displayCasePedestal.geometry = new THREE.BoxGeometry(caseWidth + 0.2, displayCasePedestalHeight, caseDepth + 0.2);
    displayCasePedestal.position.y = displayCasePedestalHeight / 2;
    
    // Reposition base
    if (displayCaseBase) {
        displayCaseBase.position.y = displayCasePedestalHeight + displayCaseBaseHeight / 2;
    }
    
    // Reposition platform
    if (displayCasePlatform) {
        displayCasePlatform.position.y = displayCasePedestalHeight + displayCaseBaseHeight + 0.025;
    }
    
    // Reposition glass & frames
    const glassBaseY = displayCasePedestalHeight + displayCaseBaseHeight + 0.05;
    
    // Front glass pos
    if (displayCaseGlasses[0]) {
        displayCaseGlasses[0].position.y = glassBaseY + caseHeight / 2;
    }
    
    // Back glass pos
    if (displayCaseGlasses[1]) {
        displayCaseGlasses[1].position.y = glassBaseY + caseHeight / 2;
    }
    
    // Left glass pos
    if (displayCaseGlasses[2]) {
        displayCaseGlasses[2].position.y = glassBaseY + caseHeight / 2;
    }
    
    // Right glass pos
    if (displayCaseGlasses[3]) {
        displayCaseGlasses[3].position.y = glassBaseY + caseHeight / 2;
    }
    
    // Top glass pos
    if (displayCaseGlasses[4]) {
        displayCaseGlasses[4].position.y = glassBaseY + caseHeight;
    }
    
    // Top frame edges
    // Frame index order 0–3
    const topFrameHeight = 0.08; // Frame strip height
    // Frame[0] front
    if (displayCaseFrames[0]) {
        displayCaseFrames[0].position.y = glassBaseY + caseHeight + topFrameHeight / 2;
        displayCaseFrames[0].position.z = caseDepth / 2;
        displayCaseFrames[0].position.x = 0; // Lock X
    }
    // Frame[1] back
    if (displayCaseFrames[1]) {
        displayCaseFrames[1].position.y = glassBaseY + caseHeight + topFrameHeight / 2;
        displayCaseFrames[1].position.z = -caseDepth / 2;
        displayCaseFrames[1].position.x = 0; // Lock X
    }
    // Frame[2] left
    if (displayCaseFrames[2]) {
        displayCaseFrames[2].position.y = glassBaseY + caseHeight + topFrameHeight / 2;
        displayCaseFrames[2].position.x = -caseWidth / 2;
        displayCaseFrames[2].position.z = 0; // Lock Z
    }
    // Frame[3] right
    if (displayCaseFrames[3]) {
        displayCaseFrames[3].position.y = glassBaseY + caseHeight + topFrameHeight / 2;
        displayCaseFrames[3].position.x = caseWidth / 2;
        displayCaseFrames[3].position.z = 0; // Lock Z
    }
    
    // Posts indices 4–7
    // Post order FL,FR,BL,BR
    if (displayCaseFrames[4]) { // Post FL
        displayCaseFrames[4].position.y = glassBaseY + caseHeight / 2;
        displayCaseFrames[4].position.x = -caseWidth / 2;
        displayCaseFrames[4].position.z = caseDepth / 2;
    }
    if (displayCaseFrames[5]) { // Post FR
        displayCaseFrames[5].position.y = glassBaseY + caseHeight / 2;
        displayCaseFrames[5].position.x = caseWidth / 2;
        displayCaseFrames[5].position.z = caseDepth / 2;
    }
    if (displayCaseFrames[6]) { // Post BL
        displayCaseFrames[6].position.y = glassBaseY + caseHeight / 2;
        displayCaseFrames[6].position.x = -caseWidth / 2;
        displayCaseFrames[6].position.z = -caseDepth / 2;
    }
    if (displayCaseFrames[7]) { // Post BR
        displayCaseFrames[7].position.y = glassBaseY + caseHeight / 2;
        displayCaseFrames[7].position.x = caseWidth / 2;
        displayCaseFrames[7].position.z = -caseDepth / 2;
    }
    
    // Relight interior
    if (displayCaseLight) {
        displayCaseLight.position.y = glassBaseY + caseHeight / 2;
    }
    
    
    // Reposition inner group
    if (displayCaseGroup && displayCase) {
        const casePosition = displayCase.position;
        displayCaseGroup.position.set(
            casePosition.x,
            casePosition.y + displayCasePedestalHeight + displayCaseBaseHeight + 0.05,
            casePosition.z
        );
    }
    
    console.log('Display case pedestal height adjusted:', displayCasePedestalHeight);
}

// Pedestal plinth
function createPedestal() {
    const pedestal = new THREE.Group();

    // Base block
    const base = new THREE.Mesh(
        new THREE.CylinderGeometry(1.5, 1.5, 0.3, 32),
        new THREE.MeshStandardMaterial({
            color: 0x555555,
            roughness: 0.7
        })
    );
    base.position.y = 0.15;
    base.castShadow = true;
    base.receiveShadow = true;
    pedestal.add(base);

    // Column shaft
    const column = new THREE.Mesh(
        new THREE.CylinderGeometry(0.8, 1, 1.5, 32),
        new THREE.MeshStandardMaterial({
            color: 0x888888,
            roughness: 0.6,
            metalness: 0.3
        })
    );
    column.position.y = 1.05;
    column.castShadow = true;
    column.receiveShadow = true;
    pedestal.add(column);

    // Capital
    const top = new THREE.Mesh(
        new THREE.CylinderGeometry(1.2, 0.8, 0.2, 32),
        new THREE.MeshStandardMaterial({
            color: 0x666666,
            roughness: 0.5
        })
    );
    top.position.y = 2;
    top.castShadow = true;
    top.receiveShadow = true;
    pedestal.add(top);

    return pedestal;
}


// Orbit controls setup
// Pick / return
function handlePickAndPlace() {
    if (heldObject) {
        // If holding, return to case
        returnToDisplayCase();
    } else {
        // Else pick from case only
        pickObjectFromDisplayCase();
    }
}

// Pick from case without ray aim
function pickObjectFromDisplayCase() {
    if (!displayCaseGroup) return;
    
    // Check case has model
    if (currentDisplayCaseModel && displayCaseGroup.children.includes(currentDisplayCaseModel)) {
        // Save transform
        heldObject = currentDisplayCaseModel;
        heldObjectOriginalParent = heldObject.parent;
        heldObjectOriginalPosition = heldObject.position.clone();
        heldObjectOriginalRotation = heldObject.rotation.clone();
        heldObjectOriginalScale = heldObject.scale.clone();
        
        // Remove from case group
        displayCaseGroup.remove(heldObject);
        
        // Attach to held container
        heldObjectContainer.add(heldObject);
        
        // Init hold distance
        heldObjectDistance = -3.0;
        
        // Held pose in front of camera
        heldObject.position.set(0, -0.3, heldObjectDistance);
        heldObject.rotation.set(0, 0, 0);
        
        // Slightly smaller for view
        const scaleFactor = 0.8;
        heldObject.scale.multiplyScalar(scaleFactor);
        
        console.log('Picked up object from display case:', heldObject.name || 'Unnamed object');
    } else {
        console.log('No pickable object in the display case');
    }
}

// Return to case
function returnToDisplayCase() {
    if (!heldObject || !displayCaseGroup) return;
    
    // Restore scale
    heldObject.scale.copy(heldObjectOriginalScale);
    
    // Detach from hold group
    heldObjectContainer.remove(heldObject);
    
    // Re-add to case
    displayCaseGroup.add(heldObject);
    
    // Restore pose
    heldObject.position.copy(heldObjectOriginalPosition);
    heldObject.rotation.copy(heldObjectOriginalRotation);
    
    // Refresh current case model ref
    currentDisplayCaseModel = heldObject;
    currentDisplayCaseModelBaseScale = heldObject.scale.x;
    if (!currentDisplayCaseModelInitialRotation || currentDisplayCaseModelInitialRotationTarget !== currentDisplayCaseModel) {
        if (currentDisplayCaseModel.userData && currentDisplayCaseModel.userData.__initialRotation) {
            currentDisplayCaseModelInitialRotation = currentDisplayCaseModel.userData.__initialRotation.clone();
            currentDisplayCaseModelInitialRotationTarget = currentDisplayCaseModel;
        } else {
            console.warn('[Receiver] returnToDisplayCase: initial rotation cache missing, recaching current rotation');
            cacheDisplayCaseModelInitialRotation(currentDisplayCaseModel);
        }
    }
    
    // Clear held state
    heldObject = null;
    heldObjectOriginalParent = null;
    heldObjectOriginalPosition = null;
    heldObjectOriginalRotation = null;
    heldObjectOriginalScale = null;
    
    console.log('Object returned to display case');
}

function setupControls() {
    // OrbitControls like model-viewer
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    renderer.domElement.setAttribute('tabindex', '-1');
    renderer.domElement.classList.add('orbit-canvas');
    controls.target.set(0, 3, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI / 2;

    renderer.domElement.addEventListener('mousedown', function() {
        document.body.classList.add('is-grabbing');
    });
    window.addEventListener('mouseup', function() {
        document.body.classList.remove('is-grabbing');
    });

    // Wheel only on canvas
    renderer.domElement.addEventListener('wheel', function(e) {
        e.preventDefault();
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
        if (!isIframeInteractionEnabled()) return;
        if (e.code === 'Digit1' && !e.repeat) {
            e.preventDefault();
            handlePickAndPlace();
        }
        if (e.code === 'KeyR' && !e.repeat) {
            e.preventDefault();
            camera.position.set(0, 6, 8);
            if (controls) controls.target.set(0, 3, 0);
        }
    });
}

// Resize handler
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    if (!container) return;
    let w = container.clientWidth;
    let h = container.clientHeight;
    if (w <= 0 || h <= 0) {
        w = window.innerWidth || 1;
        h = window.innerHeight || 1;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

// animate()
function animate() {
    requestAnimationFrame(animate);

    if (controls) controls.update();

    // Sync held container to camera
    if (heldObjectContainer && heldObject) {
        // Match camera transform
        heldObjectContainer.position.copy(camera.position);
        heldObjectContainer.rotation.copy(camera.rotation);
    }

    // Only case model spins

    const rotationTarget = getRotationTarget();

    // Horizontal spin
    if (displayCaseRotationDirection !== 0 && rotationTarget) {
        rotationTarget.rotation.y += displayCaseRotationSpeed * displayCaseRotationDirection;
    }
    
    // Vertical tilt
    if (displayCaseVerticalRotationDirection !== 0 && rotationTarget) {
        rotationTarget.rotation.x += displayCaseRotationSpeed * displayCaseVerticalRotationDirection;
    }

    renderer.render(scene, camera);
}

// loadGLB from URL
function loadGLBModel(filePath, position = { x: 0, y: 0, z: 0 }, scale = 1, rotation = { x: 0, y: 0, z: 0 }) {
    if (!gltfLoader) {
        console.error('GLTFLoader not initialized; cannot load GLB');
        return Promise.reject('GLTFLoader not initialized');
    }

    return new Promise((resolve, reject) => {
        gltfLoader.load(
            filePath,
            // onLoad
            function(gltf) {
                const model = gltf.scene;
                
                // Set position
                model.position.set(position.x, position.y, position.z);
                
                // Set scale
                model.scale.set(scale, scale, scale);
                
                // Set rotation
                model.rotation.set(rotation.x, rotation.y, rotation.z);
                
                // Shadows on
                model.traverse(function(child) {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                // Add to scene
                scene.add(model);
                
                // Register pickable
                pickableObjects.push(model);
                
                // Register exhibit
                const itemName = `GLB_Model_${exhibitionItems.length + 1}`;
                exhibitionItems.push({ object: model, name: itemName });
                
                console.log('GLB model loaded:', filePath);
                resolve(model);
            },
            // onProgress
            function(xhr) {
                if (xhr.lengthComputable) {
                    const percentComplete = (xhr.loaded / xhr.total) * 100;
                    console.log('Load progress: ' + Math.round(percentComplete) + '%');
                }
            },
            // onError
            function(error) {
                console.error('GLB model failed to load:', error);
                reject(error);
            }
        );
    });
}

// Import GLB to scene (replace)
function loadGLBFromFile(file, customScale = null) {
    if (!gltfLoader) {
        alert('GLTFLoader not initialized; cannot load GLB');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.glb')) {
        alert('Please choose a .glb file');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        const arrayBuffer = e.target.result;
        
        gltfLoader.parse(
            arrayBuffer,
            '',
            function(gltf) {
                // Remove prior import
                if (currentImportedModel) {
                    // Remove from scene
                    scene.remove(currentImportedModel);
                    
                    // Remove from items
                    const index = exhibitionItems.findIndex(item => item.object === currentImportedModel);
                    if (index !== -1) {
                        exhibitionItems.splice(index, 1);
                    }
                    
                    // Dispose GPU resources
                    currentImportedModel.traverse(function(child) {
                        if (child.isMesh) {
                            if (child.geometry) {
                                child.geometry.dispose();
                            }
                            if (child.material) {
                                if (Array.isArray(child.material)) {
                                    child.material.forEach(material => material.dispose());
                                } else {
                                    child.material.dispose();
                                }
                            }
                        }
                    });
                    
                    console.log('Removed previously imported model');
                }
                
                const model = gltf.scene;
                
                // BBox for auto scale
                let box = new THREE.Box3().setFromObject(model);
                let size = box.getSize(new THREE.Vector3());
                let maxDim = Math.max(size.x, size.y, size.z);
                
                // Base scale from size
                const baseScale = maxDim > 0 ? 2 / maxDim : 1; // Fit max dim to ~2 units
                
                // Use custom UI scale if provided; else baseline 100 (maps to former 250)
                let scaleValue;
                let actualScale;
                if (customScale !== null && customScale > 0) {
                    // customScale is the new UI scale (100 = baseline)
                    scaleValue = customScale;
                    actualScale = (scaleValue / 100) * baseScale * 2.5;
                } else {
                    // Default baseline 100
                    scaleValue = 100;
                    actualScale = baseScale * 2.5;
                }
                
                model.scale.set(actualScale, actualScale, actualScale);
                
                // Align model bottom to floor
                box = new THREE.Box3().setFromObject(model);
                const minY = box.min.y;
                model.position.y = -minY;
                
                // Place in front of camera
                model.position.set(0, 0, 5);
                
                // Shadows on
                model.traverse(function(child) {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                // Add to scene
                scene.add(model);
                
                // Store imported model ref and base scale
                currentImportedModel = model;
                currentImportedModelBaseScale = baseScale;
                
                // Register pickable
                pickableObjects.push(model);
                
                // Register exhibit
                const itemName = `Imported_model_${file.name}`;
                exhibitionItems.push({ object: model, name: itemName });
                
                console.log('GLB model loaded (replaced):', file.name, 'scale value:', scaleValue, 'actual scale:', actualScale);
                alert('Model replaced!\nFile: ' + file.name + '\nPosition: in front of camera\nScale: ' + scaleValue);
            },
            function(error) {
                console.error('GLB parse failed:', error);
                alert('Model load failed: ' + error.message);
            }
        );
    };
    
    reader.readAsArrayBuffer(file);
}

// Apply scale to scene model (UI scale system; 100 = baseline, former 250)
function applyScaleToModel(model, scaleValue, baseScale, showAlert = false) {
    if (!model) {
        if (showAlert) {
            alert('No model to adjust');
        }
        return;
    }
    
    // UI scale: 100 matches former 250; actualScale = (scaleValue/100)*baseScale*2.5
    const actualScale = (scaleValue / 100) * baseScale * 2.5;
    
    // Keep bottom on ground while scaling
    const box = new THREE.Box3().setFromObject(model);
    const currentMinY = box.min.y;
    const currentPosition = model.position.clone();
    
    model.scale.set(actualScale, actualScale, actualScale);
    
    box.setFromObject(model);
    const newMinY = box.min.y;
    
    model.position.y = currentPosition.y - (newMinY - currentMinY);
    
    if (showAlert) {
        console.log('Model scale applied:', scaleValue, 'actual scale:', actualScale);
    }
}

// Apply scale to model inside display case (same UI scale system)
function applyScaleToDisplayCaseModel(model, scaleValue, baseScale, showAlert = false) {
    if (!model || !displayCaseGroup) {
        if (showAlert) {
            alert('No model to adjust');
        }
        return;
    }
    
    const actualScale = (scaleValue / 100) * baseScale * 2.5;
    
    // Keep bottom on platform inside case group
    const box = new THREE.Box3().setFromObject(model);
    const currentMinY = box.min.y;
    const currentPosition = model.position.clone();
    
    model.scale.set(actualScale, actualScale, actualScale);
    
    box.setFromObject(model);
    const newMinY = box.min.y;
    
    model.position.y = currentPosition.y - (newMinY - currentMinY);
    
    model.position.x = 0;
    model.position.z = 0;
    currentDisplayCaseModelScaleValue = Math.max(1, Math.min(200, scaleValue));
    
    if (showAlert) {
        console.log('Display case model scale applied:', scaleValue, 'actual scale:', actualScale);
    }
}

function getRotationTarget() {
    return currentDisplayCaseModel || null;
}

function cacheDisplayCaseModelInitialRotation(model) {
    if (!model || !model.rotation) {
        currentDisplayCaseModelInitialRotation = null;
        currentDisplayCaseModelInitialRotationTarget = null;
        return;
    }
    currentDisplayCaseModelInitialRotation = model.rotation.clone();
    currentDisplayCaseModelInitialRotationTarget = model;
    if (!model.userData) model.userData = {};
    model.userData.__initialRotation = currentDisplayCaseModelInitialRotation.clone();
}

function cacheDisplayCaseModelSource(source) {
    if (!source) return;
    currentDisplayCaseModelSourceArrayBuffer = source.arrayBuffer ? source.arrayBuffer.slice(0) : null;
    currentDisplayCaseModelSourceUrl = source.url || null;
    currentDisplayCaseModelSourceFileName = source.fileName || 'model.glb';
}

function clearDisplayCaseModelSourceCache() {
    currentDisplayCaseModelSourceArrayBuffer = null;
    currentDisplayCaseModelSourceUrl = null;
    currentDisplayCaseModelSourceFileName = 'model.glb';
}

function getCurrentDisplayCaseScaleValue() {
    return Math.max(1, Math.min(200, currentDisplayCaseModelScaleValue || 100));
}

function getCurrentPedestalScaleValue() {
    var v = Math.round(displayCasePedestalHeight / 0.1);
    return Math.max(5, Math.min(100, v));
}

// Load GLB from file input into display case (replaces prior model)
function loadGLBToDisplayCase(file, customScale = null) {
    if (!gltfLoader) {
        alert('GLTFLoader not initialized; cannot load GLB');
        return;
    }

    if (!displayCaseGroup) {
        alert('Display case not created; cannot import model');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.glb')) {
        alert('Please choose a .glb file');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        const arrayBuffer = e.target.result;
        
        gltfLoader.parse(
            arrayBuffer,
            '',
            function(gltf) {
                // Clear existing models in case
                if (displayCaseGroup.children.length > 0) {
                    const childrenToRemove = [...displayCaseGroup.children];
                    childrenToRemove.forEach(function(child) {
                        // Remove from items
                        const index = exhibitionItems.findIndex(item => item.object === child);
                        if (index !== -1) {
                            exhibitionItems.splice(index, 1);
                        }
                        
                        // Dispose GPU resources
                        child.traverse(function(mesh) {
                            if (mesh.isMesh) {
                                if (mesh.geometry) {
                                    mesh.geometry.dispose();
                                }
                                if (mesh.material) {
                                    if (Array.isArray(mesh.material)) {
                                        mesh.material.forEach(material => material.dispose());
                                    } else {
                                        mesh.material.dispose();
                                    }
                                }
                            }
                        });
                        
                        // Remove from case group
                        displayCaseGroup.remove(child);
                    });
                    
                    currentDisplayCaseModel = null;
                    currentDisplayCaseModelInitialRotation = null;
                    currentDisplayCaseModelInitialRotationTarget = null;
                    clearDisplayCaseModelSourceCache();
                    
                    console.log('Cleared all models inside the display case');
                }
                
                const model = gltf.scene;
                
                // BBox for auto scale
                let box = new THREE.Box3().setFromObject(model);
                let size = box.getSize(new THREE.Vector3());
                let maxDim = Math.max(size.x, size.y, size.z);
                
                // Case interior ~2.9 x 2.0 x 1.4; fit model (max dim <= 0.8)
                const maxSize = 0.8;
                const baseScale = maxDim > 0 ? Math.min(maxSize / maxDim, 1.0) : 0.5;
                
                let scaleValue;
                let actualScale;
                if (customScale !== null && customScale > 0) {
                    scaleValue = customScale;
                    actualScale = (scaleValue / 100) * baseScale * 2.5;
                } else {
                    scaleValue = 100;
                    actualScale = baseScale * 2.5;
                }
                
                model.scale.set(actualScale, actualScale, actualScale);
                
                box = new THREE.Box3().setFromObject(model);
                size = box.getSize(new THREE.Vector3());
                const minY = box.min.y;
                
                model.position.y = -minY;
                
                model.position.x = 0;
                model.position.z = 0;
                
                // Shadows on
                model.traverse(function(child) {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                displayCaseGroup.add(model);
                
                currentDisplayCaseModel = model;
                currentDisplayCaseModelBaseScale = baseScale;
                currentDisplayCaseModelScaleValue = scaleValue;
                cacheDisplayCaseModelInitialRotation(model);
                cacheDisplayCaseModelSource({
                    arrayBuffer: arrayBuffer,
                    fileName: file.name
                });
                
                pickableObjects.push(model);
                
                // Register exhibit
                const itemName = `Display_case_artifact_${file.name}`;
                exhibitionItems.push({ object: model, name: itemName });
                
                const scaleInput2 = document.getElementById('scale-input-2');
                if (scaleInput2) {
                    scaleInput2.value = scaleValue;
                }
                
                console.log('GLB loaded into display case (replaced):', file.name, 'scale value:', scaleValue, 'actual scale:', actualScale);
                alert('Model replaced in display case!\nFile: ' + file.name + '\nPosition: inside case\nScale: ' + scaleValue);
            },
            function(error) {
                console.error('GLB parse failed:', error);
                alert('Model load failed: ' + error.message);
            }
        );
    };
    
    reader.readAsArrayBuffer(file);
}

// Load GLB from ArrayBuffer into display case (postMessage from parent controls)
function loadGLBToDisplayCaseFromArrayBuffer(arrayBuffer, fileName, customScale) {
    if (!gltfLoader || !displayCaseGroup) return;
    if (!fileName.toLowerCase().endsWith('.glb')) return;
    gltfLoader.parse(arrayBuffer, '', function(gltf) {
        if (displayCaseGroup.children.length > 0) {
            var childrenToRemove = displayCaseGroup.children.slice();
            childrenToRemove.forEach(function(child) {
                var idx = exhibitionItems.findIndex(function(item) { return item.object === child; });
                if (idx !== -1) exhibitionItems.splice(idx, 1);
                child.traverse(function(mesh) {
                    if (mesh.isMesh) {
                        if (mesh.geometry) mesh.geometry.dispose();
                        if (mesh.material) {
                            if (Array.isArray(mesh.material)) mesh.material.forEach(function(m) { m.dispose(); });
                            else mesh.material.dispose();
                        }
                    }
                });
                displayCaseGroup.remove(child);
            });
            currentDisplayCaseModel = null;
            currentDisplayCaseModelInitialRotation = null;
            currentDisplayCaseModelInitialRotationTarget = null;
            clearDisplayCaseModelSourceCache();
        }
        var model = gltf.scene;
        var box = new THREE.Box3().setFromObject(model);
        var size = box.getSize(new THREE.Vector3());
        var maxDim = Math.max(size.x, size.y, size.z);
        var maxSize = 0.8;
        var baseScale = maxDim > 0 ? Math.min(maxSize / maxDim, 1.0) : 0.5;
        var scaleValue = (customScale !== null && customScale > 0) ? customScale : 100;
        var actualScale = (scaleValue / 100) * baseScale * 2.5;
        model.scale.set(actualScale, actualScale, actualScale);
        box = new THREE.Box3().setFromObject(model);
        size = box.getSize(new THREE.Vector3());
        var minY = box.min.y;
        model.position.y = -minY;
        model.position.x = 0;
        model.position.z = 0;
        model.traverse(function(child) {
            if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; }
        });
        displayCaseGroup.add(model);
        currentDisplayCaseModel = model;
        currentDisplayCaseModelBaseScale = baseScale;
        currentDisplayCaseModelScaleValue = scaleValue;
        cacheDisplayCaseModelInitialRotation(model);
        cacheDisplayCaseModelSource({
            arrayBuffer: arrayBuffer,
            fileName: fileName
        });
        pickableObjects.push(model);
        exhibitionItems.push({ object: model, name: 'Display_case_artifact_' + fileName });
    }, function(err) { console.error('GLB parse error:', err); });
}

// Embedded mode: listen for postMessage from parent
function setupExhibitionMessageListener() {
    window.addEventListener('message', function(e) {
        var d = e.data;
        if (!d) return;
        if (d.source !== 'vase-exhibition-controls') return;
        if (d.type === 'setInteractionActive') {
            setIframeInteractionActive(!!d.active);
            if (d.active && renderer && renderer.domElement) {
                try {
                    renderer.domElement.focus({ preventScroll: true });
                } catch (_) {
                    renderer.domElement.focus();
                }
            }
            return;
        }
        if (!displayCaseGroup) return;
        switch (d.type) {
            case 'loadGLB':
                if (d.arrayBuffer) loadGLBToDisplayCaseFromArrayBuffer(d.arrayBuffer, d.fileName || 'model.glb', d.scale);
                break;
            case 'setPedestal':
                if (typeof d.value === 'number') adjustDisplayCasePedestalHeight(Math.max(5, Math.min(100, d.value)));
                break;
            case 'setScale':
                if (currentDisplayCaseModel && typeof d.value === 'number') {
                    var v = Math.max(1, Math.min(200, d.value));
                    applyScaleToDisplayCaseModel(currentDisplayCaseModel, v, currentDisplayCaseModelBaseScale, false);
                }
                break;
            case 'rotate':
                if (d.direction === 'left') displayCaseRotationDirection = d.active ? -1 : 0;
                else if (d.direction === 'right') displayCaseRotationDirection = d.active ? 1 : 0;
                else if (d.direction === 'up') displayCaseVerticalRotationDirection = d.active ? 1 : 0;
                else if (d.direction === 'down') displayCaseVerticalRotationDirection = d.active ? -1 : 0;
                break;
            case 'setRotationSpeed':
                if (typeof d.value === 'number') {
                    displayCaseRotationSpeedScale = Math.max(1, Math.min(100, d.value));
                    displayCaseRotationSpeed = 0.001 * displayCaseRotationSpeedScale;
                }
                break;
        }
    });
}

// On load, try loading models/default_vase.glb into the display case (silent if missing)
function loadDefaultVaseInDisplayCase() {
    if (!gltfLoader || !displayCaseGroup) {
        console.warn('Default vase not loaded: gltfLoader or displayCaseGroup not ready');
        return;
    }
    var glbUrl = new URL('models/default_vase.glb', window.location.href).href;
    console.log('Default vase request URL:', glbUrl);
    gltfLoader.load(
        glbUrl,
        function(gltf) {
            if (displayCaseGroup.children.length > 0) {
                const childrenToRemove = [...displayCaseGroup.children];
                childrenToRemove.forEach(function(child) {
                    const index = exhibitionItems.findIndex(item => item.object === child);
                    if (index !== -1) exhibitionItems.splice(index, 1);
                    child.traverse(function(mesh) {
                        if (mesh.isMesh) {
                            if (mesh.geometry) mesh.geometry.dispose();
                            if (mesh.material) {
                                if (Array.isArray(mesh.material)) mesh.material.forEach(m => m.dispose());
                                else mesh.material.dispose();
                            }
                        }
                    });
                    displayCaseGroup.remove(child);
                });
                currentDisplayCaseModel = null;
                currentDisplayCaseModelInitialRotation = null;
                currentDisplayCaseModelInitialRotationTarget = null;
                clearDisplayCaseModelSourceCache();
            }
            const model = gltf.scene;
            let box = new THREE.Box3().setFromObject(model);
            let size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const maxSize = 0.8;
            const baseScale = maxDim > 0 ? Math.min(maxSize / maxDim, 1.0) : 0.5;
            const scaleValue = 100;
            const actualScale = baseScale * 2.5;
            model.scale.set(actualScale, actualScale, actualScale);
            box = new THREE.Box3().setFromObject(model);
            size = box.getSize(new THREE.Vector3());
            const minY = box.min.y;
            model.position.y = -minY;
            model.position.x = 0;
            model.position.z = 0;
            model.traverse(function(child) {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            displayCaseGroup.add(model);
            currentDisplayCaseModel = model;
            currentDisplayCaseModelBaseScale = baseScale;
            currentDisplayCaseModelScaleValue = scaleValue;
            cacheDisplayCaseModelInitialRotation(model);
            cacheDisplayCaseModelSource({
                url: glbUrl,
                fileName: 'default_vase.glb'
            });
            pickableObjects.push(model);
            exhibitionItems.push({ object: model, name: 'Display_case_artifact_default_vase' });
            console.log('Default vase loaded into display case');
        },
        function(xhr) {
            if (xhr.lengthComputable) console.log('Default vase load progress:', Math.round((xhr.loaded / xhr.total) * 100) + '%');
        },
        function(err) {
            console.warn('default_vase.glb not found or failed to load; skipped. Place a GLB in models/ as default_vase.glb.', err);
        }
    );
}

// Build file-import UI (two import panels)
function createFileInputUI() {
    // Display-case import panel
    const fileInputContainer2 = document.createElement('div');
    fileInputContainer2.id = 'file-input-container-2';
    fileInputContainer2.style.cssText = `
        background: rgba(0, 0, 0, 0.7);
        padding: 12px;
        border-radius: 6px;
        color: white;
        margin-bottom: 16px;
        border: 2px solid #00BCD4;
    `;
    
    const title2 = document.createElement('h3');
    title2.textContent = 'Import GLB into display case';
    title2.style.cssText = 'margin: 0 0 8px 0; color: #00BCD4; font-size: 14px;';
    fileInputContainer2.appendChild(title2);
    
    const fileInput2 = document.createElement('input');
    fileInput2.type = 'file';
    fileInput2.accept = '.glb';
    fileInput2.id = 'file-input-2';
    fileInput2.style.cssText = 'margin-bottom: 8px; width: 100%; padding: 4px; font-size: 12px;';
    fileInputContainer2.appendChild(fileInput2);
    
    // Scale input (display case) — UI scale (100 baseline, maps to former 250)
    const scaleLabel2 = document.createElement('label');
    scaleLabel2.textContent = 'Scale:';
    scaleLabel2.style.cssText = 'display: block; margin-bottom: 4px; font-size: 11px; color: #ccc;';
    fileInputContainer2.appendChild(scaleLabel2);
    
    // Scale control row
    const scaleContainer2 = document.createElement('div');
    scaleContainer2.style.cssText = 'display: flex; align-items: center; gap: 4px; margin-bottom: 8px;';
    
    // Minus button (long-press)
    const minusBtn2 = document.createElement('button');
    minusBtn2.textContent = '-';
    minusBtn2.id = 'minus-btn-2';
    minusBtn2.style.cssText = `
        width: 40px;
        height: 40px;
        font-size: 24px;
        font-weight: bold;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Long-press timers
    let minusInterval2 = null;
    let minusTimeout2 = null;
    
    // Decrease scale
    function decreaseScale2() {
        if (currentDisplayCaseModel) {
            const currentValue = parseFloat(scaleInput2.value) || 100;
            const newValue = Math.max(1, currentValue - 1);
            scaleInput2.value = newValue;
            applyScaleToDisplayCaseModel(currentDisplayCaseModel, newValue, currentDisplayCaseModelBaseScale, false);
        } else {
            const currentValue = parseFloat(scaleInput2.value) || 100;
            const newValue = Math.max(1, currentValue - 1);
            scaleInput2.value = newValue;
        }
    }
    
    // Mouse down
    minusBtn2.addEventListener('mousedown', function() {
        decreaseScale2(); // Fire once immediately
        // Then repeat after delay
        minusTimeout2 = setTimeout(function() {
            minusInterval2 = setInterval(decreaseScale2, 50); // Every 50 ms
        }, 300); // Continuous after 300 ms
    });
    
    // Mouse up / leave
    function stopDecrease2() {
        if (minusTimeout2) {
            clearTimeout(minusTimeout2);
            minusTimeout2 = null;
        }
        if (minusInterval2) {
            clearInterval(minusInterval2);
            minusInterval2 = null;
        }
    }
    
    minusBtn2.addEventListener('mouseup', stopDecrease2);
    minusBtn2.addEventListener('mouseleave', stopDecrease2);
    minusBtn2.addEventListener('blur', stopDecrease2);
    
    // Touch support
    minusBtn2.addEventListener('touchstart', function(e) {
        e.preventDefault();
        decreaseScale2();
        minusTimeout2 = setTimeout(function() {
            minusInterval2 = setInterval(decreaseScale2, 50);
        }, 300);
    });
    
    minusBtn2.addEventListener('touchend', stopDecrease2);
    minusBtn2.addEventListener('touchcancel', stopDecrease2);
    
    scaleContainer2.appendChild(minusBtn2);
    
    // Scale number input (hide spinners)
    const scaleInput2 = document.createElement('input');
    scaleInput2.type = 'number';
    scaleInput2.step = '1';
    scaleInput2.min = '1';
    scaleInput2.max = '500';
    scaleInput2.value = '100';
    scaleInput2.placeholder = '100';
    scaleInput2.id = 'scale-input-2';
    scaleInput2.style.cssText = `
        flex: 1;
        padding: 6px;
        text-align: center;
        font-size: 13px;
        border: 1px solid #555;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        box-sizing: border-box;
    `;
    // Hide number spinners (Chrome, Safari, Edge)
    scaleInput2.style.webkitAppearance = 'none';
    scaleInput2.style.mozAppearance = 'textfield';
    // Hide spinners (Firefox)
    scaleInput2.addEventListener('wheel', function(e) {
        e.preventDefault();
    });
    // Spinner-hiding CSS
    const style2 = document.createElement('style');
    style2.textContent = `
        #scale-input-2::-webkit-outer-spin-button,
        #scale-input-2::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        #scale-input-2[type=number] {
            -moz-appearance: textfield;
        }
    `;
    document.head.appendChild(style2);
    scaleContainer2.appendChild(scaleInput2);
    
    // Plus button (long-press)
    const plusBtn2 = document.createElement('button');
    plusBtn2.textContent = '+';
    plusBtn2.id = 'plus-btn-2';
    plusBtn2.style.cssText = `
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Long-press timers
    let plusInterval2 = null;
    let plusTimeout2 = null;
    
    // Increase scale
    function increaseScale2() {
        if (currentDisplayCaseModel) {
            const currentValue = parseFloat(scaleInput2.value) || 100;
            const newValue = Math.min(500, currentValue + 1);
            scaleInput2.value = newValue;
            applyScaleToDisplayCaseModel(currentDisplayCaseModel, newValue, currentDisplayCaseModelBaseScale, false);
        } else {
            const currentValue = parseFloat(scaleInput2.value) || 100;
            const newValue = Math.min(500, currentValue + 1);
            scaleInput2.value = newValue;
        }
    }
    
    // Mouse down
    plusBtn2.addEventListener('mousedown', function() {
        increaseScale2(); // Fire once immediately
        // Then repeat after delay
        plusTimeout2 = setTimeout(function() {
            plusInterval2 = setInterval(increaseScale2, 50); // Every 50 ms
        }, 300); // Continuous after 300 ms
    });
    
    // Mouse up / leave
    function stopIncrease2() {
        if (plusTimeout2) {
            clearTimeout(plusTimeout2);
            plusTimeout2 = null;
        }
        if (plusInterval2) {
            clearInterval(plusInterval2);
            plusInterval2 = null;
        }
    }
    
    plusBtn2.addEventListener('mouseup', stopIncrease2);
    plusBtn2.addEventListener('mouseleave', stopIncrease2);
    plusBtn2.addEventListener('blur', stopIncrease2);
    
    // Touch support
    plusBtn2.addEventListener('touchstart', function(e) {
        e.preventDefault();
        increaseScale2();
        plusTimeout2 = setTimeout(function() {
            plusInterval2 = setInterval(increaseScale2, 50);
        }, 300);
    });
    
    plusBtn2.addEventListener('touchend', stopIncrease2);
    plusBtn2.addEventListener('touchcancel', stopIncrease2);
    
    scaleContainer2.appendChild(plusBtn2);
    
    fileInputContainer2.appendChild(scaleContainer2);
    
    fileInput2.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Store file info for size display
            lastLoadedFile = {
                name: file.name,
                size: file.size,
                type: file.type
            };
            
            // On pick, use input scale if set, else null (auto / 100)
            const scaleValue = parseFloat(scaleInput2.value);
            const customScale = (scaleValue > 0) ? scaleValue : null;
            loadGLBToDisplayCase(file, customScale);
            // Clear input so same file can be re-picked
            e.target.value = '';
        }
    });
    
    // Debounce scale apply on input (500 ms)
    let scaleTimeout2 = null;
    scaleInput2.addEventListener('input', function() {
        // Clear prior timer
        if (scaleTimeout2) {
            clearTimeout(scaleTimeout2);
        }
        // Apply after 500 ms
        scaleTimeout2 = setTimeout(function() {
            if (currentDisplayCaseModel) {
                const scaleValue = parseFloat(scaleInput2.value);
                if (scaleValue > 0) {
                    applyScaleToDisplayCaseModel(currentDisplayCaseModel, scaleValue, currentDisplayCaseModelBaseScale, false);
                }
            }
        }, 500);
    });
    
    const info2 = document.createElement('p');
    info2.textContent = 'Choose a file to import, then adjust scale';
    info2.style.cssText = 'margin: 0; font-size: 11px; color: #aaa;';
    fileInputContainer2.appendChild(info2);
    
    var panelControls = document.getElementById('panel-controls');
    if (panelControls) {
        panelControls.appendChild(fileInputContainer2);
    } else {
        fileInputContainer2.style.position = 'fixed';
        fileInputContainer2.style.top = '20px';
        fileInputContainer2.style.right = '20px';
        fileInputContainer2.style.zIndex = '100';
        document.body.appendChild(fileInputContainer2);
    }
    
    // Pedestal height controls
    const pedestalControlContainer = document.createElement('div');
    pedestalControlContainer.id = 'pedestal-control-container';
    pedestalControlContainer.style.cssText = `
        background: rgba(0, 0, 0, 0.7);
        padding: 12px;
        border-radius: 6px;
        color: white;
        margin-bottom: 16px;
        border: 2px solid #FF5722;
    `;
    
    const pedestalTitle = document.createElement('h3');
    pedestalTitle.textContent = 'Display case pedestal height';
    pedestalTitle.style.cssText = 'margin: 0 0 8px 0; color: #FF5722; font-size: 14px;';
    pedestalControlContainer.appendChild(pedestalTitle);
    
    const pedestalLabel = document.createElement('label');
    pedestalLabel.textContent = 'Pedestal height:';
    pedestalLabel.style.cssText = 'display: block; margin-bottom: 4px; font-size: 11px; color: #ccc;';
    pedestalControlContainer.appendChild(pedestalLabel);
    
    // Pedestal control row
    const pedestalControl = document.createElement('div');
    pedestalControl.style.cssText = 'display: flex; align-items: center; gap: 4px; margin-bottom: 8px;';
    
    // Minus (lower pedestal)
    const minusPedestalBtn = document.createElement('button');
    minusPedestalBtn.textContent = '-';
    minusPedestalBtn.id = 'minus-pedestal-btn';
    minusPedestalBtn.style.cssText = `
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Pedestal input (scale 5–100)
    const pedestalInput = document.createElement('input');
    pedestalInput.type = 'number';
    pedestalInput.step = '1';
    pedestalInput.min = '5';
    pedestalInput.max = '100';
    pedestalInput.value = '32'; // Height 3.2 maps to scale 32
    pedestalInput.placeholder = '32';
    pedestalInput.id = 'pedestal-input';
    pedestalInput.style.cssText = `
        flex: 1;
        padding: 6px;
        text-align: center;
        font-size: 13px;
        border: 1px solid #555;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        box-sizing: border-box;
    `;
    // Hide spinners
    pedestalInput.style.webkitAppearance = 'none';
    pedestalInput.style.mozAppearance = 'textfield';
    pedestalInput.addEventListener('wheel', function(e) {
        e.preventDefault();
    });
    // Spinner-hiding CSS
    const stylePedestal = document.createElement('style');
    stylePedestal.textContent = `
        #pedestal-input::-webkit-outer-spin-button,
        #pedestal-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        #pedestal-input[type=number] {
            -moz-appearance: textfield;
        }
    `;
    document.head.appendChild(stylePedestal);
    
    // Plus (raise pedestal)
    const plusPedestalBtn = document.createElement('button');
    plusPedestalBtn.textContent = '+';
    plusPedestalBtn.id = 'plus-pedestal-btn';
    plusPedestalBtn.style.cssText = `
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Adjust pedestal (scale 5–100)
    function adjustPedestalHeight(delta) {
        const currentValue = parseFloat(pedestalInput.value) || 12;
        const newValue = Math.max(5, Math.min(100, currentValue + delta));
        pedestalInput.value = newValue;
        adjustDisplayCasePedestalHeight(newValue);
    }
    
    // Minus long-press
    let minusPedestalInterval = null;
    let minusPedestalTimeout = null;
    
    function decreasePedestalHeight() {
        adjustPedestalHeight(-1); // Scale step -1
    }
    
    minusPedestalBtn.addEventListener('mousedown', function() {
        decreasePedestalHeight();
        minusPedestalTimeout = setTimeout(function() {
            minusPedestalInterval = setInterval(decreasePedestalHeight, 50);
        }, 300);
    });
    
    function stopDecreasePedestal() {
        if (minusPedestalTimeout) {
            clearTimeout(minusPedestalTimeout);
            minusPedestalTimeout = null;
        }
        if (minusPedestalInterval) {
            clearInterval(minusPedestalInterval);
            minusPedestalInterval = null;
        }
    }
    
    minusPedestalBtn.addEventListener('mouseup', stopDecreasePedestal);
    minusPedestalBtn.addEventListener('mouseleave', stopDecreasePedestal);
    minusPedestalBtn.addEventListener('blur', stopDecreasePedestal);
    minusPedestalBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        decreasePedestalHeight();
        minusPedestalTimeout = setTimeout(function() {
            minusPedestalInterval = setInterval(decreasePedestalHeight, 50);
        }, 300);
    });
    minusPedestalBtn.addEventListener('touchend', stopDecreasePedestal);
    minusPedestalBtn.addEventListener('touchcancel', stopDecreasePedestal);
    
    // Plus long-press
    let plusPedestalInterval = null;
    let plusPedestalTimeout = null;
    
    function increasePedestalHeight() {
        adjustPedestalHeight(1); // Scale step +1
    }
    
    plusPedestalBtn.addEventListener('mousedown', function() {
        increasePedestalHeight();
        plusPedestalTimeout = setTimeout(function() {
            plusPedestalInterval = setInterval(increasePedestalHeight, 50);
        }, 300);
    });
    
    function stopIncreasePedestal() {
        if (plusPedestalTimeout) {
            clearTimeout(plusPedestalTimeout);
            plusPedestalTimeout = null;
        }
        if (plusPedestalInterval) {
            clearInterval(plusPedestalInterval);
            plusPedestalInterval = null;
        }
    }
    
    plusPedestalBtn.addEventListener('mouseup', stopIncreasePedestal);
    plusPedestalBtn.addEventListener('mouseleave', stopIncreasePedestal);
    plusPedestalBtn.addEventListener('blur', stopIncreasePedestal);
    plusPedestalBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        increasePedestalHeight();
        plusPedestalTimeout = setTimeout(function() {
            plusPedestalInterval = setInterval(increasePedestalHeight, 50);
        }, 300);
    });
    plusPedestalBtn.addEventListener('touchend', stopIncreasePedestal);
    plusPedestalBtn.addEventListener('touchcancel', stopIncreasePedestal);
    
    // Input debounce (5–100)
    let pedestalTimeout = null;
    pedestalInput.addEventListener('input', function() {
        if (pedestalTimeout) {
            clearTimeout(pedestalTimeout);
        }
        pedestalTimeout = setTimeout(function() {
            const value = parseFloat(pedestalInput.value);
            if (value >= 5 && value <= 100) {
                adjustDisplayCasePedestalHeight(value);
            }
        }, 500);
    });
    
    pedestalControl.appendChild(minusPedestalBtn);
    pedestalControl.appendChild(pedestalInput);
    pedestalControl.appendChild(plusPedestalBtn);
    pedestalControlContainer.appendChild(pedestalControl);
    
    const pedestalInfo = document.createElement('p');
    pedestalInfo.textContent = 'Display case pedestal height';
    pedestalInfo.style.cssText = 'margin: 0; font-size: 11px; color: #aaa;';
    pedestalControlContainer.appendChild(pedestalInfo);
    
    // Separator
    const separator = document.createElement('hr');
    separator.style.cssText = 'margin: 10px 0; border: none; border-top: 1px solid #555;';
    pedestalControlContainer.appendChild(separator);
    
    // In-case rotation controls
    const rotationTitle = document.createElement('h3');
    rotationTitle.textContent = 'Rotate object in display case';
    rotationTitle.style.cssText = 'margin: 0 0 8px 0; color: #FF5722; font-size: 14px;';
    pedestalControlContainer.appendChild(rotationTitle);
    
    // Rotation button row
    const rotationButtonsContainer = document.createElement('div');
    rotationButtonsContainer.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; align-items: center;';
    
    // Rotate-left button
    const leftRotationBtn = document.createElement('button');
    leftRotationBtn.textContent = '◄ Rotate left';
    leftRotationBtn.id = 'left-rotation-btn';
    leftRotationBtn.style.cssText = `
        flex: 1;
        padding: 8px;
        font-size: 12px;
        font-weight: bold;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        user-select: none;
        transition: background 0.2s;
    `;
    
    // Rotate-right button
    const rightRotationBtn = document.createElement('button');
    rightRotationBtn.textContent = 'Rotate right ►';
    rightRotationBtn.id = 'right-rotation-btn';
    rightRotationBtn.style.cssText = `
        flex: 1;
        padding: 8px;
        font-size: 12px;
        font-weight: bold;
        background: #FF9800;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        user-select: none;
        transition: background 0.2s;
    `;
    
    // Rotate-left handlers
    leftRotationBtn.addEventListener('mousedown', function() {
        displayCaseRotationDirection = -1;
        leftRotationBtn.style.background = '#1976D2';
        console.log('Rotate left started');
    });
    
    leftRotationBtn.addEventListener('mouseup', function() {
        displayCaseRotationDirection = 0;
        leftRotationBtn.style.background = '#2196F3';
        console.log('Rotation stopped');
    });
    
    leftRotationBtn.addEventListener('mouseleave', function() {
        displayCaseRotationDirection = 0;
        leftRotationBtn.style.background = '#2196F3';
        console.log('Rotation stopped');
    });
    
    leftRotationBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = -1;
        leftRotationBtn.style.background = '#1976D2';
        console.log('Rotate left started');
    });
    
    leftRotationBtn.addEventListener('touchend', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = 0;
        leftRotationBtn.style.background = '#2196F3';
        console.log('Rotation stopped');
    });
    
    leftRotationBtn.addEventListener('touchcancel', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = 0;
        leftRotationBtn.style.background = '#2196F3';
        console.log('Rotation stopped');
    });
    
    // Rotate-right handlers
    rightRotationBtn.addEventListener('mousedown', function() {
        displayCaseRotationDirection = 1;
        rightRotationBtn.style.background = '#F57C00';
        console.log('Rotate right started');
    });
    
    rightRotationBtn.addEventListener('mouseup', function() {
        displayCaseRotationDirection = 0;
        rightRotationBtn.style.background = '#FF9800';
        console.log('Rotation stopped');
    });
    
    rightRotationBtn.addEventListener('mouseleave', function() {
        displayCaseRotationDirection = 0;
        rightRotationBtn.style.background = '#FF9800';
        console.log('Rotation stopped');
    });
    
    rightRotationBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = 1;
        rightRotationBtn.style.background = '#F57C00';
        console.log('Rotate right started');
    });
    
    rightRotationBtn.addEventListener('touchend', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = 0;
        rightRotationBtn.style.background = '#FF9800';
        console.log('Rotation stopped');
    });
    
    rightRotationBtn.addEventListener('touchcancel', function(e) {
        e.preventDefault();
        displayCaseRotationDirection = 0;
        rightRotationBtn.style.background = '#FF9800';
        console.log('Rotation stopped');
    });
    
    // Tilt buttons row
    const verticalRotationContainer = document.createElement('div');
    verticalRotationContainer.style.cssText = 'margin-top: 8px;';
    
    const verticalRotationLabel = document.createElement('label');
    verticalRotationLabel.textContent = 'Tilt:';
    verticalRotationLabel.style.cssText = 'display: block; margin-bottom: 4px; font-size: 11px; color: #ccc;';
    verticalRotationContainer.appendChild(verticalRotationLabel);
    
    const verticalRotationButtons = document.createElement('div');
    verticalRotationButtons.style.cssText = 'display: flex; gap: 4px;';
    
    // Tilt-up button
    const upRotationBtn = document.createElement('button');
    upRotationBtn.textContent = '↑ Up';
    upRotationBtn.id = 'up-rotation-btn';
    upRotationBtn.style.cssText = `
        flex: 1;
        padding: 8px;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        user-select: none;
    `;
    
    // Tilt-down button
    const downRotationBtn = document.createElement('button');
    downRotationBtn.textContent = '↓ Down';
    downRotationBtn.id = 'down-rotation-btn';
    downRotationBtn.style.cssText = `
        flex: 1;
        padding: 8px;
        background: #9C27B0;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        user-select: none;
    `;
    
    // Tilt-up handlers
    upRotationBtn.addEventListener('mousedown', function() {
        displayCaseVerticalRotationDirection = 1;
        upRotationBtn.style.background = '#45a049';
        console.log('Tilt up started');
    });
    
    upRotationBtn.addEventListener('mouseup', function() {
        displayCaseVerticalRotationDirection = 0;
        upRotationBtn.style.background = '#4CAF50';
        console.log('Tilt stopped');
    });
    
    upRotationBtn.addEventListener('mouseleave', function() {
        displayCaseVerticalRotationDirection = 0;
        upRotationBtn.style.background = '#4CAF50';
        console.log('Tilt stopped');
    });
    
    upRotationBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = 1;
        upRotationBtn.style.background = '#45a049';
        console.log('Tilt up started');
    });
    
    upRotationBtn.addEventListener('touchend', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = 0;
        upRotationBtn.style.background = '#4CAF50';
        console.log('Tilt stopped');
    });
    
    upRotationBtn.addEventListener('touchcancel', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = 0;
        upRotationBtn.style.background = '#4CAF50';
        console.log('Tilt stopped');
    });
    
    // Tilt-down handlers
    downRotationBtn.addEventListener('mousedown', function() {
        displayCaseVerticalRotationDirection = -1;
        downRotationBtn.style.background = '#7b1fa2';
        console.log('Tilt down started');
    });
    
    downRotationBtn.addEventListener('mouseup', function() {
        displayCaseVerticalRotationDirection = 0;
        downRotationBtn.style.background = '#9C27B0';
        console.log('Tilt stopped');
    });
    
    downRotationBtn.addEventListener('mouseleave', function() {
        displayCaseVerticalRotationDirection = 0;
        downRotationBtn.style.background = '#9C27B0';
        console.log('Tilt stopped');
    });
    
    downRotationBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = -1;
        downRotationBtn.style.background = '#7b1fa2';
        console.log('Tilt down started');
    });
    
    downRotationBtn.addEventListener('touchend', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = 0;
        downRotationBtn.style.background = '#9C27B0';
        console.log('Tilt stopped');
    });
    
    downRotationBtn.addEventListener('touchcancel', function(e) {
        e.preventDefault();
        displayCaseVerticalRotationDirection = 0;
        downRotationBtn.style.background = '#9C27B0';
        console.log('Tilt stopped');
    });
    
    verticalRotationButtons.appendChild(upRotationBtn);
    verticalRotationButtons.appendChild(downRotationBtn);
    verticalRotationContainer.appendChild(verticalRotationButtons);
    
    // Tooltip titles
    upRotationBtn.title = 'Tilt up';
    downRotationBtn.title = 'Tilt down';
    
    rotationButtonsContainer.appendChild(leftRotationBtn);
    rotationButtonsContainer.appendChild(rightRotationBtn);
    pedestalControlContainer.appendChild(rotationButtonsContainer);
    
    // Append tilt row to panel
    pedestalControlContainer.appendChild(verticalRotationContainer);
    
    // Rotation speed (1–100; 100 = former 0.1)
    const rotationSpeedLabel = document.createElement('label');
    rotationSpeedLabel.textContent = 'Rotation speed:';
    rotationSpeedLabel.style.cssText = 'display: block; margin-bottom: 4px; font-size: 11px; color: #ccc;';
    pedestalControlContainer.appendChild(rotationSpeedLabel);
    
    const rotationSpeedControl = document.createElement('div');
    rotationSpeedControl.style.cssText = 'display: flex; align-items: center; gap: 4px; margin-bottom: 8px;';
    
    // Minus (slower)
    const minusSpeedBtn = document.createElement('button');
    minusSpeedBtn.textContent = '-';
    minusSpeedBtn.id = 'minus-speed-btn';
    minusSpeedBtn.style.cssText = `
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Speed input (1–100)
    const rotationSpeedInput = document.createElement('input');
    rotationSpeedInput.type = 'number';
    rotationSpeedInput.step = '1';
    rotationSpeedInput.min = '1';
    rotationSpeedInput.max = '100';
    rotationSpeedInput.value = displayCaseRotationSpeedScale.toString();
    rotationSpeedInput.id = 'rotation-speed-input';
    rotationSpeedInput.style.cssText = `
        flex: 1;
        padding: 6px;
        text-align: center;
        font-size: 13px;
        border: 1px solid #555;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        box-sizing: border-box;
    `;
    rotationSpeedInput.style.webkitAppearance = 'none';
    rotationSpeedInput.style.mozAppearance = 'textfield';
    rotationSpeedInput.addEventListener('wheel', function(e) {
        e.preventDefault();
    });
    
    // Hide spinner styles
    const styleRotationSpeed = document.createElement('style');
    styleRotationSpeed.textContent = `
        #rotation-speed-input::-webkit-outer-spin-button,
        #rotation-speed-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        #rotation-speed-input[type=number] {
            -moz-appearance: textfield;
        }
    `;
    document.head.appendChild(styleRotationSpeed);
    
    // Plus (faster)
    const plusSpeedBtn = document.createElement('button');
    plusSpeedBtn.textContent = '+';
    plusSpeedBtn.id = 'plus-speed-btn';
    plusSpeedBtn.style.cssText = `
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    `;
    
    // Adjust speed (1–100)
    function adjustRotationSpeed(delta) {
        const currentValue = parseFloat(rotationSpeedInput.value) || displayCaseRotationSpeedScale;
        const newScaleValue = Math.max(1, Math.min(100, currentValue + delta));
        rotationSpeedInput.value = newScaleValue.toString();
        displayCaseRotationSpeedScale = newScaleValue;
        // Map scale to speed: scale * 0.001
        displayCaseRotationSpeed = newScaleValue * 0.001;
        console.log('Rotation speed scale:', displayCaseRotationSpeedScale, 'actual speed:', displayCaseRotationSpeed);
    }
    
    // Minus long-press
    let minusSpeedInterval = null;
    let minusSpeedTimeout = null;
    
    function decreaseRotationSpeed() {
        adjustRotationSpeed(-1); // Scale step -1
    }
    
    minusSpeedBtn.addEventListener('mousedown', function() {
        decreaseRotationSpeed();
        minusSpeedTimeout = setTimeout(function() {
            minusSpeedInterval = setInterval(decreaseRotationSpeed, 50);
        }, 300);
    });
    
    function stopDecreaseSpeed() {
        if (minusSpeedTimeout) {
            clearTimeout(minusSpeedTimeout);
            minusSpeedTimeout = null;
        }
        if (minusSpeedInterval) {
            clearInterval(minusSpeedInterval);
            minusSpeedInterval = null;
        }
    }
    
    minusSpeedBtn.addEventListener('mouseup', stopDecreaseSpeed);
    minusSpeedBtn.addEventListener('mouseleave', stopDecreaseSpeed);
    minusSpeedBtn.addEventListener('blur', stopDecreaseSpeed);
    minusSpeedBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        decreaseRotationSpeed();
        minusSpeedTimeout = setTimeout(function() {
            minusSpeedInterval = setInterval(decreaseRotationSpeed, 50);
        }, 300);
    });
    minusSpeedBtn.addEventListener('touchend', stopDecreaseSpeed);
    minusSpeedBtn.addEventListener('touchcancel', stopDecreaseSpeed);
    
    // Plus long-press
    let plusSpeedInterval = null;
    let plusSpeedTimeout = null;
    
    function increaseRotationSpeed() {
        adjustRotationSpeed(1); // Scale step +1
    }
    
    plusSpeedBtn.addEventListener('mousedown', function() {
        increaseRotationSpeed();
        plusSpeedTimeout = setTimeout(function() {
            plusSpeedInterval = setInterval(increaseRotationSpeed, 50);
        }, 300);
    });
    
    function stopIncreaseSpeed() {
        if (plusSpeedTimeout) {
            clearTimeout(plusSpeedTimeout);
            plusSpeedTimeout = null;
        }
        if (plusSpeedInterval) {
            clearInterval(plusSpeedInterval);
            plusSpeedInterval = null;
        }
    }
    
    plusSpeedBtn.addEventListener('mouseup', stopIncreaseSpeed);
    plusSpeedBtn.addEventListener('mouseleave', stopIncreaseSpeed);
    plusSpeedBtn.addEventListener('blur', stopIncreaseSpeed);
    plusSpeedBtn.addEventListener('touchstart', function(e) {
        e.preventDefault();
        increaseRotationSpeed();
        plusSpeedTimeout = setTimeout(function() {
            plusSpeedInterval = setInterval(increaseRotationSpeed, 50);
        }, 300);
    });
    plusSpeedBtn.addEventListener('touchend', stopIncreaseSpeed);
    plusSpeedBtn.addEventListener('touchcancel', stopIncreaseSpeed);
    
    // Debounced input (UI scale 1–100)
    let rotationSpeedTimeout = null;
    rotationSpeedInput.addEventListener('input', function() {
        if (rotationSpeedTimeout) {
            clearTimeout(rotationSpeedTimeout);
        }
        rotationSpeedTimeout = setTimeout(function() {
            const scaleValue = parseFloat(rotationSpeedInput.value);
            if (scaleValue >= 1 && scaleValue <= 100) {
                displayCaseRotationSpeedScale = scaleValue;
                // Map scale to speed: scale * 0.001
                displayCaseRotationSpeed = scaleValue * 0.001;
                console.log('Rotation speed scale:', displayCaseRotationSpeedScale, 'actual speed:', displayCaseRotationSpeed);
            }
        }, 500);
    });
    
    rotationSpeedControl.appendChild(minusSpeedBtn);
    rotationSpeedControl.appendChild(rotationSpeedInput);
    rotationSpeedControl.appendChild(plusSpeedBtn);
    pedestalControlContainer.appendChild(rotationSpeedControl);
    
    const rotationInfo = document.createElement('p');
    rotationInfo.textContent = 'Hold left/right to rotate; release to stop';
    rotationInfo.style.cssText = 'margin: 0; font-size: 11px; color: #aaa;';
    pedestalControlContainer.appendChild(rotationInfo);
    
    var panelControlsEl = document.getElementById('panel-controls');
    if (panelControlsEl) {
        panelControlsEl.appendChild(pedestalControlContainer);
    } else {
        pedestalControlContainer.style.position = 'fixed';
        pedestalControlContainer.style.top = '200px';
        pedestalControlContainer.style.right = '20px';
        pedestalControlContainer.style.zIndex = '100';
        document.body.appendChild(pedestalControlContainer);
    }
    
    // Keyboard shortcuts
    setupKeyboardShortcuts();
}

// Keyboard shortcuts (rotation + pedestal)
// Global long-press state for pedestal keys
let keyboardPedestalInterval = null;
let keyboardPedestalTimeout = null;

function setupKeyboardShortcuts() {
    // Skip shortcuts when typing in inputs
    function isInputFocused() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.isContentEditable
        );
    }
    
    // Pedestal adjust from keyboard
    function adjustPedestalHeightByKeyboard(delta) {
        const pedestalInput = document.getElementById('pedestal-input');
        if (!pedestalInput) return;
        
        const currentValue = parseFloat(pedestalInput.value) || 32;
        const newValue = Math.max(5, Math.min(100, currentValue + delta));
        pedestalInput.value = newValue;
        adjustDisplayCasePedestalHeight(newValue);
    }
    
    // Stop pedestal key repeat
    function stopPedestalAdjustment() {
        if (keyboardPedestalTimeout) {
            clearTimeout(keyboardPedestalTimeout);
            keyboardPedestalTimeout = null;
        }
        if (keyboardPedestalInterval) {
            clearInterval(keyboardPedestalInterval);
            keyboardPedestalInterval = null;
        }
    }
    
    window.addEventListener('keydown', function(e) {
        if (!isIframeInteractionEnabled()) return;
        // Skip shortcuts when typing (except arrows and J/K)
        if (isInputFocused() && !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'j', 'J', 'k', 'K'].includes(e.key)) {
            return;
        }
        
        // Rotate left (Left Arrow)
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const leftRotationBtn = document.getElementById('left-rotation-btn');
            if (leftRotationBtn) {
                // Simulate button press
                displayCaseRotationDirection = -1;
                leftRotationBtn.style.background = '#1976D2';
                console.log('Rotate left (keyboard)');
            }
        }
        
        // Rotate right (Right Arrow)
        if (e.key === 'ArrowRight') {
            e.preventDefault();
            const rightRotationBtn = document.getElementById('right-rotation-btn');
            if (rightRotationBtn) {
                // Simulate button press
                displayCaseRotationDirection = 1;
                rightRotationBtn.style.background = '#F57C00';
                console.log('Rotate right (keyboard)');
            }
        }
        
        // Tilt down (Up Arrow — swapped mapping)
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            displayCaseVerticalRotationDirection = -1;
            const downRotationBtn = document.getElementById('down-rotation-btn');
            if (downRotationBtn) {
                downRotationBtn.style.background = '#7b1fa2';
            }
            console.log('Tilt down (keyboard)');
        }
        
        // Tilt up (Down Arrow — swapped mapping)
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            displayCaseVerticalRotationDirection = 1;
            const upRotationBtn = document.getElementById('up-rotation-btn');
            if (upRotationBtn) {
                upRotationBtn.style.background = '#45a049';
            }
            console.log('Tilt up (keyboard)');
        }
        
        // Raise pedestal (J)
        if (e.key === 'j' || e.key === 'J') {
            e.preventDefault();
            // Apply once immediately
            adjustPedestalHeightByKeyboard(1);
            // Clear prior timers
            stopPedestalAdjustment();
            // Start long-press repeat
            keyboardPedestalTimeout = setTimeout(function() {
                keyboardPedestalInterval = setInterval(function() {
                    adjustPedestalHeightByKeyboard(1);
                }, 50);
            }, 300);
        }
        
        // Lower pedestal (K)
        if (e.key === 'k' || e.key === 'K') {
            e.preventDefault();
            // Apply once immediately
            adjustPedestalHeightByKeyboard(-1);
            // Clear prior timers
            stopPedestalAdjustment();
            // Start long-press repeat
            keyboardPedestalTimeout = setTimeout(function() {
                keyboardPedestalInterval = setInterval(function() {
                    adjustPedestalHeightByKeyboard(-1);
                }, 50);
            }, 300);
        }
    });
    
    window.addEventListener('keyup', function(e) {
        // Skip shortcuts when typing (except arrows and J/K)
        if (isInputFocused() && !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'j', 'J', 'k', 'K'].includes(e.key)) {
            return;
        }
        
        // Release rotate left
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const leftRotationBtn = document.getElementById('left-rotation-btn');
            if (leftRotationBtn) {
                displayCaseRotationDirection = 0;
                leftRotationBtn.style.background = '#2196F3';
                console.log('Rotation stopped (keyboard)');
            }
        }
        
        // Release rotate right
        if (e.key === 'ArrowRight') {
            e.preventDefault();
            const rightRotationBtn = document.getElementById('right-rotation-btn');
            if (rightRotationBtn) {
                displayCaseRotationDirection = 0;
                rightRotationBtn.style.background = '#FF9800';
                console.log('Rotation stopped (keyboard)');
            }
        }
        
        // Release tilt (Up or Down Arrow)
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault();
            displayCaseVerticalRotationDirection = 0;
            const upRotationBtn = document.getElementById('up-rotation-btn');
            const downRotationBtn = document.getElementById('down-rotation-btn');
            // Swapped: Up → tilt down, Down → tilt up
            if (e.key === 'ArrowUp' && downRotationBtn) {
                downRotationBtn.style.background = '#9C27B0';
            }
            if (e.key === 'ArrowDown' && upRotationBtn) {
                upRotationBtn.style.background = '#4CAF50';
            }
            console.log('Tilt stopped (keyboard)');
        }
        
        // Release pedestal adjust (J or K)
        if (e.key === 'j' || e.key === 'J' || e.key === 'k' || e.key === 'K') {
            e.preventDefault();
            stopPedestalAdjustment();
        }
    });
    
    addShortcutHints();
}

function addShortcutHints() {
    const shortcuts = {
        'left-rotation-btn': '← Rotate left',
        'right-rotation-btn': '→ Rotate right',
        'up-rotation-btn': '↑ Tilt up',
        'down-rotation-btn': 'Tilt down',
        'minus-pedestal-btn': 'K: lower pedestal',
        'plus-pedestal-btn': 'J: raise pedestal'
    };
    
    Object.keys(shortcuts).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.title = shortcuts[id];
        }
    });
}

// Defer init until layout has non-zero size (iframe safety)
function tryInit() {
    var container = document.getElementById('canvas-container');
    if (container && container.clientWidth > 0 && container.clientHeight > 0) {
        init();
        return;
    }
    requestAnimationFrame(tryInit);
}
if (typeof THREE === 'undefined') {
    var el = document.getElementById('canvas-container');
    if (el) el.innerHTML = '<p style="color:#fff;padding:20px;text-align:center">Three.js failed to load. Check your network and refresh.</p>';
} else {
    requestAnimationFrame(tryInit);
}

// After load: show file-import UI only when opened standalone (not in iframe)
// When embedded, the parent page supplies controls
window.addEventListener('load', function() {
    if (window === window.top) {
        setTimeout(createFileInputUI, 1000);
    } else {
        setupExhibitionMessageListener();
    }
});

