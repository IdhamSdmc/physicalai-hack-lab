/**
 * Escena 3D fotorrealista offline (Three.js ESM + HDRI 360° + modelos CC0 Poly Haven)
 * Mesa de frutería Eroski con Zona A, cestas reales (wooden_crate_01 / plastic_crate_01),
 * Zona de Descontaminación UV y brazo robótico SO-101 con agarre sincronizado.
 *
 * API global: window.EroskiScene3D = { init(containerEl), setState(state), dispose() }
 */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/RoomEnvironment.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { RGBELoader } from 'three/addons/RGBELoader.js';
import { GLTFLoader } from 'three/addons/GLTFLoader.js';

const VALID_STATES = {
  libre: 'libre',
  idle: 'libre',
  ejecutando: 'ejecutando',
  running: 'ejecutando',
  detenido: 'detenido',
  stopped: 'detenido',
  error: 'error'
};

const HALO_COLORS = {
  libre: 0x15803d,
  ejecutando: 0x22c55e,
  detenido: 0xf59e0b,
  error: 0xdc2626
};

let currentState = 'libre';
let currentView = null;
let instance = null;

function normalizeState(rawState) {
  if (!rawState) return 'libre';
  const key = String(rawState).trim().toLowerCase();
  return VALID_STATES[key] || 'libre';
}

/**
 * Normaliza un modelo GLTF cargado para que su base quede en y=0, centrado en X/Z,
 * y escalado al tamaño objetivo en metros de escena.
 */
function normalizeGLTFModel(rootObj, targetDims, tintOptions) {
  rootObj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(rootObj);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const scaleX = targetDims.x ? targetDims.x / Math.max(size.x, 1e-4) : Infinity;
  const scaleY = targetDims.y ? targetDims.y / Math.max(size.y, 1e-4) : Infinity;
  const scaleZ = targetDims.z ? targetDims.z / Math.max(size.z, 1e-4) : Infinity;
  const uniformScale = Math.min(scaleX, scaleY, scaleZ);

  const wrapper = new THREE.Group();
  rootObj.scale.multiplyScalar(uniformScale);
  rootObj.position.set(
    -center.x * uniformScale,
    -box.min.y * uniformScale,
    -center.z * uniformScale
  );

  rootObj.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      if (tintOptions && child.material) {
        child.material = child.material.clone();
        if (tintOptions.colorHex !== undefined && child.material.color) {
          child.material.color.multiply(new THREE.Color(tintOptions.colorHex));
        }
        if (tintOptions.roughness !== undefined) {
          child.material.roughness = tintOptions.roughness;
        }
        if (tintOptions.envMapIntensity !== undefined) {
          child.material.envMapIntensity = tintOptions.envMapIntensity;
        }
      }
    }
  });

  if (tintOptions && tintOptions.squashY) {
    rootObj.scale.y *= tintOptions.squashY;
  }

  wrapper.add(rootObj);
  return wrapper;
}

/**
 * Genera cartel frontal legible para las cajas/cestas de Almacén y Desechos.
 */
function createSignTexture(title, bgColor, borderColor) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 144;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 512, 144);

  ctx.fillStyle = bgColor;
  ctx.fillRect(10, 10, 492, 124);

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 6;
  ctx.strokeRect(10, 10, 492, 124);

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 46px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(title, 256, 74);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

/**
 * Genera la textura superior de la mesa blanca con etiquetas delante de las zonas.
 */
function createTableTopTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1536;
  canvas.height = 960;
  const ctx = canvas.getContext('2d');

  // Mismo tono gris oscuro (#27272a) que el resto del mostrador de frutería
  ctx.fillStyle = '#27272a';
  ctx.fillRect(0, 0, 1536, 960);

  ctx.strokeStyle = '#3f3f46';
  ctx.lineWidth = 8;
  ctx.strokeRect(18, 18, 1500, 924);

  // Zona A (izquierda): Cesta de entrada
  ctx.fillStyle = 'rgba(20, 83, 45, 0.24)';
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 4;
  ctx.setLineDash([16, 10]);
  ctx.fillRect(80, 130, 495, 570);
  ctx.strokeRect(80, 130, 495, 570);
  ctx.setLineDash([]);

  ctx.fillStyle = '#86efac';
  ctx.font = 'bold 32px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('ZONA A · ENTRADA', 328, 756);

  // Desechos (derecha atrás)
  ctx.fillStyle = 'rgba(127, 29, 29, 0.26)';
  ctx.strokeStyle = '#f87171';
  ctx.lineWidth = 4;
  ctx.fillRect(965, 86, 480, 310);
  ctx.strokeRect(965, 86, 480, 310);

  ctx.fillStyle = '#fca5a5';
  ctx.font = 'bold 28px system-ui, -apple-system, sans-serif';
  ctx.fillText('DESECHOS (MALAS)', 1205, 434);

  // Almacén (derecha adelante)
  ctx.fillStyle = 'rgba(20, 83, 45, 0.24)';
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 4;
  ctx.fillRect(965, 486, 480, 310);
  ctx.strokeRect(965, 486, 480, 310);

  ctx.fillStyle = '#86efac';
  ctx.font = 'bold 28px system-ui, -apple-system, sans-serif';
  ctx.fillText('ALMACÉN (BUENAS)', 1205, 836);

  // Marca base robot (centro trasero, mirando de frente hacia el cliente)
  ctx.fillStyle = 'rgba(154, 52, 18, 0.24)';
  ctx.strokeStyle = '#fb923c';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(768, 248, 112, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#fdba74';
  ctx.font = 'bold 26px system-ui, -apple-system, sans-serif';
  ctx.fillText('SO-101', 768, 396);

  // Zona de Descontaminación (centro frontal: x=0, z=+0.42 -> u=768, v=628)
  ctx.fillStyle = 'rgba(12, 74, 110, 0.28)';
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 4;
  ctx.setLineDash([12, 8]);
  ctx.beginPath();
  ctx.arc(768, 628, 96, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#7dd3fc';
  ctx.font = 'bold 26px system-ui, -apple-system, sans-serif';
  ctx.fillText('DESCONTAMINACIÓN', 768, 760);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

/**
 * Construye los 4 pads físicos de la Zona A sobre la mesa.
 */
function createZoneAPad(labelNumber) {
  const padGroup = new THREE.Group();
  const ringMat = new THREE.MeshStandardMaterial({
    color: 0x16a34a,
    roughness: 0.38,
    metalness: 0.08
  });
  const innerMat = new THREE.MeshStandardMaterial({
    color: 0xdcfce7,
    roughness: 0.65
  });

  const baseDisk = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.25, 0.014, 28), innerMat);
  baseDisk.position.y = 0.007;
  baseDisk.receiveShadow = true;
  padGroup.add(baseDisk);

  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.012, 10, 32), ringMat);
  rim.rotation.x = Math.PI / 2;
  rim.position.y = 0.014;
  padGroup.add(rim);

  padGroup.userData.slot = labelNumber;
  return padGroup;
}

/**
 * Construye un placeholder ligero de manzana mientras carga el modelo GLTF real.
 */
function createApplePlaceholder(isGood) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 18, 16),
    new THREE.MeshStandardMaterial({
      color: isGood ? 0x22c55e : 0x7f1d1d,
      roughness: isGood ? 0.3 : 0.8
    })
  );
  mesh.position.y = 0.16;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * Construye una bandeja/cesta baja de frutería estilo Eroski (como en el lineal real):
 * paredes bajas abiertas, base moldeada oscura y cartel tipo pizarra en el borde.
 */
function createProduceDisplayTray({
  width = 1.24,
  depth = 0.86,
  height = 0.14,
  wallColor = 0x27272a,
  linerColor = 0x18181b,
  accentHex = 0x15803d,
  signTitle = 'APTAS VENTA',
  signSub = '1,25 €/kg',
  signBg = '#18181b',
  signBorder = '#22c55e'
}) {
  const tray = new THREE.Group();

  const wallMat = new THREE.MeshStandardMaterial({
    color: wallColor,
    roughness: 0.62,
    metalness: 0.08
  });
  const linerMat = new THREE.MeshStandardMaterial({
    color: linerColor,
    roughness: 0.78,
    metalness: 0.04
  });
  const rimMat = new THREE.MeshStandardMaterial({
    color: accentHex,
    roughness: 0.42,
    metalness: 0.06
  });

  // Base moldeada interior (como los alveolos oscuros de las cajas de frutería)
  const base = new THREE.Mesh(new THREE.BoxGeometry(width, 0.028, depth), linerMat);
  base.position.y = 0.014;
  base.receiveShadow = true;
  base.castShadow = true;
  tray.add(base);

  const wallT = 0.032;
  // Paredes frontal y trasera bajas
  const fbGeo = new THREE.BoxGeometry(width, height, wallT);
  const frontWall = new THREE.Mesh(fbGeo, wallMat);
  frontWall.position.set(0, height / 2, depth / 2 - wallT / 2);
  frontWall.castShadow = true;
  frontWall.receiveShadow = true;
  tray.add(frontWall);

  const backWall = new THREE.Mesh(fbGeo, wallMat);
  backWall.position.set(0, height / 2, -depth / 2 + wallT / 2);
  backWall.castShadow = true;
  backWall.receiveShadow = true;
  tray.add(backWall);

  // Paredes laterales bajas
  const lrGeo = new THREE.BoxGeometry(wallT, height, depth - wallT * 2);
  const leftWall = new THREE.Mesh(lrGeo, wallMat);
  leftWall.position.set(-width / 2 + wallT / 2, height / 2, 0);
  leftWall.castShadow = true;
  leftWall.receiveShadow = true;
  tray.add(leftWall);

  const rightWall = new THREE.Mesh(lrGeo, wallMat);
  rightWall.position.set(width / 2 - wallT / 2, height / 2, 0);
  rightWall.castShadow = true;
  rightWall.receiveShadow = true;
  tray.add(rightWall);

  // Borde superior de color identificador (verde/rojo/teal)
  const rimH = 0.022;
  const rimFB = new THREE.BoxGeometry(width + 0.02, rimH, wallT + 0.014);
  for (const zSign of [-1, 1]) {
    const r = new THREE.Mesh(rimFB, rimMat);
    r.position.set(0, height + rimH / 2, zSign * (depth / 2 - wallT / 2));
    tray.add(r);
  }
  const rimLR = new THREE.BoxGeometry(wallT + 0.014, rimH, depth + 0.02);
  for (const xSign of [-1, 1]) {
    const r = new THREE.Mesh(rimLR, rimMat);
    r.position.set(xSign * (width / 2 - wallT / 2), height + rimH / 2, 0);
    tray.add(r);
  }

  // Cartel tipo pizarra de frutería Eroski en el borde trasero superior (no estorba a la garra)
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 192;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = signBg;
  ctx.fillRect(0, 0, 512, 192);
  ctx.strokeStyle = signBorder;
  ctx.lineWidth = 12;
  ctx.strokeRect(8, 8, 496, 176);

  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = 'bold 46px system-ui, -apple-system, sans-serif';
  ctx.fillText(signTitle, 256, 68);

  ctx.fillStyle = signBorder;
  ctx.font = 'bold 34px system-ui, -apple-system, sans-serif';
  ctx.fillText(signSub, 256, 134);

  const signTex = new THREE.CanvasTexture(canvas);
  signTex.colorSpace = THREE.SRGBColorSpace;
  const signMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.68, 0.20, 0.024),
    [
      rimMat,
      rimMat,
      rimMat,
      rimMat,
      new THREE.MeshStandardMaterial({ map: signTex, roughness: 0.35 }),
      wallMat
    ]
  );
  signMesh.position.set(0, height + 0.10, -depth / 2 + 0.01);
  signMesh.rotation.x = 0.18;
  signMesh.castShadow = true;
  tray.add(signMesh);

  return tray;
}

/**
 * Construye la estación/vaso blanco higienizador de Descontaminación con anillo UV azul/cian.
 */
function createDecontaminationStation() {
  const station = new THREE.Group();

  const cupMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.22,
    metalness: 0.04,
    clearcoat: 0.55,
    clearcoatRoughness: 0.18
  });
  const innerMat = new THREE.MeshStandardMaterial({
    color: 0x0c4a6e,
    roughness: 0.35
  });
  const uvRingMat = new THREE.MeshStandardMaterial({
    color: 0x0284c7,
    emissive: 0x38bdf8,
    emissiveIntensity: 0.45,
    roughness: 0.2
  });

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.27, 0.04, 32), cupMat);
  base.position.y = 0.02;
  base.receiveShadow = true;
  base.castShadow = true;
  station.add(base);

  const wall = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.20, 0.26, 32, 1, true),
    new THREE.MeshPhysicalMaterial({
      color: 0xf8fafc,
      roughness: 0.24,
      clearcoat: 0.5,
      side: THREE.DoubleSide
    })
  );
  wall.position.y = 0.15;
  wall.castShadow = true;
  station.add(wall);

  const innerFloor = new THREE.Mesh(new THREE.CircleGeometry(0.195, 28), innerMat);
  innerFloor.rotation.x = -Math.PI / 2;
  innerFloor.position.y = 0.045;
  station.add(innerFloor);

  const uvRing = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.022, 14, 36), uvRingMat);
  uvRing.rotation.x = Math.PI / 2;
  uvRing.position.y = 0.28;
  station.add(uvRing);

  return { group: station, uvRingMat };
}

/**
 * Crea el cartel frontal para colocar delante de la caja GLTF de Almacén o Desechos.
 */
function createFrontSign(title, bgColor, borderColor, zOffset) {
  const signTex = createSignTexture(title, bgColor, borderColor);
  const signMat = new THREE.MeshStandardMaterial({
    map: signTex,
    roughness: 0.35
  });
  const signMesh = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.21, 0.02), signMat);
  signMesh.position.set(0, 0.16, zOffset);
  signMesh.castShadow = true;
  return signMesh;
}

/**
 * Construye el brazo robótico SO-101 con mordazas anchas (±0.30 abierto / ±0.24 cerrado).
 */
function createRobotArm() {
  const root = new THREE.Group();

  const orangeMat = new THREE.MeshPhysicalMaterial({
    color: 0xf97316,
    roughness: 0.34,
    metalness: 0.06,
    clearcoat: 0.28,
    clearcoatRoughness: 0.25
  });
  const servoMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.42,
    metalness: 0.38
  });
  const metalMat = new THREE.MeshStandardMaterial({
    color: 0xcbd5e1,
    roughness: 0.24,
    metalness: 0.85
  });
  const padMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.85
  });

  const haloGeo = new THREE.TorusGeometry(0.47, 0.024, 16, 56);
  const haloMat = new THREE.MeshStandardMaterial({
    color: HALO_COLORS.libre,
    emissive: HALO_COLORS.libre,
    emissiveIntensity: 0.55,
    roughness: 0.25
  });
  const halo = new THREE.Mesh(haloGeo, haloMat);
  halo.rotation.x = Math.PI / 2;
  halo.position.y = 0.016;
  root.add(halo);

  const basePlate = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.40, 0.075, 32), servoMat);
  basePlate.position.y = 0.038;
  basePlate.castShadow = true;
  basePlate.receiveShadow = true;
  root.add(basePlate);

  const boltGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.02, 12);
  for (let b = 0; b < 4; b++) {
    const ang = (b * Math.PI) / 2 + Math.PI / 4;
    const bolt = new THREE.Mesh(boltGeo, metalMat);
    bolt.position.set(Math.cos(ang) * 0.30, 0.08, Math.sin(ang) * 0.30);
    root.add(bolt);
  }

  const turretPivot = new THREE.Group();
  turretPivot.position.y = 0.075;
  root.add(turretPivot);

  const turretBody = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.28, 0.17, 28), orangeMat);
  turretBody.position.y = 0.085;
  turretBody.castShadow = true;
  turretPivot.add(turretBody);

  const shoulderMount = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.21, 0.23), servoMat);
  shoulderMount.position.y = 0.23;
  shoulderMount.castShadow = true;
  turretPivot.add(shoulderMount);

  const shoulderPivot = new THREE.Group();
  shoulderPivot.position.set(0, 0.29, 0);
  turretPivot.add(shoulderPivot);

  const shoulderServo = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.28, 22), servoMat);
  shoulderServo.rotation.x = Math.PI / 2;
  shoulderServo.castShadow = true;
  shoulderPivot.add(shoulderServo);

  const shoulderCap = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.30, 16), metalMat);
  shoulderCap.rotation.x = Math.PI / 2;
  shoulderPivot.add(shoulderCap);

  const upperArmLen = 1.05;
  const upperArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.072, upperArmLen - 0.14, 8, 16), orangeMat);
  upperArm.position.y = upperArmLen / 2;
  upperArm.castShadow = true;
  shoulderPivot.add(upperArm);

  const elbowPivot = new THREE.Group();
  elbowPivot.position.set(0, upperArmLen, 0);
  shoulderPivot.add(elbowPivot);

  const elbowServo = new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.095, 0.23, 22), servoMat);
  elbowServo.rotation.x = Math.PI / 2;
  elbowServo.castShadow = true;
  elbowPivot.add(elbowServo);

  const elbowCap = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.25, 16), metalMat);
  elbowCap.rotation.x = Math.PI / 2;
  elbowPivot.add(elbowCap);

  const forearmLen = 1.05;
  const forearm = new THREE.Mesh(new THREE.CapsuleGeometry(0.058, forearmLen - 0.12, 8, 16), orangeMat);
  forearm.position.y = forearmLen / 2;
  forearm.castShadow = true;
  elbowPivot.add(forearm);

  const wristPivot = new THREE.Group();
  wristPivot.position.set(0, forearmLen, 0);
  elbowPivot.add(wristPivot);

  const wristServo = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.17, 18), servoMat);
  wristServo.rotation.x = Math.PI / 2;
  wristPivot.add(wristServo);

  // Carril ancho de la pinza para ±0.30 al abrir y ±0.24 al cerrar
  const gripperBase = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.055, 0.11), servoMat);
  gripperBase.position.y = 0.075;
  gripperBase.castShadow = true;
  wristPivot.add(gripperBase);

  const jawGeo = new THREE.BoxGeometry(0.032, 0.19, 0.075);
  const gripPadGeo = new THREE.BoxGeometry(0.012, 0.12, 0.068);

  const leftJaw = new THREE.Group();
  leftJaw.position.set(-0.19, 0.195, 0);
  const leftMetal = new THREE.Mesh(jawGeo, metalMat);
  leftMetal.castShadow = true;
  leftJaw.add(leftMetal);
  const leftPad = new THREE.Mesh(gripPadGeo, padMat);
  leftPad.position.set(0.018, 0, 0);
  leftJaw.add(leftPad);
  wristPivot.add(leftJaw);

  const rightJaw = new THREE.Group();
  rightJaw.position.set(0.19, 0.195, 0);
  const rightMetal = new THREE.Mesh(jawGeo, metalMat);
  rightMetal.castShadow = true;
  rightJaw.add(rightMetal);
  const rightPad = new THREE.Mesh(gripPadGeo, padMat);
  rightPad.position.set(-0.018, 0, 0);
  rightJaw.add(rightPad);
  wristPivot.add(rightJaw);

  const gripperMount = new THREE.Group();
  gripperMount.position.set(0, 0.21, 0);
  wristPivot.add(gripperMount);

  return {
    root,
    halo,
    haloMat,
    turretPivot,
    shoulderPivot,
    elbowPivot,
    wristPivot,
    leftJaw,
    rightJaw,
    gripperMount
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(t) {
  const c = Math.max(0, Math.min(1, t));
  return c * c * (3 - 2 * c);
}

/**
 * Cinemática inversa planar 2-DOF para que la pinza baje verticalmente y centrada sobre cualquier punto (x, y, z).
 */
function solveArmPose(robotPos, targetPos, jawHalfWidth) {
  const dx = targetPos.x - robotPos.x;
  const dz = targetPos.z - robotPos.z;
  const yaw = Math.atan2(-dz, dx);
  const horiz = Math.hypot(dx, dz);

  const shoulderY = 0.365;
  const wristTargetY = targetPos.y + 0.21;
  const dy = wristTargetY - shoulderY;

  const L1 = 1.05;
  const L2 = 1.05;
  const dist = Math.min(Math.hypot(horiz, dy), L1 + L2 - 0.04);

  const cosElbow = Math.max(-1, Math.min(1, (dist * dist - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const elbowBend = -Math.acos(cosElbow);

  const alpha = Math.atan2(dy, horiz);
  const beta = Math.atan2(L2 * Math.sin(-elbowBend), L1 + L2 * Math.cos(-elbowBend));
  const shoulderAngleFromX = alpha + beta;

  const shoulder = shoulderAngleFromX - Math.PI / 2;
  const elbow = elbowBend;
  const wrist = -Math.PI / 2 - shoulderAngleFromX - elbowBend;

  return { yaw, shoulder, elbow, wrist, jaw: jawHalfWidth };
}

function createSceneInstance(containerEl) {
  const prefersReducedMotion =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  containerEl.innerHTML = '';

  const width = containerEl.clientWidth || 480;
  const height = containerEl.clientHeight || 320;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 50);
  camera.position.set(3.15, 2.85, 4.45);
  camera.lookAt(0.0, 0.25, -0.25);

  // alpha: false para que el fondo HDRI 360° de Three.js comparta perspectiva exacta con la mesa
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance'
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  containerEl.appendChild(renderer.domElement);

  // Entorno inicial inmediato con RoomEnvironment mientras carga metro_noord_1k.hdr
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  const fallbackEnvRT = pmremGenerator.fromScene(new RoomEnvironment(), 0.04);
  scene.environment = fallbackEnvRT.texture;
  scene.background = new THREE.Color(0xe7e5e4);
  scene.environmentIntensity = 0.85;
  pmremGenerator.dispose();

  let hdrTexture = null;
  const rgbeLoader = new RGBELoader();
  rgbeLoader.load('/static/env/metro_noord_1k.hdr', (tex) => {
    if (disposed) {
      tex.dispose();
      return;
    }
    tex.mapping = THREE.EquirectangularReflectionMapping;
    hdrTexture = tex;
    scene.background = hdrTexture;
    scene.environment = hdrTexture;
    scene.backgroundBlurriness = 0.08;
    scene.environmentIntensity = 0.95;
  });

  const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.4);
  keyLight.position.set(3.2, 6.2, 3.6);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 1024;
  keyLight.shadow.mapSize.height = 1024;
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 16;
  keyLight.shadow.camera.left = -3.6;
  keyLight.shadow.camera.right = 3.6;
  keyLight.shadow.camera.top = 3.0;
  keyLight.shadow.camera.bottom = -3.0;
  keyLight.shadow.bias = -0.0006;
  keyLight.shadow.radius = 3;
  scene.add(keyLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.enablePan = false;
  controls.minDistance = 2.8;
  controls.maxDistance = 8.4;
  controls.minAzimuthAngle = -Math.PI / 2;
  controls.maxAzimuthAngle = Math.PI / 2;
  controls.minPolarAngle = 0.28;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.target.set(0.0, 0.25, -0.25);
  controls.update();

  // Mostrador oscuro unificado (#27272a) igual que todo el lineal de frutería
  const tableTex = createTableTopTexture();
  const tableGroup = new THREE.Group();
  const tableMatSide = new THREE.MeshStandardMaterial({
    color: 0x27272a,
    roughness: 0.55
  });
  const tableMatTop = new THREE.MeshStandardMaterial({
    map: tableTex,
    roughness: 0.52
  });
  const tableTop = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.16, 3.25), [
    tableMatSide,
    tableMatSide,
    tableMatTop,
    tableMatSide,
    tableMatSide,
    tableMatSide
  ]);
  tableTop.position.y = -0.08;
  tableTop.receiveShadow = true;
  tableTop.castShadow = true;
  tableGroup.add(tableTop);

  const apron = new THREE.Mesh(
    new THREE.BoxGeometry(14.4, 1.60, 3.16),
    new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.72 })
  );
  apron.position.y = -0.96;
  apron.castShadow = true;
  apron.receiveShadow = true;
  tableGroup.add(apron);

  // Extensiones laterales del lineal de frutería (izquierda y derecha de la estación central)
  const wingMat = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.55 });
  for (const xWing of [-4.9, 4.9]) {
    const wingTop = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.16, 3.25), wingMat);
    wingTop.position.set(xWing, -0.08, 0);
    wingTop.receiveShadow = true;
    wingTop.castShadow = true;
    tableGroup.add(wingTop);
  }

  // Pared trasera del lineal y balda superior inclinada con suficiente separación del brazo SO-101
  const backPanel = new THREE.Mesh(
    new THREE.BoxGeometry(14.4, 1.95, 0.14),
    new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.65 })
  );
  backPanel.position.set(0, 0.90, -1.98);
  backPanel.receiveShadow = true;
  tableGroup.add(backPanel);

  const upperShelf = new THREE.Mesh(
    new THREE.BoxGeometry(14.2, 0.08, 0.95),
    new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.6 })
  );
  upperShelf.position.set(0, 0.88, -1.58);
  upperShelf.rotation.x = 0.24;
  upperShelf.receiveShadow = true;
  tableGroup.add(upperShelf);

  // Letrero superior verde FRUTERÍA · EROSKI
  const headerCanvas = document.createElement('canvas');
  headerCanvas.width = 1024;
  headerCanvas.height = 160;
  const hCtx = headerCanvas.getContext('2d');
  hCtx.fillStyle = '#65a30d';
  hCtx.fillRect(0, 0, 1024, 160);
  hCtx.fillStyle = '#15803d';
  hCtx.fillRect(0, 134, 1024, 26);
  hCtx.fillStyle = '#ffffff';
  hCtx.font = 'bold 82px system-ui, -apple-system, sans-serif';
  hCtx.textAlign = 'center';
  hCtx.textBaseline = 'middle';
  hCtx.fillText('FRUTERÍA · SUPERMERCADO', 512, 72);
  const headerTex = new THREE.CanvasTexture(headerCanvas);
  headerTex.colorSpace = THREE.SRGBColorSpace;

  const greenFascia = new THREE.Mesh(
    new THREE.BoxGeometry(14.4, 0.42, 0.16),
    [
      new THREE.MeshStandardMaterial({ color: 0x65a30d, roughness: 0.45 }),
      new THREE.MeshStandardMaterial({ color: 0x65a30d, roughness: 0.45 }),
      new THREE.MeshStandardMaterial({ color: 0x65a30d, roughness: 0.45 }),
      new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.45 }),
      new THREE.MeshStandardMaterial({ map: headerTex, roughness: 0.38 }),
      new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.45 })
    ]
  );
  greenFascia.position.set(0, 1.72, -1.88);
  tableGroup.add(greenFascia);

  // Cestas decorativas de supermercado llenando las alas laterales y la balda superior
  const decorFruitGeo = new THREE.SphereGeometry(0.088, 12, 10);
  const decorFruitConfigs = [
    { color: 0xd9f99d, title: 'MANZANA GOLDEN', price: '1,25 €/kg' },
    { color: 0xdc2626, title: 'MANZANA ROYAL', price: '1,65 €/kg' },
    { color: 0xf97316, title: 'NARANJA MESA', price: '1,95 €/kg' },
    { color: 0x84cc16, title: 'PERA CONFERENCIA', price: '2,35 €/kg' },
    { color: 0xfacc15, title: 'LIMÓN FRESCO', price: '1,49 €/kg' }
  ];
  const decorMats = decorFruitConfigs.map(
    (c) => new THREE.MeshStandardMaterial({ color: c.color, roughness: 0.34, metalness: 0.04 })
  );

  function addFilledProduceCrate(x, y, z, rotX, rotY, cfgIdx) {
    const cfg = decorFruitConfigs[cfgIdx % decorFruitConfigs.length];
    const crate = createProduceDisplayTray({
      width: 1.12,
      depth: 0.78,
      height: 0.13,
      wallColor: 0x27272a,
      linerColor: 0x18181b,
      accentHex: 0x3f3f46,
      signTitle: cfg.title,
      signSub: cfg.price,
      signBorder: '#a3e635'
    });
    const fMat = decorMats[cfgIdx % decorMats.length];
    for (let ix = -2; ix <= 2; ix++) {
      for (let iz = -1; iz <= 1; iz++) {
        const fruit = new THREE.Mesh(decorFruitGeo, fMat);
        fruit.position.set(ix * 0.19, 0.09, iz * 0.20);
        crate.add(fruit);
      }
    }
    crate.position.set(x, y, z);
    crate.rotation.set(rotX, rotY, 0);
    tableGroup.add(crate);
  }

  // Alas izquierda y derecha del mostrador (2 filas de cestas llenas de fruta)
  let cIdx = 0;
  for (const sideX of [-3.35, -4.75, -6.15, 3.35, 4.75, 6.15]) {
    addFilledProduceCrate(sideX, 0.02, -0.58, 0.10, 0, cIdx++);
    addFilledProduceCrate(sideX, 0.0, 0.56, 0.06, 0, cIdx++);
  }
  // Balda superior inclinada con cestas de fruta a lo largo de todo el lineal
  for (const upperX of [-5.6, -4.2, -2.8, -1.4, 0, 1.4, 2.8, 4.2, 5.6]) {
    addFilledProduceCrate(upperX, 0.94, -1.56, 0.24, 0, cIdx++);
  }

  scene.add(tableGroup);

  // Suelo real de supermercado bajo las patas de la mesa (y = -1.76) para que no flote
  const floorCanvas = document.createElement('canvas');
  floorCanvas.width = 512;
  floorCanvas.height = 512;
  const fCtx = floorCanvas.getContext('2d');
  fCtx.fillStyle = '#e2dfd9';
  fCtx.fillRect(0, 0, 512, 512);
  fCtx.strokeStyle = 'rgba(120, 113, 108, 0.22)';
  fCtx.lineWidth = 4;
  for (let i = 0; i <= 512; i += 128) {
    fCtx.beginPath();
    fCtx.moveTo(i, 0);
    fCtx.lineTo(i, 512);
    fCtx.stroke();
    fCtx.beginPath();
    fCtx.moveTo(0, i);
    fCtx.lineTo(512, i);
    fCtx.stroke();
  }
  const floorTex = new THREE.CanvasTexture(floorCanvas);
  floorTex.colorSpace = THREE.SRGBColorSpace;
  floorTex.wrapS = THREE.RepeatWrapping;
  floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.repeat.set(10, 10);

  const floorMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(24, 24),
    new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.62, metalness: 0.05 })
  );
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.y = -1.76;
  floorMesh.receiveShadow = true;
  scene.add(floorMesh);

  // Cesta de Entrada ocupando toda la zona sombreada verde de Zona A con margen mínimo
  const entradaTray = createProduceDisplayTray({
    width: 1.60,
    depth: 1.84,
    height: 0.14,
    wallColor: 0x3f3f46,
    linerColor: 0x18181b,
    accentHex: 0x0f766e,
    signTitle: 'CESTA ENTRADA',
    signSub: 'LOTE SIN CLASIFICAR',
    signBorder: '#2dd4bf'
  });
  entradaTray.position.set(-1.49, 0, -0.22);
  scene.add(entradaTray);

  // Almacén (derecha adelante) estilo bandeja APTAS VENTA
  const almacenTray = createProduceDisplayTray({
    width: 1.24,
    depth: 0.86,
    height: 0.14,
    wallColor: 0x27272a,
    linerColor: 0x14532d,
    accentHex: 0x15803d,
    signTitle: 'APTAS VENTA',
    signSub: 'CALIDAD A · 1,25 €/kg',
    signBorder: '#4ade80'
  });
  almacenTray.position.set(1.48, 0, 0.54);
  scene.add(almacenTray);

  // Desechos (derecha atrás) estilo bandeja MERMA / RETIRAR
  const desechosTray = createProduceDisplayTray({
    width: 1.24,
    depth: 0.86,
    height: 0.14,
    wallColor: 0x27272a,
    linerColor: 0x450a0a,
    accentHex: 0xd91b24,
    signTitle: 'MERMA / RETIRAR',
    signSub: 'NO APTAS LINEAL',
    signBorder: '#f87171'
  });
  desechosTray.position.set(1.48, 0, -0.74);
  scene.add(desechosTray);

  // Manzanas ya depositadas en el interior de las cestas abiertas
  const storedAppleGroup = new THREE.Group();
  storedAppleGroup.position.set(1.18, 0.032, 0.54);
  storedAppleGroup.rotation.y = 0.8;
  storedAppleGroup.add(createApplePlaceholder(true));
  scene.add(storedAppleGroup);

  const discardedAppleGroup = new THREE.Group();
  discardedAppleGroup.position.set(1.76, 0.032, -0.74);
  discardedAppleGroup.rotation.set(0.12, 1.4, 0.08);
  discardedAppleGroup.add(createApplePlaceholder(false));
  scene.add(discardedAppleGroup);

  // Zona A (izquierda): 8 frutas mezcladas llenando la bandeja CESTA ENTRADA con holgura para la pinza
  const zoneASlots = [
    { pos: new THREE.Vector3(-1.97, 0.032, -0.68), isGood: true },
    { pos: new THREE.Vector3(-1.49, 0.032, -0.68), isGood: false },
    { pos: new THREE.Vector3(-1.01, 0.032, -0.68), isGood: true },
    { pos: new THREE.Vector3(-1.97, 0.032, -0.24), isGood: false },
    { pos: new THREE.Vector3(-1.49, 0.032, -0.24), isGood: true },
    { pos: new THREE.Vector3(-1.01, 0.032, -0.24), isGood: false },
    { pos: new THREE.Vector3(-1.78, 0.032, 0.20), isGood: true },
    { pos: new THREE.Vector3(-1.20, 0.032, 0.20), isGood: false }
  ];

  const fruits = [];
  for (let s = 0; s < zoneASlots.length; s++) {
    const slot = zoneASlots[s];
    const appleGroup = new THREE.Group();
    appleGroup.position.copy(slot.pos);
    appleGroup.rotation.y = s * 1.3;
    appleGroup.add(createApplePlaceholder(slot.isGood));
    scene.add(appleGroup);

    fruits.push({
      mesh: appleGroup,
      homePos: slot.pos.clone(),
      homeRotY: s * 1.3,
      isGood: slot.isGood
    });
  }

  // Cargar modelo CC0 real de manzana (Poly Haven) offline con GLTFLoader
  const gltfLoader = new GLTFLoader();

  gltfLoader.load('/static/models/food_apple_01/food_apple_01_1k.gltf', (gltf) => {
    if (disposed) return;
    const appleTemplate = gltf.scene;

    const applyAppleGLTF = (targetGroup, isGood) => {
      targetGroup.clear();
      const cloned = appleTemplate.clone(true);
      const normalized = normalizeGLTFModel(
        cloned,
        { x: 0.25, y: 0.21, z: 0.25 },
        isGood
          ? { colorHex: 0xffffff, roughness: 0.22, envMapIntensity: 1.25 }
          : { colorHex: 0x4a2518, roughness: 0.92, envMapIntensity: 0.35, squashY: 0.88 }
      );
      targetGroup.add(normalized);
    };

    applyAppleGLTF(storedAppleGroup, true);
    applyAppleGLTF(discardedAppleGroup, false);
    for (const f of fruits) {
      applyAppleGLTF(f.mesh, f.isGood);
    }
  });

  // Estación de Descontaminación (centro frontal: x = 0, z = +0.42)
  const decontamPos = new THREE.Vector3(0, 0, 0.42);
  const decontam = createDecontaminationStation();
  decontam.group.position.copy(decontamPos);
  scene.add(decontam.group);

  // Brazo robótico SO-101 al fondo del mostrador, mirando de frente hacia el cliente (180° girado)
  const robotPos = new THREE.Vector3(0, 0, -0.78);
  const robot = createRobotArm();
  robot.root.position.copy(robotPos);
  scene.add(robot.root);

  const POSE_IDLE = { yaw: -Math.PI / 2, shoulder: -0.28, elbow: -1.45, wrist: -0.35, jaw: 0.19 };
  const POSE_ERROR = { yaw: -Math.PI / 2 + 0.2, shoulder: -0.15, elbow: -1.65, wrist: -0.2, jaw: 0.19 };

  // Posiciones de destino sin solapamiento dentro de APTAS VENTA y MERMA / RETIRAR
  const APTAS_SLOTS = [
    new THREE.Vector3(1.72, 0.032, 0.36),
    new THREE.Vector3(1.72, 0.032, 0.72),
    new THREE.Vector3(1.44, 0.032, 0.36),
    new THREE.Vector3(1.44, 0.032, 0.72)
  ];
  const MERMA_SLOTS = [
    new THREE.Vector3(1.20, 0.032, -0.56),
    new THREE.Vector3(1.20, 0.032, -0.92),
    new THREE.Vector3(1.48, 0.032, -0.56),
    new THREE.Vector3(1.48, 0.032, -0.92)
  ];
  let goodSlotCounter = 0;
  let badSlotCounter = 0;
  for (const f of fruits) {
    if (f.isGood) {
      f.dropPos = APTAS_SLOTS[goodSlotCounter % APTAS_SLOTS.length].clone();
      goodSlotCounter++;
    } else {
      f.dropPos = MERMA_SLOTS[badSlotCounter % MERMA_SLOTS.length].clone();
      badSlotCounter++;
    }
  }

  let currentPose = { ...POSE_IDLE };
  let cycleTime = 0;
  let activeFruitIdx = 0;
  let attachedFruit = null;
  const clock = new THREE.Clock();
  let animFrameId = null;
  let disposed = false;

  function applyPose(pose) {
    robot.turretPivot.rotation.y = pose.yaw;
    robot.shoulderPivot.rotation.z = pose.shoulder;
    robot.elbowPivot.rotation.z = pose.elbow;
    robot.wristPivot.rotation.z = pose.wrist;
    robot.leftJaw.position.x = -pose.jaw;
    robot.rightJaw.position.x = pose.jaw;
  }

  function releaseAttachedFruit() {
    if (attachedFruit) {
      scene.attach(attachedFruit.mesh);
      attachedFruit.mesh.position.copy(attachedFruit.homePos);
      attachedFruit.mesh.rotation.set(0, attachedFruit.homeRotY, 0);
      attachedFruit = null;
    }
  }

  function resetAllFruitsToZoneA() {
    releaseAttachedFruit();
    for (const f of fruits) {
      f.mesh.position.copy(f.homePos);
      f.mesh.rotation.set(0, f.homeRotY, 0);
    }
    activeFruitIdx = 0;
    cycleTime = 0;
  }

  function updateHalo(state, elapsed) {
    const targetColor = HALO_COLORS[state] || HALO_COLORS.libre;
    robot.haloMat.color.setHex(targetColor);
    robot.haloMat.emissive.setHex(targetColor);

    if (state === 'error') {
      robot.haloMat.emissiveIntensity = prefersReducedMotion
        ? 1.0
        : 0.45 + 0.55 * Math.abs(Math.sin(elapsed * 6.5));
    } else if (state === 'ejecutando') {
      robot.haloMat.emissiveIntensity = prefersReducedMotion
        ? 0.75
        : 0.6 + 0.35 * Math.sin(elapsed * 4.0);
    } else if (state === 'detenido') {
      robot.haloMat.emissiveIntensity = 0.85;
    } else {
      robot.haloMat.emissiveIntensity = 0.4;
    }
  }

  function interpolatePose(a, b, t) {
    const s = smoothstep(t);
    return {
      yaw: lerp(a.yaw, b.yaw, s),
      shoulder: lerp(a.shoulder, b.shoulder, s),
      elbow: lerp(a.elbow, b.elbow, s),
      wrist: lerp(a.wrist, b.wrist, s),
      jaw: lerp(a.jaw, b.jaw, s)
    };
  }

  function animate() {
    if (disposed) return;
    animFrameId = window.requestAnimationFrame(animate);

    const dt = Math.min(clock.getDelta(), 0.08);
    const elapsed = clock.elapsedTime;
    let uvActive = false;

    updateHalo(currentState, elapsed);

    if (currentState === 'libre') {
      releaseAttachedFruit();
      const breath = prefersReducedMotion ? 0 : Math.sin(elapsed * 1.4) * 0.025;
      currentPose.yaw = lerp(currentPose.yaw, POSE_IDLE.yaw, 0.08);
      currentPose.shoulder = lerp(currentPose.shoulder, POSE_IDLE.shoulder + breath, 0.08);
      currentPose.elbow = lerp(currentPose.elbow, POSE_IDLE.elbow - breath, 0.08);
      currentPose.wrist = lerp(currentPose.wrist, POSE_IDLE.wrist, 0.08);
      currentPose.jaw = lerp(currentPose.jaw, POSE_IDLE.jaw, 0.08);
      applyPose(currentPose);
    } else if (currentState === 'ejecutando') {
      if (prefersReducedMotion) {
        const hoverPose = solveArmPose(
          robotPos,
          new THREE.Vector3(fruits[0].homePos.x, 0.65, fruits[0].homePos.z),
          0.30
        );
        currentPose = interpolatePose(currentPose, hoverPose, 0.05);
        applyPose(currentPose);
      } else if (activeFruitIdx >= fruits.length) {
        // Todas las frutas de Zona A ya fueron clasificadas: permanecer en reposo hasta reiniciar
        currentPose = interpolatePose(currentPose, POSE_IDLE, 0.08);
        applyPose(currentPose);
      } else {
        let fruitItem = fruits[activeFruitIdx];
        let cycleDuration = fruitItem.isGood ? 5.6 : 7.8;
        cycleTime += dt;
        if (cycleTime >= cycleDuration) {
          cycleTime -= cycleDuration;
          activeFruitIdx += 1;
          if (activeFruitIdx >= fruits.length) {
            applyPose(POSE_IDLE);
            controls.update();
            renderer.render(scene, camera);
            return;
          }
          fruitItem = fruits[activeFruitIdx];
          cycleDuration = fruitItem.isGood ? 5.6 : 7.8;
        }

        const pickCenterY = fruitItem.homePos.y + 0.105;
        const dropTarget = fruitItem.dropPos;
        const dropCenterY = dropTarget.y + 0.105;

        // Poses exactas por encima de los bordes (y = 0.78) para no traspasar ninguna cesta
        const abovePick = solveArmPose(
          robotPos,
          new THREE.Vector3(fruitItem.homePos.x, 0.78, fruitItem.homePos.z),
          0.19
        );
        const downPickOpen = solveArmPose(
          robotPos,
          new THREE.Vector3(fruitItem.homePos.x, pickCenterY, fruitItem.homePos.z),
          0.19
        );
        const downPickClosed = { ...downPickOpen, jaw: 0.155 };
        const liftClosed = { ...abovePick, jaw: 0.155 };

        const aboveDropClosed = solveArmPose(
          robotPos,
          new THREE.Vector3(dropTarget.x, 0.78, dropTarget.z),
          0.155
        );
        const downDropClosed = solveArmPose(
          robotPos,
          new THREE.Vector3(dropTarget.x, dropCenterY, dropTarget.z),
          0.155
        );
        const downDropOpen = { ...downDropClosed, jaw: 0.19 };
        const liftDropOpen = { ...aboveDropClosed, jaw: 0.19 };

        // Poses de Descontaminación (pinzas cerradas a 0.12 para entrar dentro del vaso frontal sin tocar bordes)
        const aboveDecontam = solveArmPose(
          robotPos,
          new THREE.Vector3(0, 0.58, 0.42),
          0.12
        );
        const inDecontam = solveArmPose(
          robotPos,
          new THREE.Vector3(0, 0.22, 0.42),
          0.12
        );

        const t = cycleTime;
        if (t < 0.90) {
          currentPose = interpolatePose(POSE_IDLE, abovePick, t / 0.90);
        } else if (t < 1.56) {
          currentPose = interpolatePose(abovePick, downPickOpen, (t - 0.90) / 0.66);
        } else if (t < 2.02) {
          currentPose = interpolatePose(downPickOpen, downPickClosed, (t - 1.56) / 0.46);
        } else if (t < 2.58) {
          if (!attachedFruit) {
            attachedFruit = fruitItem;
            robot.gripperMount.attach(attachedFruit.mesh);
          }
          currentPose = interpolatePose(downPickClosed, liftClosed, (t - 2.02) / 0.56);
        } else if (t < 3.70) {
          currentPose = interpolatePose(liftClosed, aboveDropClosed, (t - 2.58) / 1.12);
        } else if (t < 4.26) {
          currentPose = interpolatePose(aboveDropClosed, downDropClosed, (t - 3.70) / 0.56);
        } else if (t < 4.70) {
          if (attachedFruit) {
            scene.attach(attachedFruit.mesh);
            attachedFruit.mesh.position.copy(dropTarget);
            attachedFruit.mesh.rotation.set(0, attachedFruit.homeRotY, 0);
            attachedFruit = null;
          }
          currentPose = interpolatePose(downDropClosed, downDropOpen, (t - 4.26) / 0.44);
        } else if (t < 5.15) {
          currentPose = interpolatePose(downDropOpen, liftDropOpen, (t - 4.70) / 0.45);
        } else if (fruitItem.isGood) {
          currentPose = interpolatePose(liftDropOpen, POSE_IDLE, (t - 5.15) / 0.45);
        } else if (t < 5.75) {
          currentPose = interpolatePose(liftDropOpen, aboveDecontam, (t - 5.15) / 0.60);
        } else if (t < 6.10) {
          currentPose = interpolatePose(aboveDecontam, inDecontam, (t - 5.75) / 0.35);
          uvActive = true;
        } else if (t < 7.15) {
          currentPose = { ...inDecontam };
          uvActive = true;
        } else if (t < 7.45) {
          currentPose = interpolatePose(inDecontam, aboveDecontam, (t - 7.15) / 0.30);
        } else {
          currentPose = interpolatePose(aboveDecontam, POSE_IDLE, (t - 7.45) / 0.35);
        }
        applyPose(currentPose);
      }
    } else if (currentState === 'detenido') {
      applyPose(currentPose);
    } else if (currentState === 'error') {
      releaseAttachedFruit();
      currentPose = interpolatePose(currentPose, POSE_ERROR, 0.1);
      applyPose(currentPose);
    }

    if (uvActive) {
      decontam.uvRingMat.color.setHex(0x38bdf8);
      decontam.uvRingMat.emissive.setHex(0x38bdf8);
      decontam.uvRingMat.emissiveIntensity = 1.6 + 0.9 * Math.sin(elapsed * 14.0);
    } else {
      decontam.uvRingMat.color.setHex(0x0284c7);
      decontam.uvRingMat.emissive.setHex(0x0284c7);
      decontam.uvRingMat.emissiveIntensity = 0.35;
    }

    controls.update();
    renderer.render(scene, camera);
  }

  function handleResize() {
    if (disposed || !containerEl) return;
    const w = containerEl.clientWidth;
    const h = containerEl.clientHeight;
    if (w <= 0 || h <= 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  const resizeObserver =
    typeof window.ResizeObserver === 'function' ? new window.ResizeObserver(handleResize) : null;
  if (resizeObserver) {
    resizeObserver.observe(containerEl);
  }
  window.addEventListener('resize', handleResize);

  applyPose(currentPose);
  animate();

  const VIEW_PRESETS = {
    iso: [3.15, 2.85, 4.45],
    front: [0.0, 2.05, 4.55],
    top: [0.0, 4.4, 2.2]
  };

  function setView(preset) {
    const coords = VIEW_PRESETS[preset] || VIEW_PRESETS.iso;
    camera.position.set(coords[0], coords[1], coords[2]);
    controls.target.set(0.0, 0.25, -0.25);
    controls.update();
  }

  return {
    setView,
    restartBatch: resetAllFruitsToZoneA,
    dispose() {
      disposed = true;
      if (animFrameId !== null) {
        window.cancelAnimationFrame(animFrameId);
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      fallbackEnvRT.dispose();
      if (hdrTexture) {
        hdrTexture.dispose();
      }
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          for (const m of mats) {
            if (m.map) m.map.dispose();
            if (m.normalMap) m.normalMap.dispose();
            if (m.roughnessMap) m.roughnessMap.dispose();
            m.dispose();
          }
        }
      });
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode === containerEl) {
        containerEl.removeChild(renderer.domElement);
      }
    }
  };
}

window.EroskiScene3D = {
  init(containerEl) {
    if (!containerEl) return;
    if (instance) {
      instance.dispose();
    }
    instance = createSceneInstance(containerEl);
    if (instance && currentView) {
      instance.setView(currentView);
    }
  },
  setState(state) {
    const nextState = normalizeState(state);
    if (nextState === 'ejecutando' && currentState !== 'ejecutando' && instance && instance.restartBatch) {
      instance.restartBatch();
    }
    currentState = nextState;
  },
  setView(preset) {
    currentView = preset;
    if (instance) {
      instance.setView(preset);
    }
  },
  dispose() {
    if (instance) {
      instance.dispose();
      instance = null;
    }
  }
};
