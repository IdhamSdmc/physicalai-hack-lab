import { useEffect, useRef, useState } from "react";
import {
  Play,
  OctagonAlert,
  RotateCcw,
  Sliders,
  ClipboardList,
  Activity,
  Box,
  Eye,
  ArrowDown,
  Wifi,
  WifiOff,
  CheckCircle2,
  ShieldCheck,
  ChevronDown,
  Trash2,
  Download,
  Check,
  RotateCw,
  Gauge,
  Video,
  Zap,
  Info,
  Maximize2,
  Minimize2
} from "lucide-react";

const STATE_CONFIG = {
  libre: {
    label: "LIBRE (EN ESPERA)",
    sub: "Estación preparada",
    dot: "bg-slate-400",
    pill: "bg-slate-500/10 text-slate-700 border-slate-300"
  },
  ejecutando: {
    label: "EJECUTANDO (AUTO)",
    sub: "Velocidad 100%",
    dot: "bg-emerald-500 animate-pulse",
    pill: "bg-emerald-500/15 text-emerald-800 border-emerald-500/35"
  },
  detenido: {
    label: "DETENIDO (PARADA SEGURIDAD)",
    sub: "Intervención de operador",
    dot: "bg-amber-500",
    pill: "bg-amber-500/15 text-amber-900 border-amber-500/35"
  },
  error: {
    label: "ERROR DE ESTACIÓN",
    sub: "Verificar celdas",
    dot: "bg-rose-500 animate-ping",
    pill: "bg-rose-500/15 text-rose-800 border-rose-500/35"
  }
};

function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  const mins = String(Math.floor(s / 60)).padStart(2, "0");
  const secs = String(s % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

export default function App() {
  const sceneRef = useRef(null);
  const sceneReadyRef = useRef(false);
  const prevRunningRef = useRef(false);
  const lastElapsedRef = useRef(0);
  const demoTimerRef = useRef(null);
  const demoStartedAtRef = useRef(0);
  const pollTimerRef = useRef(null);

  // Pestañas activas: "triaje" | "calibracion" | "lotes"
  const [activeTab, setActiveTab] = useState("triaje");

  const [status, setStatus] = useState({
    state: "libre",
    running: false,
    mock_mode: true,
    elapsed_seconds: 0,
    message: "Conectando con la estación SO-101...",
    logs: []
  });
  const [connected, setConnected] = useState(true);
  const [staticDemoMode, setStaticDemoMode] = useState(false);
  const [viewerMode, setViewerMode] = useState("dual");
  const [cameraView, setCameraView] = useState("iso");
  const [completedCycles, setCompletedCycles] = useState(142);
  const [lastDuration, setLastDuration] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [logOpen, setLogOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [cssFullscreen, setCssFullscreen] = useState(false);
  const cssFullscreenRef = useRef(false);
  const appRootRef = useRef(null);

  // Parámetros de Calibración
  const [armSpeed, setArmSpeed] = useState("100");
  const [vacuumActive, setVacuumActive] = useState(false);
  const [calibMessage, setCalibMessage] = useState(null);

  const demoRef = useRef({
    state: "libre",
    running: false,
    mock_mode: true,
    elapsed_seconds: 0,
    message: "Estación lista para iniciar la selección de fruta.",
    logs: [
      "[16:10:02] Sistema PLC SO-101 enlazado vía RT (latencia 12ms).",
      "[16:12:15] Calibración de ventosa y ejes 6-DOF completada OK.",
      "[16:14:30] Lote #842A cargado en bandeja de entrada A."
    ]
  });

  function showToast(text, isError = false, durationMs = 3600) {
    setFeedback({ text, isError });
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(() => setFeedback(null), durationMs);
  }

  const sceneMap = status.telemetry?.scene_map || null;
  const hudAvailable = !status.mock_mode && Boolean(status.telemetry?.hud_available);
  const showHudPanel = (viewerMode === "dual" || viewerMode === "hud") && hudAvailable;
  const show3dPanel = viewerMode === "dual" || viewerMode === "3d" || !hudAvailable;

  // Inicializar escena 3D Three.js al montar o volver a la pestaña de triaje con panel 3D visible
  useEffect(() => {
    if (activeTab !== "triaje" || !show3dPanel) return;

    const el = sceneRef.current;
    if (!el) return;
    let cancelled = false;

    async function bootScene() {
      if (!window.EroskiScene3D) {
        try {
          const sceneUrl = "/static/eroski-scene.js";
          await import(/* @vite-ignore */ sceneUrl);
        } catch (_err) {
          return;
        }
      }
      if (cancelled) return;
      if (window.EroskiScene3D && typeof window.EroskiScene3D.init === "function") {
        try {
          window.EroskiScene3D.init(el);
          sceneReadyRef.current = true;
          window.EroskiScene3D.setState(status.state);
        } catch (_err) {
          sceneReadyRef.current = false;
        }
      }
    }

    bootScene();
    return () => {
      cancelled = true;
      if (window.EroskiScene3D && typeof window.EroskiScene3D.dispose === "function") {
        window.EroskiScene3D.dispose();
        sceneReadyRef.current = false;
      }
    };
  }, [activeTab, show3dPanel]);

  // Sincronizar estado con escena 3D y métricas de turno
  useEffect(() => {
    if (sceneReadyRef.current && window.EroskiScene3D) {
      window.EroskiScene3D.setState(status.state);
    }

    if (status.running) {
      lastElapsedRef.current = status.elapsed_seconds || 0;
    } else if (prevRunningRef.current && !status.running) {
      if (status.state === "libre") {
        setCompletedCycles((c) => c + 1);
      }
      setLastDuration(lastElapsedRef.current);
    }
    prevRunningRef.current = status.running;
  }, [status.state, status.running, status.elapsed_seconds]);

  // Polling con respaldo automático a Demo Web estática (Netlify)
  useEffect(() => {
    let active = true;
    let intervalMs = 750;

    async function poll() {
      if (!active) return;

      if (demoRef.current._enabled) {
        if (demoRef.current.running) {
          demoRef.current.elapsed_seconds = Math.floor(
            (Date.now() - demoStartedAtRef.current) / 1000
          );
        }
        setConnected(true);
        setStaticDemoMode(true);
        setStatus({ ...demoRef.current, logs: [...demoRef.current.logs] });
        pollTimerRef.current = window.setTimeout(poll, intervalMs);
        return;
      }

      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        if (res.status === 404) {
          demoRef.current._enabled = true;
          setStaticDemoMode(true);
          setConnected(true);
          setStatus({ ...demoRef.current, logs: [...demoRef.current.logs] });
          pollTimerRef.current = window.setTimeout(poll, intervalMs);
          return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!active) return;
        setConnected(true);
        setStatus(data);
        if (typeof data.poll_interval_ms === "number" && data.poll_interval_ms >= 250) {
          intervalMs = data.poll_interval_ms;
        }
      } catch (_err) {
        if (!active) return;
        setConnected(false);
      } finally {
        if (active && !demoRef.current._enabled) {
          pollTimerRef.current = window.setTimeout(poll, intervalMs);
        }
      }
    }

    poll();
    return () => {
      active = false;
      window.clearTimeout(pollTimerRef.current);
      window.clearTimeout(demoTimerRef.current);
    };
  }, []);

  // Sincronizar estado nativo de pantalla completa (incluyendo Safari / iPadOS) y tecla Esc
  useEffect(() => {
    function syncFullscreen() {
      const nativeFs = Boolean(
        document.fullscreenElement || document.webkitFullscreenElement
      );
      if (nativeFs) {
        cssFullscreenRef.current = false;
        setCssFullscreen(false);
        setIsFullscreen(true);
      } else if (!cssFullscreenRef.current) {
        setIsFullscreen(false);
      }
      window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
    }

    function onKeyDown(ev) {
      if (ev.key === "Escape" && cssFullscreenRef.current) {
        cssFullscreenRef.current = false;
        setCssFullscreen(false);
        setIsFullscreen(false);
        window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
      }
    }

    document.addEventListener("fullscreenchange", syncFullscreen);
    document.addEventListener("webkitfullscreenchange", syncFullscreen);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("fullscreenchange", syncFullscreen);
      document.removeEventListener("webkitfullscreenchange", syncFullscreen);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  function activateCssFullscreen(showIosNotice = false) {
    cssFullscreenRef.current = true;
    setCssFullscreen(true);
    setIsFullscreen(true);
    if (showIosNotice) {
      const isAppleMobile =
        /iPad|iPhone|iPod/.test(navigator.userAgent) ||
        (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
      if (isAppleMobile) {
        showToast(
          "Modo inmersivo activado. En iPhone o Chrome/Brave iOS Apple bloquea el fullscreen nativo; usa Safari → Compartir → Añadir a inicio para ocultar las barras.",
          false,
          6500
        );
      }
    }
    window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
  }

  function handleToggleFullscreen() {
    const nativeFs = Boolean(
      document.fullscreenElement || document.webkitFullscreenElement
    );

    if (nativeFs) {
      try {
        if (document.exitFullscreen) {
          Promise.resolve(document.exitFullscreen()).catch(() => {});
        } else if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen();
        }
      } catch (_err) {
        // Ignorar error al salir; limpiamos estado igualmente
      }
      cssFullscreenRef.current = false;
      setCssFullscreen(false);
      setIsFullscreen(false);
      window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
      return;
    }

    if (cssFullscreenRef.current) {
      cssFullscreenRef.current = false;
      setCssFullscreen(false);
      setIsFullscreen(false);
      window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
      return;
    }

    // En iPadOS Safari debe invocarse síncronamente dentro del clic sin await previo.
    const rootEl = appRootRef.current || document.documentElement;
    const fsSupported = Boolean(
      document.fullscreenEnabled || document.webkitFullscreenEnabled
    );
    const reqFs =
      (rootEl && (rootEl.requestFullscreen || rootEl.webkitRequestFullscreen)) ||
      document.documentElement.requestFullscreen ||
      document.documentElement.webkitRequestFullscreen;
    const targetForCall =
      rootEl && (rootEl.requestFullscreen || rootEl.webkitRequestFullscreen)
        ? rootEl
        : document.documentElement;

    if (fsSupported && typeof reqFs === "function") {
      try {
        let settled = false;
        const verifyOrFallback = () => {
          const enteredNative = Boolean(
            document.fullscreenElement || document.webkitFullscreenElement
          );
          if (enteredNative) {
            cssFullscreenRef.current = false;
            setCssFullscreen(false);
            setIsFullscreen(true);
            window.setTimeout(() => window.dispatchEvent(new Event("resize")), 120);
          } else {
            activateCssFullscreen(true);
          }
        };

        const res = reqFs.call(targetForCall);
        if (res && typeof res.then === "function") {
          const timer = window.setTimeout(() => {
            if (!settled) {
              settled = true;
              verifyOrFallback();
            }
          }, 300);

          res
            .then(() => {
              if (!settled) {
                settled = true;
                window.clearTimeout(timer);
                verifyOrFallback();
              }
            })
            .catch(() => {
              if (!settled) {
                settled = true;
                window.clearTimeout(timer);
                activateCssFullscreen(true);
              }
            });
          return;
        }

        // webkitRequestFullscreen antiguo no devuelve Promise: comprobamos tras 220ms
        window.setTimeout(verifyOrFallback, 220);
        return;
      } catch (_err) {
        activateCssFullscreen(true);
        return;
      }
    }

    activateCssFullscreen(true);
  }

  function pushDemoLog(msg) {
    const hhmmss = new Date().toTimeString().slice(0, 8);
    demoRef.current.logs = [...demoRef.current.logs.slice(-24), `[${hhmmss}] ${msg}`];
  }

  async function handleStart() {
    if (staticDemoMode || demoRef.current._enabled) {
      if (demoRef.current.running) {
        showToast("Ya hay una tarea en ejecución.", true);
        return;
      }
      demoRef.current.state = "ejecutando";
      demoRef.current.running = true;
      demoRef.current.elapsed_seconds = 0;
      demoRef.current.message = "Separación de manzana y descontaminación UV en marcha.";
      demoStartedAtRef.current = Date.now();
      pushDemoLog("Ciclo de triaje continuo iniciado en modo Demo.");
      setStatus({ ...demoRef.current, logs: [...demoRef.current.logs] });
      showToast("Selección iniciada.");

      window.clearTimeout(demoTimerRef.current);
      demoTimerRef.current = window.setTimeout(() => {
        if (demoRef.current.running) {
          demoRef.current.state = "libre";
          demoRef.current.running = false;
          demoRef.current.message = "Ciclo completado correctamente. Pieza clasificada.";
          pushDemoLog("Ciclo completado: Manzana apta transferida a lineal.");
          setStatus({ ...demoRef.current, logs: [...demoRef.current.logs] });
        }
      }, 24000);
      return;
    }

    try {
      const res = await fetch("/api/start", { method: "POST" });
      const payload = await res.json();
      if (payload.status) setStatus(payload.status);
      showToast(payload.message || (res.ok ? "Selección iniciada." : "Orden rechazada."), !res.ok);
    } catch (_err) {
      showToast("Error de red al enviar la orden.", true);
    }
  }

  async function handleStop() {
    if (staticDemoMode || demoRef.current._enabled) {
      window.clearTimeout(demoTimerRef.current);
      demoRef.current.state = "detenido";
      demoRef.current.running = false;
      demoRef.current.message = "Parada de emergencia ejecutada por operador.";
      pushDemoLog("PARADA SEGURIDAD (STOP): Bloqueo de vacío neumático.");
      setStatus({ ...demoRef.current, logs: [...demoRef.current.logs] });
      showToast("Parada de emergencia ejecutada.");
      return;
    }

    try {
      const res = await fetch("/api/stop", { method: "POST" });
      const payload = await res.json();
      if (payload.status) setStatus(payload.status);
      showToast(payload.message || "Parada ejecutada.", !res.ok);
    } catch (_err) {
      showToast("Error de red al enviar STOP.", true);
    }
  }

  function handleCameraView(preset) {
    setCameraView(preset);
    if (window.EroskiScene3D && typeof window.EroskiScene3D.setView === "function") {
      window.EroskiScene3D.setView(preset);
    }
  }

  function handleClearLogs() {
    if (staticDemoMode || demoRef.current._enabled) {
      demoRef.current.logs = [];
    }
    setStatus((prev) => ({ ...prev, logs: [] }));
    showToast("Registro limpiado.");
  }

  function handleHoming() {
    setCalibMessage("Ejecutando homing: brazo regresando a posición de reposo...");
    setTimeout(() => {
      setCalibMessage("Homing completado. Los 6 ejes están en origen cero.");
      setTimeout(() => setCalibMessage(null), 4000);
    }, 1800);
  }

  function handleTestVacuum() {
    setVacuumActive(true);
    setCalibMessage("Test de vacío iniciado: electroválvula activada (94 kPa)...");
    setTimeout(() => {
      setVacuumActive(false);
      setCalibMessage("Test de vacío completado con éxito. Presión nominal verificada.");
      setTimeout(() => setCalibMessage(null), 3500);
    }, 2200);
  }

  function handleExportReport() {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Lote,Fruta,PiezasProcesadas,Aptas,Merma,UV_Exposicion,Fecha\n" +
      `#842A,Manzana Royal Gala,1020,978,42,45.2 mJ/cm2,${new Date().toISOString()}\n`;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `informe_lote_842A_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Informe de trazabilidad descargado.");
  }

  const stCfg = STATE_CONFIG[status.state] || STATE_CONFIG.libre;
  const progressPct = status.running
    ? Math.min(96, Math.max(8, ((status.elapsed_seconds % 10) / 10) * 100))
    : 68;

  const activePhase = !status.running
    ? 2
    : (status.elapsed_seconds % 8) < 3
    ? 1
    : (status.elapsed_seconds % 8) < 5
    ? 2
    : 3;

  return (
    <div
      ref={appRootRef}
      className={
        cssFullscreen
          ? "fixed inset-0 z-50 w-full h-dvh overflow-y-auto flex flex-col bg-slate-100 text-slate-800"
          : isFullscreen
          ? "min-h-screen w-full flex flex-col bg-slate-100 text-slate-800"
          : "min-h-screen w-full flex flex-col bg-transparent text-slate-800"
      }
    >
      
      {/* NAVBAR SUPERIOR COMPACTO Y RESPONSIVO (se minimiza en modo inmersivo CSS para maximizar espacio) */}
      {cssFullscreen ? (
        <div className="app-header-safe w-full pb-1 flex items-center justify-between gap-2 z-30">
          <div className={`inline-flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-bold border ${stCfg.pill}`}>
            <span className={`w-2 h-2 rounded-full ${stCfg.dot}`} />
            <span>{stCfg.label}</span>
            <span className="font-mono pl-1.5 border-l border-current/20">
              {formatDuration(status.elapsed_seconds)}
            </span>
          </div>
          <button
            type="button"
            onClick={handleToggleFullscreen}
            aria-pressed="true"
            aria-label="Salir de pantalla completa"
            className="touch-target min-h-[42px] px-3.5 py-1.5 rounded-xl bg-slate-900 text-white border border-slate-900 text-xs font-bold flex items-center gap-1.5 shadow-sm cursor-pointer whitespace-nowrap"
          >
            <Minimize2 className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>Salir pantalla</span>
          </button>
        </div>
      ) : (
      <header className="app-header-safe w-full pb-2 border-b border-slate-200/80 bg-white/85 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-[1680px] mx-auto w-full sm:px-3 flex items-center justify-between gap-2 sm:gap-3">
          
          {/* Identidad de Marca y Estación */}
          <div className="flex items-center space-x-2.5 sm:space-x-3 min-w-0 flex-1">
            <div
              className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center text-white font-black text-xs sm:text-sm tracking-tight shadow-sm shrink-0"
              style={{ background: "linear-gradient(145deg, #e5232c 0%, #b91c1c 100%)" }}
            >
              5.12
            </div>
            <div className="min-w-0">
              <div className="flex items-center space-x-1.5 sm:space-x-2">
                <span className="text-sm sm:text-base font-extrabold text-slate-900 tracking-tight leading-tight whitespace-nowrap">
                  SUPERMARKET 5.12
                </span>
                <span className="hidden sm:inline-flex bg-red-50 text-[#d91b24] text-[9px] sm:text-[10px] font-bold px-1.5 py-0.5 rounded border border-red-200 uppercase tracking-wide whitespace-nowrap">
                  Triage Bioseguro
                </span>
              </div>
              <p className="text-[11px] sm:text-xs text-slate-500 font-medium hidden md:block truncate">
                Celda SO-101 · Zona A → Almacén / Desechos + Vaso D
              </p>
            </div>
          </div>

          {/* Estado, conexión y botón de pantalla completa en Header */}
          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
            <div className={`hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-bold border ${stCfg.pill}`}>
              <span className={`w-2 h-2 rounded-full ${stCfg.dot}`} />
              <span className="whitespace-nowrap">{stCfg.label}</span>
              <span className="font-mono pl-1.5 border-l border-current/20">
                {formatDuration(status.elapsed_seconds)}
              </span>
            </div>

            <div
              className={`flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 rounded-full text-xs font-semibold border whitespace-nowrap ${
                connected
                  ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                  : "bg-rose-50 text-rose-800 border-rose-200"
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
              {connected ? <Wifi className="w-3.5 h-3.5 text-emerald-700" aria-hidden="true" /> : <WifiOff className="w-3.5 h-3.5 text-rose-600" aria-hidden="true" />}
              <span className="hidden md:inline">{connected ? "Enlazado" : "Sin conexión"}</span>
            </div>

            <button
              type="button"
              onClick={handleToggleFullscreen}
              aria-pressed={isFullscreen}
              aria-label={isFullscreen ? "Salir de pantalla completa" : "Poner aplicación en pantalla completa"}
              title={isFullscreen ? "Salir de pantalla completa" : "Pantalla completa"}
              className={`touch-target min-h-[40px] sm:min-h-[42px] px-2.5 sm:px-3.5 py-1.5 rounded-xl border text-xs font-bold flex items-center gap-1.5 cursor-pointer transition-all whitespace-nowrap ${
                isFullscreen
                  ? "bg-slate-900 text-white border-slate-900 shadow-sm"
                  : "bg-white/90 hover:bg-white text-slate-800 border-slate-200 shadow-xs"
              }`}
            >
              {isFullscreen ? (
                <Minimize2 className="w-4 h-4 shrink-0" aria-hidden="true" />
              ) : (
                <Maximize2 className="w-4 h-4 shrink-0" aria-hidden="true" />
              )}
              <span className="hidden sm:inline">
                {isFullscreen ? "Salir pantalla" : "Pantalla completa"}
              </span>
            </button>
          </div>

        </div>
      </header>
      )}

      {/* MENSAJE TOAST FLOTANTE */}
      {feedback && (
        <div className="max-w-[1680px] mx-auto w-full px-4 sm:px-6 pt-3">
          <div
            className={`rounded-2xl px-4 py-2.5 text-xs font-bold border flex items-center justify-between shadow-sm transition-all ${
              feedback.isError
                ? "bg-rose-500/15 border-rose-500/35 text-rose-900"
                : "bg-emerald-500/15 border-emerald-500/35 text-emerald-900"
            }`}
          >
            <span>{feedback.text}</span>
            <button type="button" onClick={() => setFeedback(null)} className="ml-2 text-slate-500 hover:text-slate-800">
              ✕
            </button>
          </div>
        </div>
      )}

      {/* CONTENIDO PRINCIPAL: RESPONSIVE IPAD A16 (VERTICAL Y HORIZONTAL) + DESKTOP */}
      <main className="flex-1 w-full max-w-[1680px] mx-auto px-3 sm:px-5 lg:px-6 py-3 sm:py-4 pb-28">
        
        {/* PESTAÑA 1: TRIAJE EN VIVO */}
        {activeTab === "triaje" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 xl:gap-5 items-start">
            
            {/* BLOQUE 1 (MANDO + ESTADO): Arriba en vertical/móvil; Derecha arriba en horizontal/desktop */}
            <div className="order-1 lg:col-span-5 lg:col-start-8 lg:row-start-1 flex flex-col gap-3 min-w-0">
              {/* 1. CONTROLES PRINCIPALES (SIEMPRE ARRIBA Y ACCESIBLES SIN SCROLL) */}
              <section className="controls-container liquid-panel rounded-3xl p-3 sm:p-3.5" aria-label="Controles principales">
                <div className="controls-grid">
                  <button
                    type="button"
                    disabled={status.running || !connected}
                    onClick={handleStart}
                    className="touch-target min-h-[64px] h-auto rounded-2xl bg-gradient-to-r from-emerald-600 to-[#15803d] text-white font-bold px-3.5 py-2.5 flex items-center justify-between gap-2.5 shadow-liquid-glow disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer text-left transition-all"
                  >
                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="text-[10px] uppercase font-extrabold tracking-wider text-emerald-100 flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-300 shrink-0" />
                        <span className="truncate">{status.running ? "En curso" : "Listo"}</span>
                      </div>
                      <div className="text-sm sm:text-base font-extrabold leading-tight">
                        INICIAR TAREA
                      </div>
                      <div className="text-[11px] text-emerald-100/90 font-medium leading-snug">
                        Lanzar ciclo SO-101
                      </div>
                    </div>
                    <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center shrink-0 border border-white/30">
                      <Play className="w-5 h-5 fill-current text-white ml-0.5" aria-hidden="true" />
                    </div>
                  </button>

                  <button
                    type="button"
                    disabled={!status.running && !staticDemoMode}
                    onClick={handleStop}
                    className="touch-target min-h-[64px] h-auto rounded-2xl bg-gradient-to-r from-[#d91b24] to-red-700 text-white font-bold px-3.5 py-2.5 flex items-center justify-between gap-2.5 shadow-stop-glow disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer text-left border border-red-400/50 transition-all"
                  >
                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="text-[10px] uppercase font-extrabold tracking-wider text-red-200 truncate">
                        Emergencia
                      </div>
                      <div className="text-sm sm:text-base font-extrabold leading-tight text-white">
                        PARAR (STOP)
                      </div>
                      <div className="text-[11px] text-red-100/90 font-medium leading-snug">
                        Parada inmediata
                      </div>
                    </div>
                    <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center shrink-0 border border-white/30">
                      <OctagonAlert className="w-5 h-5 text-white" aria-hidden="true" />
                    </div>
                  </button>
                </div>
              </section>

              {/* 2. Tarjeta de Estado Operativo */}
              <section className="liquid-panel rounded-3xl p-3.5 sm:p-4" aria-live="polite">
                <div className="flex items-center justify-between gap-3 border-b border-slate-200/70 pb-2.5">
                  <div className="flex items-center space-x-2.5 min-w-0">
                    <span className="relative flex h-3.5 w-3.5 shrink-0">
                      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${stCfg.dot}`} />
                      <span className={`relative inline-flex rounded-full h-3.5 w-3.5 ${stCfg.dot}`} />
                    </span>
                    <div className="min-w-0">
                      <span className="text-[10px] uppercase tracking-wider font-extrabold text-slate-500 block">
                        Estado de la Estación
                      </span>
                      <div className="text-sm sm:text-base font-extrabold text-slate-900 tracking-tight flex flex-wrap items-baseline gap-x-1.5">
                        <span>{stCfg.label}</span>
                        <span className="text-xs font-normal text-slate-500">· {stCfg.sub}</span>
                      </div>
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">
                      Tiempo
                    </span>
                    <div className="font-mono text-lg sm:text-xl font-bold text-slate-900 leading-none mt-0.5">
                      {formatDuration(status.elapsed_seconds)}
                    </div>
                  </div>
                </div>

                {/* Progreso del Lote o Telemetría Real del Supervisor */}
                {sceneMap ? (
                  <div className="mt-2.5 space-y-2">
                    <div className="flex items-center justify-between text-xs font-medium gap-2">
                      <span className="text-slate-700 font-semibold">
                        Mapa Escena OpenCV · Clean-First
                      </span>
                      <span className="font-mono text-[11px] font-bold text-slate-700 bg-slate-100 px-2 py-0.5 rounded">
                        {sceneMap.objects ? `${sceneMap.objects.length} objetos` : "0 objetos"}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-1.5 text-xs">
                      <div className="bg-emerald-50/90 border border-emerald-200 rounded-xl px-2.5 py-1.5 flex items-center justify-between">
                        <span className="text-emerald-900 font-semibold">ZONA A · Sanos (S)</span>
                        <strong className="font-mono text-sm text-emerald-800">
                          {sceneMap.counts?.verdes_en_A ?? 0}
                        </strong>
                      </div>
                      <div className="bg-rose-50/90 border border-rose-200 rounded-xl px-2.5 py-1.5 flex items-center justify-between">
                        <span className="text-rose-900 font-semibold">ZONA A · Contam. (C)</span>
                        <strong className="font-mono text-sm text-[#d91b24]">
                          {sceneMap.counts?.rojos_en_A ?? 0}
                        </strong>
                      </div>
                      <div className="bg-white/80 border border-slate-200 rounded-xl px-2.5 py-1 flex items-center justify-between text-slate-700">
                        <span>B_SUP (Sanos)</span>
                        <strong className="font-mono">{sceneMap.counts?.verdes_en_B_SUP ?? 0}</strong>
                      </div>
                      <div className="bg-white/80 border border-slate-200 rounded-xl px-2.5 py-1 flex items-center justify-between text-slate-700">
                        <span>B_INF (Desechos)</span>
                        <strong className="font-mono">{sceneMap.counts?.rojos_en_B_INF ?? 0}</strong>
                      </div>
                    </div>

                    <div className="pt-1.5 border-t border-slate-100 text-xs flex items-center justify-between gap-2">
                      <span className="text-slate-500 font-medium shrink-0">Siguiente orden:</span>
                      <span className="font-mono font-bold text-slate-900 truncate">
                        {sceneMap.next_order || "—"}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs mb-1.5 font-medium">
                      <span className="text-slate-700 font-semibold">Lote #842A · Manzana Royal Gala</span>
                      <span className="font-mono font-bold text-slate-900">
                        1.020 / 1.500 uds <span className="text-[#15803d] ml-1 font-bold">({progressPct}%)</span>
                      </span>
                    </div>

                    <div className="w-full h-2.5 bg-slate-200/80 rounded-full overflow-hidden p-0.5 border border-slate-300/40">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 via-teal-500 to-[#15803d] rounded-full transition-all duration-500"
                        style={{ width: `${progressPct}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-xs text-slate-600 font-medium">
                      <div className="flex items-center space-x-1.5">
                        <span className="w-2 h-2 rounded-full bg-emerald-500" />
                        <span>Aptas Lineal: <strong className="text-slate-800">978 uds</strong> (95.9%)</span>
                      </div>
                      <div className="flex items-center space-x-1.5">
                        <span className="w-2 h-2 rounded-full bg-[#d91b24]" />
                        <span>Merma Retirada: <strong className="text-slate-800">42 uds</strong> (4.1%)</span>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </div>

            {/* BLOQUE 2 (ESCENARIO VISUAL DUAL): Justo debajo de controles en vertical; 7/12 izquierda en horizontal */}
            <div className="order-2 lg:col-span-7 lg:col-start-1 lg:row-start-1 lg:row-span-2 flex flex-col space-y-2.5 min-w-0">
              <div className="flex flex-col gap-2 w-full">
                <div className="flex items-center justify-between gap-2 px-1">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-slate-700">
                    Supervisión Visual
                  </span>
                  <span className="text-[11px] font-mono text-slate-500 bg-white/80 border border-slate-200 px-2.5 py-0.5 rounded-full whitespace-nowrap">
                    {showHudPanel && show3dPanel ? "Mesa Real + Gemelo 3D" : showHudPanel ? "Cámara HUD" : "Visor 3D"}
                  </span>
                </div>

                {hudAvailable && (
                  <div
                    className="w-full p-1.5 rounded-2xl bg-slate-900/95 border border-slate-700/80 shadow-sm grid grid-cols-3 gap-1.5"
                    role="group"
                    aria-label="Modo de visualización"
                  >
                    {[
                      { id: "dual", shortLabel: "Dual", fullLabel: "Vista Dual" },
                      { id: "hud", shortLabel: "HUD", fullLabel: "Cámara HUD" },
                      { id: "3d", shortLabel: "3D", fullLabel: "Visor 3D" }
                    ].map(({ id, shortLabel, fullLabel }) => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setViewerMode(id)}
                        aria-pressed={viewerMode === id}
                        className={`touch-target w-full min-h-[46px] sm:min-h-[48px] px-2 sm:px-4 py-2 rounded-xl text-xs sm:text-sm font-extrabold transition-all cursor-pointer flex items-center justify-center whitespace-nowrap ${
                          viewerMode === id
                            ? "bg-white text-slate-900 shadow-sm"
                            : "text-white/85 hover:text-white hover:bg-white/10"
                        }`}
                      >
                        <span className="sm:hidden">{shortLabel}</span>
                        <span className="hidden sm:inline">{fullLabel}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div
                className={`grid gap-3 items-stretch ${
                  showHudPanel && show3dPanel ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1"
                }`}
              >
                {/* PANEL A: MESA DE TRABAJO REAL CON ZONA A, ZONA B, VASO D Y OBJETOS */}
                {showHudPanel && (
                  <section
                    className="relative rounded-3xl overflow-hidden shadow-md border border-slate-200/90 bg-slate-950 min-h-[240px] sm:min-h-[260px] lg:min-h-[300px] xl:min-h-[400px] flex flex-col justify-between min-w-0"
                    aria-label="Cámara cenital HUD en vivo con zonas y objetos"
                  >
                    <div className="relative z-10 p-2.5 sm:p-3 flex items-center justify-between gap-2 pointer-events-none">
                      <div className="pointer-events-auto flex items-center space-x-2 px-2.5 py-1 rounded-full bg-slate-950/80 backdrop-blur-md border border-white/20 text-white shadow-sm min-w-0">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping shrink-0" />
                        <span className="text-[11px] font-bold tracking-wide truncate">MESA REAL · HUD</span>
                        <span className="text-[10px] text-slate-300 font-mono pl-1.5 border-l border-white/20 shrink-0">
                          {sceneMap?.timestamp?.slice(11, 19) || "640x480"}
                        </span>
                      </div>
                    </div>

                    <div className="flex-1 flex items-center justify-center px-2.5 pb-2.5 min-w-0">
                      <img
                        src={`/api/hud.jpg?t=${status.telemetry?.hud_mtime ?? 0}`}
                        alt={`Detección OpenCV Zona A, Zona B y Vaso D (${sceneMap?.next_order || "en espera"})`}
                        width={640}
                        height={480}
                        fetchPriority="high"
                        className="block w-full h-auto max-h-[240px] sm:max-h-[250px] lg:max-h-[290px] xl:max-h-[380px] aspect-[4/3] object-contain rounded-2xl border border-white/15 bg-black"
                      />
                    </div>

                    <div className="px-3.5 pb-2.5 flex items-center justify-between text-[11px] text-slate-300 font-mono gap-2">
                      <span className="truncate">
                        Orden: <strong className="text-emerald-400">{sceneMap?.next_order || "Esperando escaneo"}</strong>
                      </span>
                      <span className="shrink-0">Vaso D</span>
                    </div>
                  </section>
                )}

                {/* PANEL B: ANIMACIÓN 3D DE LO QUE DEBE HACER EL ROBOT */}
                {show3dPanel && (
                  <section
                    className="relative rounded-3xl overflow-hidden shadow-md border border-slate-200/90 bg-stone-100 min-h-[240px] sm:min-h-[260px] lg:min-h-[300px] xl:min-h-[400px] flex flex-col justify-between min-w-0"
                    aria-label="Animación 3D de la trayectoria esperada del robot"
                  >
                    <div
                      ref={sceneRef}
                      id="scene-container"
                      className="absolute inset-0 w-full h-full bg-stone-100"
                    />

                    <div className="relative z-10 p-2.5 sm:p-3 flex items-center justify-between gap-2 pointer-events-none">
                      <div className="pointer-events-auto flex items-center space-x-2 px-2.5 py-1 rounded-full bg-slate-950/75 backdrop-blur-md border border-white/20 text-white shadow-sm min-w-0">
                        <span className={`w-2 h-2 rounded-full shrink-0 ${status.running ? "bg-emerald-400 animate-ping" : "bg-sky-400"}`} />
                        <span className="text-[11px] font-bold tracking-wide truncate">GEMELO 3D · SO-101</span>
                      </div>
                    </div>

                    <div className="relative z-10 pb-2.5 px-2 flex justify-center pointer-events-none">
                      <div className="pointer-events-auto p-1 bg-slate-950/75 backdrop-blur-md rounded-full border border-white/20 shadow-lg flex flex-wrap items-center justify-center gap-0.5">
                        {[
                          { id: "iso", label: "Iso", Icon: Box },
                          { id: "front", label: "Frontal", Icon: Eye },
                          { id: "top", label: "Top", Icon: ArrowDown }
                        ].map(({ id, label, Icon }) => {
                          const active = cameraView === id;
                          return (
                            <button
                              key={id}
                              type="button"
                              onClick={() => handleCameraView(id)}
                              className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-[11px] transition-all cursor-pointer ${
                                active
                                  ? "bg-white text-slate-900 font-bold shadow-sm"
                                  : "text-white/80 hover:text-white font-medium"
                              }`}
                            >
                              <Icon className={`w-3 h-3 ${active ? "text-[#15803d]" : "text-slate-400"}`} aria-hidden="true" />
                              <span>{label}</span>
                            </button>
                          );
                        })}
                        <button
                          type="button"
                          onClick={() => handleCameraView("iso")}
                          aria-label="Centrar vista 3D"
                          title="Centrar vista 3D"
                          className="px-2 py-1 text-slate-400 hover:text-white transition-colors cursor-pointer"
                        >
                          <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                        </button>
                      </div>
                    </div>
                  </section>
                )}
              </div>
            </div>

            {/* BLOQUE 3 (PIPELINE, TELEMETRÍA Y LOG): Debajo de los visores en vertical; Derecha abajo en horizontal */}
            <aside className="order-3 lg:col-span-5 lg:col-start-8 lg:row-start-2 flex flex-col gap-3 min-w-0" aria-label="Pipeline y registros">

              {/* 3. Pipeline de 3 fases */}
              <section className="grid grid-cols-3 gap-2">
                <div
                  className={`rounded-2xl p-3 flex flex-col justify-between border transition-all ${
                    activePhase === 1
                      ? "bg-white border-emerald-500 shadow-sm"
                      : "bg-white/70 border-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-emerald-800 bg-emerald-100/70 px-1.5 py-0.5 rounded">
                      PASO 1
                    </span>
                    {activePhase > 1 ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : activePhase === 1 ? (
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                    )}
                  </div>
                  <div className="mt-1.5">
                    <div className="text-xs font-bold text-slate-900 leading-tight">Recogida Cesta</div>
                    <div className="text-[10px] text-emerald-700 font-medium">
                      {activePhase === 1 ? "En curso" : "Completada OK"}
                    </div>
                  </div>
                </div>

                <div
                  className={`rounded-2xl p-3 flex flex-col justify-between border transition-all ${
                    activePhase === 2
                      ? "bg-white border-emerald-500 shadow-sm ring-1 ring-emerald-500"
                      : "bg-white/70 border-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-white bg-emerald-700 px-1.5 py-0.5 rounded">
                      PASO 2
                    </span>
                    {activePhase === 2 ? (
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                    ) : activePhase > 2 ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                    )}
                  </div>
                  <div className="mt-1.5">
                    <div className="text-xs font-bold text-slate-900 leading-tight">Clasificación</div>
                    <div className="text-[10px] text-emerald-800 font-semibold">
                      {activePhase === 2 ? "En curso · IA Óptica" : "Pendiente"}
                    </div>
                  </div>
                </div>

                <div
                  className={`rounded-2xl p-3 flex flex-col justify-between border transition-all ${
                    activePhase === 3
                      ? "bg-white border-emerald-500 shadow-sm"
                      : "bg-slate-50/70 border-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-600 bg-slate-200/80 px-1.5 py-0.5 rounded">
                      PASO 3
                    </span>
                    {activePhase === 3 ? (
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                    )}
                  </div>
                  <div className="mt-1.5">
                    <div className="text-xs font-bold text-slate-700 leading-tight">Higiene UV-C</div>
                    <div className="text-[10px] text-slate-500 font-medium">
                      {activePhase === 3 ? "254nm Activa" : "En cola (2s)"}
                    </div>
                  </div>
                </div>
              </section>

              {/* 4. Telemetría compacta */}
              <section className="grid grid-cols-2 gap-2.5">
                <div className="liquid-panel rounded-2xl p-3 flex flex-col justify-between">
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Ciclos Robot
                  </span>
                  <div className="flex items-baseline space-x-1 mt-1">
                    <span className="text-lg font-extrabold text-slate-900 font-mono">
                      {completedCycles}
                    </span>
                    <span className="text-[10px] text-slate-500 font-semibold">turnos</span>
                  </div>
                  <span className="text-[9px] text-[#15803d] font-bold mt-0.5">Cadencia 42 uds/min</span>
                </div>

                <div className="liquid-panel rounded-2xl p-3 flex flex-col justify-between border-l-2 border-l-sky-500">
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Higiene UV-C
                  </span>
                  <div className="flex items-baseline space-x-1 mt-1">
                    <span className="text-lg font-extrabold text-sky-700 font-mono">254</span>
                    <span className="text-[10px] text-sky-600 font-semibold">nm</span>
                  </div>
                  <span className="text-[9px] text-sky-600 font-bold mt-0.5">Lámpara 100% Activa</span>
                </div>

                <div className="liquid-panel rounded-2xl p-3 flex flex-col justify-between col-span-2">
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    {status.last_event || sceneMap ? "Último Evento del Supervisor" : "Última Selección"}
                  </span>
                  {status.last_event || sceneMap ? (
                    <>
                      <div className="flex items-baseline justify-between mt-1 gap-2">
                        <span className="text-xs font-mono font-bold text-slate-800 truncate">
                          {status.last_event || sceneMap?.next_order}
                        </span>
                        <span className="text-xs font-mono font-bold text-emerald-700 shrink-0">
                          {sceneMap?.target ? `target: ${sceneMap.target}` : "en vivo"}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 mt-0.5 truncate">
                        {status.message}
                      </span>
                    </>
                  ) : (
                    <>
                      <div className="flex items-baseline justify-between mt-1">
                        <span className="text-xs font-bold text-slate-800">Corteza &gt; 12mm · Merma</span>
                        <span className="text-xs font-mono font-bold text-[#d91b24] shrink-0">-1 ud (#1019)</span>
                      </div>
                      <span className="text-[10px] text-slate-500 mt-0.5">
                        Desvío automático a bandeja merma completado
                      </span>
                    </>
                  )}
                </div>
              </section>

              {/* 5. Cajón desplegable de Registro (Audit Log) */}
              <section className="rounded-2xl border border-slate-200/80 bg-white/70 overflow-hidden shadow-sm">
                <button
                  type="button"
                  onClick={() => setLogOpen((v) => !v)}
                  className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-white/90 transition-colors cursor-pointer"
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold text-slate-800">
                      Registro de actividad en tiempo real (Audit Log)
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-slate-200 text-slate-700 text-[10px] font-mono font-medium">
                      {status.logs ? status.logs.length : 0} eventos
                    </span>
                  </div>
                  <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${logOpen ? "rotate-180" : ""}`} />
                </button>

                {logOpen && (
                  <div className="p-3.5 border-t border-slate-200/70 bg-slate-50/90 text-xs font-mono space-y-2">
                    <div className="flex items-center justify-between pb-1 mb-1 border-b border-slate-200/60">
                      <span className="text-[10px] uppercase font-bold text-slate-500">Historial reciente</span>
                      <button
                        type="button"
                        onClick={handleClearLogs}
                        className="text-[11px] text-slate-500 hover:text-slate-800 flex items-center space-x-1 cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                        <span>Limpiar</span>
                      </button>
                    </div>

                    <div className="max-h-40 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] space-y-1.5 text-slate-700">
                      {status.logs && status.logs.length > 0 ? (
                        status.logs.map((logLine, idx) => (
                          <div key={idx} className="leading-snug">
                            {logLine}
                          </div>
                        ))
                      ) : (
                        <div className="text-slate-400 italic">Sin eventos registrados.</div>
                      )}
                    </div>
                  </div>
                )}
              </section>

            </aside>
          </div>
        )}

        {/* PESTAÑA 2: CALIBRACIÓN DEL BRAZO SO-101 */}
        {activeTab === "calibracion" && (
          <div className="space-y-4">
            {calibMessage && (
              <div className="rounded-2xl p-3 bg-sky-500/15 border border-sky-500/35 text-sky-900 text-xs font-bold flex items-center space-x-2 shadow-sm">
                <Info className="w-4 h-4 text-sky-700 shrink-0" />
                <span>{calibMessage}</span>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Articulaciones del brazo */}
              <section className="liquid-panel rounded-3xl p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Articulaciones SO-101 (6-DOF)
                    </span>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                      Dynamixel Calibrados
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    Comprobación de offset angular y estado de comunicación serial Leader/Follower.
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-2.5 text-xs font-mono">
                  {[
                    { joint: "J1 Base", angle: "0.0°", ok: true },
                    { joint: "J2 Hombro", angle: "-45.2°", ok: true },
                    { joint: "J3 Codo", angle: "+88.5°", ok: true },
                    { joint: "J4 Muñeca P", angle: "-43.1°", ok: true },
                    { joint: "J5 Muñeca R", angle: "0.0°", ok: true },
                    { joint: "J6 Gripper", angle: "100%", ok: true }
                  ].map(({ joint, angle, ok }) => (
                    <div key={joint} className="bg-white/80 p-2.5 rounded-xl border border-slate-200 flex flex-col">
                      <span className="text-[10px] text-slate-400 font-sans font-semibold">{joint}</span>
                      <span className="font-bold text-slate-800 mt-0.5">{angle}</span>
                      <span className="text-[9px] text-emerald-600 font-bold mt-1">✓ OK</span>
                    </div>
                  ))}
                </div>

                <div className="pt-2 border-t border-slate-200 flex gap-2">
                  <button
                    type="button"
                    onClick={handleHoming}
                    className="flex-1 py-2.5 px-4 rounded-xl bg-slate-900 text-white font-bold text-xs hover:bg-slate-800 cursor-pointer flex items-center justify-center space-x-1.5 shadow-sm"
                  >
                    <RotateCw className="w-3.5 h-3.5" />
                    <span>Homing (Posición Reposo)</span>
                  </button>
                </div>
              </section>

              {/* Sistema de succión y vacío */}
              <section className="liquid-panel rounded-3xl p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Circuito Neumático de Vacío
                    </span>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                      94.2 kPa Nominal
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    Ventosa de agarre suave diseñada para piel sensible de manzanas sin marcas mecánicas.
                  </p>
                </div>

                <div className="bg-white/90 p-4 rounded-2xl border border-slate-200 flex items-center justify-between">
                  <div className="flex items-center space-x-3.5">
                    <div className="w-11 h-11 rounded-xl bg-sky-50 text-sky-700 flex items-center justify-center">
                      <Gauge className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-base font-bold text-slate-900 font-mono">
                        {vacuumActive ? "96.4 kPa (Activo)" : "94.2 kPa (Nominal)"}
                      </div>
                      <div className="text-xs text-slate-500">Electroválvula SMC · Filtro OK</div>
                    </div>
                  </div>
                  <span className={`w-3.5 h-3.5 rounded-full ${vacuumActive ? "bg-emerald-500 animate-ping" : "bg-emerald-500"}`} />
                </div>

                <div className="pt-2 border-t border-slate-200 flex gap-2">
                  <button
                    type="button"
                    onClick={handleTestVacuum}
                    className="flex-1 py-2.5 px-4 rounded-xl bg-[#15803d] text-white font-bold text-xs hover:bg-emerald-700 cursor-pointer flex items-center justify-center space-x-1.5 shadow-sm"
                  >
                    <Zap className="w-3.5 h-3.5" />
                    <span>Test Succión (2s)</span>
                  </button>
                </div>
              </section>

              {/* Cámaras de Visión Mac */}
              <section className="liquid-panel rounded-3xl p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Cámaras de Visión Artificial (Mac)
                    </span>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-sky-100 text-sky-800 border border-sky-200">
                      2x Enlazadas
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    Flujo de inferencia gestionado localmente en LeRobot / LeLab en la Mac host.
                  </p>
                </div>

                <div className="space-y-2.5">
                  <div className="bg-white/80 p-3 rounded-xl border border-slate-200 flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-2.5">
                      <Video className="w-4 h-4 text-slate-600" />
                      <div>
                        <div className="font-bold text-slate-800">Cámara Cenital (Overhead)</div>
                        <div className="text-[10px] text-slate-500">1080p @ 60 FPS · Detección de fruta</div>
                      </div>
                    </div>
                    <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded border border-emerald-200">
                      11 ms
                    </span>
                  </div>

                  <div className="bg-white/80 p-3 rounded-xl border border-slate-200 flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-2.5">
                      <Video className="w-4 h-4 text-slate-600" />
                      <div>
                        <div className="font-bold text-slate-800">Cámara Muñeca (Lateral)</div>
                        <div className="text-[10px] text-slate-500">1080p @ 60 FPS · Alineación de agarre</div>
                      </div>
                    </div>
                    <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded border border-emerald-200">
                      13 ms
                    </span>
                  </div>
                </div>
              </section>

              {/* Perfil de Velocidad */}
              <section className="liquid-panel rounded-3xl p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Velocidad y Modo de Seguridad
                    </span>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-200">
                      {armSpeed}% Seleccionado
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    Límite cinemático para demostración o clasificación a ritmo continuo de tienda.
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-2.5">
                  {[
                    { label: "100%", desc: "Producción", val: "100" },
                    { label: "50%", desc: "Mantenimiento", val: "50" },
                    { label: "25%", desc: "Modo Seguro", val: "25" }
                  ].map(({ label, desc, val }) => (
                    <button
                      key={val}
                      type="button"
                      onClick={() => {
                        setArmSpeed(val);
                        showToast(`Velocidad configurada al ${val}%.`);
                      }}
                      className={`p-3 rounded-xl border text-center transition-all cursor-pointer ${
                        armSpeed === val
                          ? "bg-slate-900 text-white border-slate-900 shadow-sm"
                          : "bg-white/80 text-slate-700 border-slate-200 hover:bg-white"
                      }`}
                    >
                      <div className="font-extrabold text-sm">{label}</div>
                      <div className="text-[10px] opacity-80 mt-0.5">{desc}</div>
                    </button>
                  ))}
                </div>
              </section>
            </div>
          </div>
        )}

        {/* PESTAÑA 3: LOTES Y TRAZABILIDAD */}
        {activeTab === "lotes" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            
            {/* Tarjeta de Lote Activo */}
            <section className="liquid-panel rounded-3xl p-5 space-y-3.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Ficha de Lote Activo
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-red-100 text-[#d91b24] border border-red-200">
                  Eusko Label · SUPERMARKET 5.12 Natur
                </span>
              </div>

              <div className="bg-white/80 p-4 rounded-2xl border border-slate-200 space-y-2.5">
                <div className="text-base font-extrabold text-slate-900">
                  LOTE #842A · Manzana Royal Gala
                </div>
                <div className="grid grid-cols-2 gap-2.5 text-xs text-slate-600">
                  <div>
                    <span className="text-slate-400 block text-[10px]">Origen:</span>
                    <strong>Finca Valdegovía (Álava)</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Recepción Almacén:</span>
                    <strong>Hoy 06:30 h</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Temp. Cámara:</span>
                    <strong className="font-mono">4.2 °C (Óptima)</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Humedad Relativa:</span>
                    <strong className="font-mono">88%</strong>
                  </div>
                </div>
              </div>
            </section>

            {/* Balance de Calidad y Merma */}
            <section className="liquid-panel rounded-3xl p-5 space-y-3.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Balance de Calidad (Turno)
                </span>
                <span className="font-mono text-xs font-bold text-slate-800">
                  1.020 uds totales
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="bg-emerald-50 p-3 rounded-xl border border-emerald-200 flex items-center justify-between">
                  <div>
                    <strong className="text-emerald-900">Aptas para Lineal (Cat. Extra / I)</strong>
                    <div className="text-[10px] text-emerald-700">Calibre óptimo &gt; 65mm sin golpes</div>
                  </div>
                  <span className="font-mono font-bold text-sm text-emerald-800">978 uds (95.9%)</span>
                </div>

                <div className="bg-red-50 p-3 rounded-xl border border-red-200 flex items-center justify-between">
                  <div>
                    <strong className="text-red-900">Merma Retirada (No Comercial)</strong>
                    <div className="text-[10px] text-red-700">Derivadas a compostaje o zumos</div>
                  </div>
                  <span className="font-mono font-bold text-sm text-[#d91b24]">42 uds (4.1%)</span>
                </div>
              </div>
            </section>

            {/* Protocolo de Higiene UV-C */}
            <section className="liquid-panel rounded-3xl p-5 space-y-3.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Protocolo Sanitario UV-C
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-sky-100 text-sky-800 border border-sky-200">
                  254 nm Germicida
                </span>
              </div>

              <div className="bg-white/80 p-4 rounded-2xl border border-slate-200 space-y-2 text-xs text-slate-700">
                <div className="flex justify-between border-b border-slate-100 pb-1.5">
                  <span>Dosis media administrada por fruta:</span>
                  <strong className="font-mono">45.2 mJ/cm²</strong>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-1.5">
                  <span>Eficacia bacteriana superficial estimada:</span>
                  <strong className="font-mono text-emerald-700">&gt; 99.98%</strong>
                </div>
                <div className="flex justify-between">
                  <span>Estado del emisor de cuarzo:</span>
                  <strong className="text-slate-900">100% Vida Útil (320h)</strong>
                </div>
              </div>
            </section>

            {/* Exportación y Acciones de Trazabilidad */}
            <section className="liquid-panel rounded-3xl p-5 flex flex-col justify-between space-y-4">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Cierre y Exportación
                </span>
                <p className="text-xs text-slate-600 mt-1">
                  Genera el registro para el sistema central de SUPERMARKET 5.12 o finaliza el turno del robot.
                </p>
              </div>

              <div className="space-y-2.5">
                <button
                  type="button"
                  onClick={handleExportReport}
                  className="w-full py-2.5 px-4 rounded-xl bg-slate-900 text-white font-bold text-xs hover:bg-slate-800 cursor-pointer flex items-center justify-center space-x-2 shadow-sm"
                >
                  <Download className="w-4 h-4" />
                  <span>Descargar Informe de Trazabilidad (.CSV)</span>
                </button>

                <button
                  type="button"
                  onClick={() => showToast("Lote #842A precintado y listo para expedición.")}
                  className="w-full py-2.5 px-4 rounded-xl bg-white text-slate-700 border border-slate-300 font-bold text-xs hover:bg-slate-50 cursor-pointer flex items-center justify-center space-x-2"
                >
                  <CheckCircle2 className="w-4 h-4 text-[#15803d]" />
                  <span>Precintar Lote y Finalizar Turno</span>
                </button>
              </div>
            </section>
          </div>
        )}

      </main>

      {/* PIE DE PÁGINA WEB REAL (FOOTER) */}
      {!isFullscreen && (
        <footer className="w-full border-t border-slate-200/80 bg-white/50 backdrop-blur-sm py-4 px-6 mt-auto pb-28">
          <div className="max-w-[1600px] mx-auto flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2">
            <span>Estación Autónoma SO-101 · Hackathon Bilbao 2026</span>
            <span>SUPERMARKET 5.12 · Control de Calidad y Clasificación Robótica</span>
          </div>
        </footer>
      )}

      {/* BARRA DE NAVEGACIÓN INFERIOR (se oculta en modo inmersivo CSS para aprovechar toda la pantalla) */}
      {!cssFullscreen && (
      <nav
        className="bottom-dock fixed left-1/2 -translate-x-1/2 z-40 pointer-events-auto"
        aria-label="Navegación principal de la estación"
      >
        <div className="liquid-panel-strong rounded-full p-1.5 shadow-2xl border border-white/90 inline-flex items-center space-x-1 backdrop-blur-2xl">
          <button
            type="button"
            onClick={() => setActiveTab("triaje")}
            className={`px-3.5 sm:px-4 py-2 sm:py-2.5 rounded-full font-bold text-xs sm:text-sm whitespace-nowrap transition-all inline-flex items-center space-x-1.5 cursor-pointer shrink-0 ${
              activeTab === "triaje"
                ? "bg-slate-900 text-white shadow-md"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
            }`}
          >
            <Activity className={`w-4 h-4 shrink-0 ${activeTab === "triaje" ? "text-emerald-400" : "text-slate-500"}`} />
            <span className="whitespace-nowrap">Triaje<span className="hidden md:inline">&nbsp;en Vivo</span></span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("calibracion")}
            className={`px-3.5 sm:px-4 py-2 sm:py-2.5 rounded-full font-bold text-xs sm:text-sm whitespace-nowrap transition-all inline-flex items-center space-x-1.5 cursor-pointer shrink-0 ${
              activeTab === "calibracion"
                ? "bg-slate-900 text-white shadow-md"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
            }`}
          >
            <Sliders className={`w-4 h-4 shrink-0 ${activeTab === "calibracion" ? "text-emerald-400" : "text-slate-500"}`} />
            <span className="whitespace-nowrap">Calibración</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("lotes")}
            className={`px-3.5 sm:px-4 py-2 sm:py-2.5 rounded-full font-bold text-xs sm:text-sm whitespace-nowrap transition-all inline-flex items-center space-x-1.5 cursor-pointer shrink-0 ${
              activeTab === "lotes"
                ? "bg-slate-900 text-white shadow-md"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
            }`}
          >
            <ClipboardList className={`w-4 h-4 shrink-0 ${activeTab === "lotes" ? "text-emerald-400" : "text-slate-500"}`} />
            <span className="whitespace-nowrap">Lotes<span className="hidden md:inline">&nbsp;& Trazabilidad</span></span>
          </button>
        </div>
      </nav>
      )}

    </div>
  );
}
