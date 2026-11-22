#!/usr/bin/env python3
"""
Fase 3 — Calibrador HSV/ROI y Supervisor Autónomo OpenCV para "Zero-Cross-Contamination Triage".

Modos de operación:
  1. --mode calibrate : Calibrador interactivo (ROIs por ratón + trackbars HSV/Área).
                        Guarda `outputs/triage_vision_config.json` al pulsar 's'.
  2. --mode check     : HUD en vivo (sin mover el robot) para verificar detección
                        de cubos Verdes (S) y Rojos (C) en ZONA_A, ZONA_B_SUP y ZONA_B_INF.
  3. --mode map       : Escanea `camera_top` y guarda el mapa de coordenadas en `outputs/triage_scene_map.json`.
  4. --mode align     : Superposición en vivo 50/50 sobre el frame de referencia de entrenamiento.
  5. --mode home      : Mueve suavemente el brazo a `DEFAULT_HOME_POSE` manteniendo torque.
  6. --mode run       : Orquestador autónomo en bucle cerrado (Clean-First):
                        - Si hay Verdes en ZONA_A -> ejecuta `act_sano`
                        - Si no hay Verdes y hay Rojos en ZONA_A -> ejecuta `act_contaminado`
                        - Cierra `cv2.VideoCapture` antes de cada `lerobot-rollout` (regla macOS)
                        - Vuelve a HOME, remapea coordenadas si el escenario cambió y reintenta hasta --max-retries
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "outputs" / "triage_vision_config.json"
DEFAULT_HUD_SNAPSHOT_PATH = REPO_ROOT / "outputs" / "triage_hud_latest.jpg"
DEFAULT_SCENE_MAP_PATH = REPO_ROOT / "outputs" / "triage_scene_map.json"
DEFAULT_REMAP_THRESHOLD_PX = 12.0
DEFAULT_POLICY_SANO = REPO_ROOT / "outputs" / "train" / "act_sano" / "checkpoints" / "last" / "pretrained_model"
DEFAULT_POLICY_CONTAMINADO = REPO_ROOT / "outputs" / "train" / "act_contaminado" / "checkpoints" / "last" / "pretrained_model"

DEFAULT_REFERENCE_FRAME_PATH = REPO_ROOT / "outputs" / "viz" / "train_sano_camera_top_file000_f0.jpg"

# Postura media HOME medida sobre los 192 episodios limpios del dataset V3 (evita mismatch de ~18° en Chunk 0)
DEFAULT_HOME_POSE: dict[str, float] = {
    "shoulder_pan.pos": -4.9,
    "shoulder_lift.pos": -46.7,
    "elbow_flex.pos": 8.9,
    "wrist_flex.pos": 99.6,
    "wrist_roll.pos": -5.5,
    "gripper.pos": 1.4,
}

# Postura media VASO D medida sobre los episodios del dataset V4 (baño de bioseguridad)
DEFAULT_VASO_D_POSE: dict[str, float] = {
    "shoulder_pan.pos": -1.5,
    "shoulder_lift.pos": 49.9,
    "elbow_flex.pos": -37.9,
    "wrist_flex.pos": 66.7,
    "wrist_roll.pos": -1.0,
    "gripper.pos": 1.4,
}

ROI_NAMES = ("ZONA_A", "ZONA_B_SUP", "ZONA_B_INF", "VASO_D")
ROI_COLORS_BGR = {
    "ZONA_A": (255, 200, 0),      # Cyan/Azul claro
    "ZONA_B_SUP": (0, 220, 0),    # Verde
    "ZONA_B_INF": (0, 0, 230),    # Rojo
    "VASO_D": (255, 0, 255),      # Magenta
}


def default_vision_config(cam_top_idx: int = 1, width: int = 640, height: int = 480) -> dict[str, Any]:
    """Return baseline ROI and HSV thresholds calibrated for 640x480 top-view triage."""
    return {
        "camera_top_index": int(cam_top_idx),
        "width": int(width),
        "height": int(height),
        "rois": {
            "ZONA_A": [345, 55, 575, 245],
            "ZONA_B_SUP": [120, 120, 325, 240],
            "ZONA_B_INF": [115, 25, 325, 116],
            "VASO_D": [315, 210, 375, 275],
        },
        "hsv_green": {
            "h_min": 35,
            "s_min": 60,
            "v_min": 50,
            "h_max": 90,
            "s_max": 255,
            "v_max": 255,
        },
        "hsv_red": {
            "h1_min": 0,
            "h1_max": 7,
            "h2_min": 165,
            "h2_max": 180,
            "s_min": 80,
            "v_min": 50,
            "s_max": 255,
            "v_max": 255,
        },
        "min_area_px": 300,
        "max_area_px": 2500,
    }


def load_vision_config(config_path: Path, cam_top_idx: int | None = None) -> dict[str, Any]:
    """Load vision config from JSON, falling back to defaults if file does not exist."""
    base = default_vision_config(cam_top_idx=cam_top_idx if cam_top_idx is not None else 1)
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        for key, val in loaded.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key].update(val)
            else:
                base[key] = val
    if cam_top_idx is not None:
        base["camera_top_index"] = int(cam_top_idx)
    return base


def save_vision_config(config: dict[str, Any], config_path: Path) -> None:
    """Persist ROI and HSV configuration atomically to JSON."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class DetectedCube:
    color: str  # "S" (verde) or "C" (rojo)
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    centroid: tuple[int, int]        # (cx, cy)
    area: float
    roi: str | None


def _point_in_roi(cx: int, cy: int, roi_box: list[int] | tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi_box
    return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)


def _point_in_polygon(cx: int, cy: int, polygon: list[list[int]] | list[tuple[int, int]]) -> bool:
    pts = np.asarray(polygon, dtype=np.int32)
    return float(cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)) >= 0.0


def _assign_roi(
    cx: int,
    cy: int,
    rois: dict[str, list[int]],
    roi_polygons: dict[str, list[list[int]]] | None = None,
) -> str | None:
    polys = roi_polygons or {}
    for name in ("ZONA_A", "ZONA_B_SUP", "ZONA_B_INF", "VASO_D"):
        poly = polys.get(name)
        if poly and len(poly) >= 3:
            if _point_in_polygon(cx, cy, poly):
                return name
            continue
        box = rois.get(name)
        if box and _point_in_roi(cx, cy, box):
            return name
    return None


def compute_color_masks(frame_bgr: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build cleaned HSV binary masks for Green (Sano) and dual-range Red (Contaminado)."""
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    g = config["hsv_green"]
    lower_g = np.array([g["h_min"], g["s_min"], g["v_min"]], dtype=np.uint8)
    upper_g = np.array([g["h_max"], g["s_max"], g["v_max"]], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_g, upper_g)

    r = config["hsv_red"]
    lower_r1 = np.array([r["h1_min"], r["s_min"], r["v_min"]], dtype=np.uint8)
    upper_r1 = np.array([r["h1_max"], r["s_max"], r["v_max"]], dtype=np.uint8)
    lower_r2 = np.array([r["h2_min"], r["s_min"], r["v_min"]], dtype=np.uint8)
    upper_r2 = np.array([r["h2_max"], r["s_max"], r["v_max"]], dtype=np.uint8)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), cv2.inRange(hsv, lower_r2, upper_r2))

    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    return mask_green, mask_red


def _extract_cubes_from_mask(
    mask: np.ndarray,
    color_label: str,
    rois: dict[str, list[int]],
    min_area: float,
    max_area: float,
    roi_polygons: dict[str, list[list[int]]] | None = None,
) -> list[DetectedCube]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cubes: list[DetectedCube] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        roi = _assign_roi(cx, cy, rois, roi_polygons=roi_polygons)
        cubes.append(DetectedCube(color=color_label, bbox=(x, y, w, h), centroid=(cx, cy), area=area, roi=roi))
    return cubes


def determine_next_action(verdes_en_a: int, rojos_en_a: int) -> tuple[str, str]:
    """Return (target_policy_key, human_readable_order) following Clean-First biosecurity."""
    if verdes_en_a > 0:
        return "sano", "EVACUAR SANO -> act_sano"
    if rojos_en_a > 0:
        return "contaminado", "CUARENTENA + VASO D -> act_contaminado"
    return "completed", "ZONA A LIMPIA - COMPLETADO"


def analyze_frame(frame_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Detect green (S) and red (C) cubes across ROIs and compute Clean-First counts."""
    mask_green, mask_red = compute_color_masks(frame_bgr, config)
    rois = config["rois"]
    roi_polygons = config.get("roi_polygons")
    min_area = float(config.get("min_area_px", 180))
    max_area = float(config.get("max_area_px", 15000))

    green_cubes = _extract_cubes_from_mask(mask_green, "S", rois, min_area, max_area, roi_polygons=roi_polygons)
    red_cubes = _extract_cubes_from_mask(mask_red, "C", rois, min_area, max_area, roi_polygons=roi_polygons)
    all_cubes = green_cubes + red_cubes

    verdes_a = sum(1 for c in green_cubes if c.roi == "ZONA_A")
    rojos_a = sum(1 for c in red_cubes if c.roi == "ZONA_A")
    verdes_b_sup = sum(1 for c in green_cubes if c.roi == "ZONA_B_SUP")
    rojos_b_inf = sum(1 for c in red_cubes if c.roi == "ZONA_B_INF")
    verdes_perdidos = sum(1 for c in green_cubes if c.roi is None)
    rojos_perdidos = sum(1 for c in red_cubes if c.roi is None)
    rojos_area_b_inf = float(sum(c.area for c in red_cubes if c.roi == "ZONA_B_INF"))

    target_key, next_order = determine_next_action(verdes_a, rojos_a)
    return {
        "cubes": all_cubes,
        "mask_green": mask_green,
        "mask_red": mask_red,
        "verdes_en_A": verdes_a,
        "rojos_en_A": rojos_a,
        "verdes_en_B_SUP": verdes_b_sup,
        "rojos_en_B_INF": rojos_b_inf,
        "verdes_perdidos": verdes_perdidos,
        "rojos_perdidos": rojos_perdidos,
        "rojos_area_en_B_INF": round(rojos_area_b_inf, 1),
        "target": target_key,
        "next_order": next_order,
    }


def _coerce_cube_dict(cube: Any) -> dict[str, Any]:
    if isinstance(cube, DetectedCube):
        return {
            "color": cube.color,
            "roi": cube.roi or "OUTSIDE",
            "centroid": [int(cube.centroid[0]), int(cube.centroid[1])],
            "bbox": [int(v) for v in cube.bbox],
            "area": round(float(cube.area), 1),
        }
    cx, cy = cube["centroid"]
    return {
        "color": str(cube["color"]),
        "roi": str(cube.get("roi") or "OUTSIDE"),
        "centroid": [int(cx), int(cy)],
        "bbox": [int(v) for v in cube.get("bbox", [cx, cy, 0, 0])],
        "area": round(float(cube.get("area", 0.0)), 1),
    }


def build_scene_map(analysis: dict[str, Any]) -> dict[str, Any]:
    """Serialize detected cube coordinates and counts per ROI into a JSON-friendly scenario map."""
    objects = [_coerce_cube_dict(c) for c in analysis.get("cubes", [])]
    objects.sort(key=lambda item: (item["roi"], item["color"], item["centroid"][1], item["centroid"][0]))
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts": {
            "verdes_en_A": int(analysis.get("verdes_en_A", 0)),
            "rojos_en_A": int(analysis.get("rojos_en_A", 0)),
            "verdes_en_B_SUP": int(analysis.get("verdes_en_B_SUP", 0)),
            "rojos_en_B_INF": int(analysis.get("rojos_en_B_INF", 0)),
        },
        "target": str(analysis.get("target", "")),
        "next_order": str(analysis.get("next_order", "")),
        "objects": objects,
    }


def save_scene_map(scene_map: dict[str, Any], map_path: Path = DEFAULT_SCENE_MAP_PATH) -> None:
    """Atomically persist mapped cube coordinates and ROI counts to JSON."""
    map_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = map_path.with_name(f".{map_path.stem}.tmp{map_path.suffix}")
    tmp_path.write_text(json.dumps(scene_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, map_path)


def _save_hud_atomic(hud_path: Path, hud_bgr: np.ndarray) -> None:
    """Atomically write HUD JPEG so web polling never reads a half-written image."""
    hud_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = hud_path.with_name(f".{hud_path.stem}.tmp{hud_path.suffix}")
    if cv2.imwrite(str(tmp_path), hud_bgr):
        os.replace(tmp_path, hud_path)


def _archive_fail_snapshot(hud_path: Path, fail_path: Path) -> bool:
    """Copy latest rendered HUD atomically to a diagnostic fail file when an attempt fails."""
    if not hud_path.exists():
        return False
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = fail_path.with_name(f".{fail_path.stem}.tmp{fail_path.suffix}")
    shutil.copyfile(hud_path, tmp_path)
    os.replace(tmp_path, fail_path)
    return True


def detect_scene_change(
    prev_map: dict[str, Any] | None,
    curr_map: dict[str, Any],
    move_threshold_px: float = DEFAULT_REMAP_THRESHOLD_PX,
) -> dict[str, Any]:
    """Detect count changes or significant cube coordinate shifts between two scenario maps."""
    if prev_map is None:
        return {
            "changed": True,
            "counts_changed": False,
            "moved_objects": [],
            "max_shift_px": 0.0,
            "summary": "Mapa inicial generado",
        }

    counts_changed = prev_map.get("counts") != curr_map.get("counts")
    prev_objs = [o for o in prev_map.get("objects", []) if o.get("roi") == "ZONA_A"]
    curr_objs = [o for o in curr_map.get("objects", []) if o.get("roi") == "ZONA_A"]

    moved_objects: list[dict[str, Any]] = []
    max_shift_px = 0.0

    for color in ("S", "C"):
        prev_group = [o for o in prev_objs if o.get("color") == color]
        curr_group = [o for o in curr_objs if o.get("color") == color]
        used_curr: set[int] = set()
        for prev_obj in prev_group:
            px, py = prev_obj["centroid"]
            best_idx: int | None = None
            best_dist = float("inf")
            for idx, curr_obj in enumerate(curr_group):
                if idx in used_curr:
                    continue
                cx, cy = curr_obj["centroid"]
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                used_curr.add(best_idx)
                max_shift_px = max(max_shift_px, best_dist)
                if best_dist >= move_threshold_px:
                    cx, cy = curr_group[best_idx]["centroid"]
                    moved_objects.append(
                        {
                            "color": color,
                            "from": [px, py],
                            "to": [cx, cy],
                            "shift_px": round(best_dist, 1),
                        }
                    )

    changed = counts_changed or bool(moved_objects)
    parts: list[str] = []
    if counts_changed:
        parts.append("conteo actualizado")
    if moved_objects:
        parts.append(f"{len(moved_objects)} objeto(s) movido(s), máx Δ={max_shift_px:.1f} px")
    summary = "; ".join(parts) if parts else f"sin desplazamientos >= {move_threshold_px:.0f} px (máx Δ={max_shift_px:.1f} px)"
    return {
        "changed": changed,
        "counts_changed": counts_changed,
        "moved_objects": moved_objects,
        "max_shift_px": round(max_shift_px, 1),
        "summary": summary,
    }


class EarlyStopTracker:
    """Thread-safe live OpenCV + kinematic completion detector to avoid cutting/overrunning cycles."""

    def __init__(self, target_key: str, baseline_counts: dict[str, int]) -> None:
        self.target_key = str(target_key)
        self.base_v_a = int(baseline_counts.get("verdes_en_A", 0))
        self.base_r_a = int(baseline_counts.get("rojos_en_A", 0))
        self.base_v_b = int(baseline_counts.get("verdes_en_B_SUP", 0))
        self.base_r_b = int(baseline_counts.get("rojos_en_B_INF", 0))
        self.base_v_lost = int(baseline_counts.get("verdes_perdidos", 0))
        self.base_r_lost = int(baseline_counts.get("rojos_perdidos", 0))
        self.base_r_b_area = float(baseline_counts.get("rojos_area_en_B_INF", 0.0))
        self._has_area_baseline = "rojos_area_en_B_INF" in baseline_counts

        self._lock = threading.Lock()
        self._sano_stable = 0
        self._desechos_stable = 0
        self.dropped_in_desechos = False
        self.lost_outside_tape = False
        self.visited_vaso_d = False
        self.stop_reason: str | None = None

    def observe_robot_state(self, obs_state: dict[str, Any], elapsed_s: float) -> bool:
        """Track biosecurity Vaso D visit at full control rate after Desechos drop is confirmed."""
        with self._lock:
            if self.stop_reason is not None:
                return True
            if self.target_key != "contaminado" or not self.dropped_in_desechos:
                return False
            pan = float(obs_state.get("shoulder_pan.pos", 99.0))
            lift = float(obs_state.get("shoulder_lift.pos", -99.0))
            if abs(pan) <= 14.0 and lift >= 5.0:
                self.visited_vaso_d = True
            elif self.visited_vaso_d and lift <= -28.0:
                self.stop_reason = (
                    f"Cubo rojo en Desechos + desinfección VASO D completada a los {elapsed_s:.1f}s -> retorno a HOME."
                )
                return True
            return False

    def update_vision(self, analysis: dict[str, Any], obs_state: dict[str, Any], elapsed_s: float) -> bool:
        """Update cycle completion state from live overhead OpenCV counts (~3–4 Hz)."""
        with self._lock:
            if self.stop_reason is not None:
                return True
            v_a = int(analysis.get("verdes_en_A", 0))
            r_a = int(analysis.get("rojos_en_A", 0))
            v_b = int(analysis.get("verdes_en_B_SUP", 0))
            r_b = int(analysis.get("rojos_en_B_INF", 0))
            v_lost = int(analysis.get("verdes_perdidos", 0))
            r_lost = int(analysis.get("rojos_perdidos", 0))
            pan = float(obs_state.get("shoulder_pan.pos", 0.0))
            lift = float(obs_state.get("shoulder_lift.pos", 0.0))

            if self.target_key == "sano":
                if elapsed_s < 7.0:
                    return False
                placed = (v_b >= self.base_v_b + 1) and (v_a < self.base_v_a)
                lost_out = (v_lost >= self.base_v_lost + 1) and (v_a < self.base_v_a)
                arm_clear = (lift <= -25.0) or (abs(pan) <= 20.0)
                if (placed or lost_out) and arm_clear:
                    self._sano_stable += 1
                    if self._sano_stable >= 3:
                        if lost_out and not placed:
                            self.lost_outside_tape = True
                            self.stop_reason = (
                                f"Cubo verde salió de la cinta a los {elapsed_s:.1f}s -> considerado perdido, retorno a HOME."
                            )
                        else:
                            self.stop_reason = (
                                f"Cubo verde confirmado en Almacén ({v_b}) a los {elapsed_s:.1f}s -> retorno a HOME."
                            )
                        return True
                else:
                    self._sano_stable = 0
                return False

            # target_key == "contaminado"
            if elapsed_s < 10.0:
                return False
            r_b_area = float(analysis.get("rojos_area_en_B_INF", 0.0))
            merged_drop = (
                self._has_area_baseline
                and self.base_r_b > 0
                and r_b >= 1
                and "rojos_area_en_B_INF" in analysis
                and r_b_area >= self.base_r_b_area + 250.0
            )
            lost_red_out = (r_lost >= self.base_r_lost + 1) and (r_a < self.base_r_a)
            if (r_b >= self.base_r_b + 1 or merged_drop or lost_red_out) and (r_a < self.base_r_a):
                self._desechos_stable += 1
                if self._desechos_stable >= 2:
                    self.dropped_in_desechos = True
                    if lost_red_out and r_b < self.base_r_b + 1 and not merged_drop:
                        self.lost_outside_tape = True
            elif not self.dropped_in_desechos:
                self._desechos_stable = 0
            return False


def render_hud(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    analysis: dict[str, Any],
    active_roi: str | None = None,
    footer_hint: str | None = None,
) -> np.ndarray:
    """Draw ROIs, cube bounding boxes/centroids, and jury-ready top HUD banner."""
    canvas = frame_bgr.copy()
    rois = config["rois"]
    roi_polygons = config.get("roi_polygons") or {}

    for name in ROI_NAMES:
        box = rois.get(name)
        poly = roi_polygons.get(name)
        if not box and not (poly and len(poly) >= 3):
            continue
        color = ROI_COLORS_BGR.get(name, (200, 200, 200))
        thickness = 3 if active_roi == name else 2
        if poly and len(poly) >= 3:
            pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, color, thickness)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x_lbl, y_top, y_bot = min(xs), min(ys), max(ys)
        else:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            x_lbl, y_top, y_bot = min(x1, x2), min(y1, y2), max(y1, y2)
        label = f"{name} [ACTIVA]" if active_roi == name else name
        label_y = (y_bot - 8) if y_top < 80 else (y_top - 6)
        cv2.putText(canvas, label, (x_lbl + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

    for cube in analysis["cubes"]:
        x, y, w, h = cube.bbox
        cx, cy = cube.centroid
        is_lost = cube.roi is None
        box_color = (0, 215, 255) if is_lost else ((0, 255, 0) if cube.color == "S" else (0, 0, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)
        tag = f"{cube.color} PERDIDO ({cx},{cy})" if is_lost else f"{cube.color} ({cx},{cy})"
        cv2.putText(canvas, tag, (x, max(y - 5, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 2)

    # Top dark HUD banner
    h_img, w_img = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, 62), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    line1 = f"ZONA A: {analysis['verdes_en_A']} Sanos (S) | {analysis['rojos_en_A']} Contaminados (C)"
    line2 = f"ZONA B SUP (Sanos): {analysis['verdes_en_B_SUP']} | ZONA B INF (Desechos): {analysis['rojos_en_B_INF']}"
    line3 = f"SIGUIENTE ORDEN: [{analysis['next_order']}]"
    order_color = (0, 255, 120) if analysis["target"] == "sano" else ((0, 140, 255) if analysis["target"] == "contaminado" else (255, 230, 0))

    cv2.putText(canvas, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(canvas, line2, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
    cv2.putText(canvas, line3, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, order_color, 2)

    if footer_hint:
        cv2.rectangle(canvas, (0, h_img - 26), (w_img, h_img), (20, 20, 20), -1)
        cv2.putText(canvas, footer_hint, (8, h_img - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1)

    return canvas


def open_camera_and_capture(
    cam_index: int,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 15,
) -> np.ndarray:
    """Open top camera, discard warmup frames for AE stability, read frame, and ALWAYS release."""
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara top índice {cam_index}.")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))

        frame: np.ndarray | None = None
        for _ in range(max(1, warmup_frames)):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
            time.sleep(0.02)

        if frame is None:
            raise RuntimeError(f"Cámara top índice {cam_index} abierta pero no devolvió frames válidos.")
        return frame
    finally:
        # Crítico en macOS AVFoundation: liberar el dispositivo antes de lanzar lerobot-rollout
        cap.release()


def run_calibrate_mode(config_path: Path, cam_top_idx: int | None) -> int:
    """Interactive OpenCV ROI drawer + HSV trackbar calibrator."""
    config = load_vision_config(config_path, cam_top_idx=cam_top_idx)
    idx = int(config["camera_top_index"])
    width, height = int(config["width"]), int(config["height"])

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara top índice {idx}.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    win_main = "Calibrador Triage OpenCV (1-4: ROI | s: Guardar | m: Mascara | q: Salir)"
    win_ctrl = "Controles HSV & Area"
    cv2.namedWindow(win_main)
    cv2.namedWindow(win_ctrl)

    state: dict[str, Any] = {
        "active_roi": "ZONA_A",
        "drawing": False,
        "start_pt": None,
        "show_mask": False,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        roi_key = state["active_roi"]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start_pt"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"] and state["start_pt"]:
            x0, y0 = state["start_pt"]
            config["rois"][roi_key] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"] and state["start_pt"]:
            x0, y0 = state["start_pt"]
            state["drawing"] = False
            if abs(x - x0) > 8 and abs(y - y0) > 8:
                config["rois"][roi_key] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]

    cv2.setMouseCallback(win_main, on_mouse)

    noop = lambda _v: None
    g, r = config["hsv_green"], config["hsv_red"]
    cv2.createTrackbar("G H_min", win_ctrl, int(g["h_min"]), 180, noop)
    cv2.createTrackbar("G H_max", win_ctrl, int(g["h_max"]), 180, noop)
    cv2.createTrackbar("G S_min", win_ctrl, int(g["s_min"]), 255, noop)
    cv2.createTrackbar("G V_min", win_ctrl, int(g["v_min"]), 255, noop)
    cv2.createTrackbar("R H1_max", win_ctrl, int(r["h1_max"]), 30, noop)
    cv2.createTrackbar("R H2_min", win_ctrl, int(r["h2_min"]), 180, noop)
    cv2.createTrackbar("R S_min", win_ctrl, int(r["s_min"]), 255, noop)
    cv2.createTrackbar("R V_min", win_ctrl, int(r["v_min"]), 255, noop)
    cv2.createTrackbar("Area Min", win_ctrl, int(config["min_area_px"]), 5000, noop)
    cv2.createTrackbar("Area Max", win_ctrl, int(config["max_area_px"]), 30000, noop)

    print("==================================================================")
    print("  CALIBRADOR OPENCV — ZERO-CROSS-CONTAMINATION TRIAGE")
    print("  Teclas: [1] ZONA_A | [2] ZONA_B_SUP | [3] ZONA_B_INF | [4] VASO_D")
    print("          Arrastra con raton para dibujar la ROI activa")
    print("          [m] Ver mascara | [s] Guardar JSON | [q] Salir")
    print("==================================================================")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            config["hsv_green"]["h_min"] = cv2.getTrackbarPos("G H_min", win_ctrl)
            config["hsv_green"]["h_max"] = max(config["hsv_green"]["h_min"], cv2.getTrackbarPos("G H_max", win_ctrl))
            config["hsv_green"]["s_min"] = cv2.getTrackbarPos("G S_min", win_ctrl)
            config["hsv_green"]["v_min"] = cv2.getTrackbarPos("G V_min", win_ctrl)
            config["hsv_red"]["h1_max"] = cv2.getTrackbarPos("R H1_max", win_ctrl)
            config["hsv_red"]["h2_min"] = cv2.getTrackbarPos("R H2_min", win_ctrl)
            config["hsv_red"]["s_min"] = cv2.getTrackbarPos("R S_min", win_ctrl)
            config["hsv_red"]["v_min"] = cv2.getTrackbarPos("R V_min", win_ctrl)
            config["min_area_px"] = max(20, cv2.getTrackbarPos("Area Min", win_ctrl))
            config["max_area_px"] = max(config["min_area_px"] + 100, cv2.getTrackbarPos("Area Max", win_ctrl))

            analysis = analyze_frame(frame, config)
            hint = f"ROI Activa: {state['active_roi']} | 1:A 2:B_SUP 3:B_INF 4:VASO | s:Guardar m:Mascara q:Salir"
            hud = render_hud(frame, config, analysis, active_roi=state["active_roi"], footer_hint=hint)

            if state["show_mask"]:
                mask_vis = np.zeros_like(frame)
                mask_vis[analysis["mask_green"] > 0] = (0, 255, 0)
                mask_vis[analysis["mask_red"] > 0] = (0, 0, 255)
                hud = cv2.addWeighted(hud, 0.55, mask_vis, 0.45, 0)

            cv2.imshow(win_main, hud)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("1"):
                state["active_roi"] = "ZONA_A"
            elif key == ord("2"):
                state["active_roi"] = "ZONA_B_SUP"
            elif key == ord("3"):
                state["active_roi"] = "ZONA_B_INF"
            elif key == ord("4"):
                state["active_roi"] = "VASO_D"
            elif key in (ord("m"), ord("M")):
                state["show_mask"] = not state["show_mask"]
            elif key in (ord("s"), ord("S")):
                save_vision_config(config, config_path)
                print(f"✅ Configuración guardada en: {config_path}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_check_mode(config_path: Path, cam_top_idx: int | None, once: bool = False) -> int:
    """Live HUD verification without moving the robot."""
    config = load_vision_config(config_path, cam_top_idx=cam_top_idx)
    idx = int(config["camera_top_index"])

    if once:
        frame = open_camera_and_capture(idx, config["width"], config["height"])
        analysis = analyze_frame(frame, config)
        scene_map = build_scene_map(analysis)
        save_scene_map(scene_map, DEFAULT_SCENE_MAP_PATH)
        hud = render_hud(frame, config, analysis)
        DEFAULT_HUD_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEFAULT_HUD_SNAPSHOT_PATH), hud)
        print(f"ZONA A: {analysis['verdes_en_A']} Sanos (S) | {analysis['rojos_en_A']} Contaminados (C)")
        print(f"SIGUIENTE ORDEN: {analysis['next_order']}")
        print(f"Mapa de coordenadas: {DEFAULT_SCENE_MAP_PATH}")
        print(f"Snapshot guardado en: {DEFAULT_HUD_SNAPSHOT_PATH}")
        return 0

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara top índice {idx}.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
    cap.set(cv2.CAP_PROP_FPS, 30)

    win = "Triage HUD Live Check (q: Salir)"
    cv2.namedWindow(win)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            analysis = analyze_frame(frame, config)
            hud = render_hud(frame, config, analysis, footer_hint="Modo CHECK (sin mover robot) | q: Salir")
            cv2.imshow(win, hud)
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_map_mode(
    config_path: Path,
    cam_top_idx: int | None,
    map_path: Path = DEFAULT_SCENE_MAP_PATH,
    hud_path: Path = DEFAULT_HUD_SNAPSHOT_PATH,
) -> int:
    """Capture a clean overhead frame, map all cube coordinates per ROI, and persist JSON + HUD."""
    config = load_vision_config(config_path, cam_top_idx=cam_top_idx)
    idx = int(config["camera_top_index"])
    frame = open_camera_and_capture(idx, int(config["width"]), int(config["height"]))
    analysis = analyze_frame(frame, config)
    scene_map = build_scene_map(analysis)
    save_scene_map(scene_map, map_path)
    hud = render_hud(frame, config, analysis, footer_hint=f"MAPA ACTUALIZADO ({len(scene_map['objects'])} objetos)")
    hud_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(hud_path), hud)
    print(f"[MAPA] ZONA A: {analysis['verdes_en_A']} Sanos (S) | {analysis['rojos_en_A']} Contaminados (C)")
    for obj in scene_map["objects"]:
        cx, cy = obj["centroid"]
        print(f"  - {obj['color']} en {obj['roi']}: centroid=({cx}, {cy}), area={obj['area']} px")
    print(f"[MAPA] Guardado en: {map_path}")
    print(f"[MAPA] HUD guardado en: {hud_path}")
    return 0


def run_align_mode(config_path: Path, cam_top_idx: int | None, reference_path: Path = DEFAULT_REFERENCE_FRAME_PATH) -> int:
    """Live ghost overlay comparing current top camera to the training reference frame."""
    if not reference_path.exists():
        print(f"[ERROR] No existe imagen de referencia: {reference_path}", file=sys.stderr)
        return 1
    ref_img = cv2.imread(str(reference_path))
    if ref_img is None:
        print(f"[ERROR] No se pudo leer imagen de referencia: {reference_path}", file=sys.stderr)
        return 1

    config = load_vision_config(config_path, cam_top_idx=cam_top_idx)
    idx = int(config["camera_top_index"])
    ref_resized = cv2.resize(ref_img, (int(config["width"]), int(config["height"])))

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara top índice {idx}.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
    cap.set(cv2.CAP_PROP_FPS, 30)

    win = "Alineacion Camara Top vs Entrenamiento (q: Salir)"
    cv2.namedWindow(win)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            blended = cv2.addWeighted(frame, 0.5, ref_resized, 0.5, 0)
            cv2.putText(
                blended,
                "Alinea cintas negras y base naranja del robot hasta que coincidan (q: Salir)",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                2,
            )
            cv2.imshow(win, blended)
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _ensure_lerobot_on_path() -> None:
    lerobot_src = str(REPO_ROOT / "lerobot" / "src")
    if lerobot_src not in sys.path:
        sys.path.insert(0, lerobot_src)


def _move_connected_robot_to_pose(
    robot: Any,
    target_pose: dict[str, float] | None = None,
    duration_s: float = 2.0,
    fps: int = 50,
    label: str = "HOME",
    lift_first: bool = True,
) -> None:
    """Smoothly interpolate an already-connected SO-101 Follower to `target_pose`.

    When `lift_first` is enabled and the arm is lowered over the table (`shoulder_lift > -35`),
    raises shoulder_lift/elbow vertically first before sweeping shoulder_pan back to HOME.
    """
    target = target_pose or DEFAULT_HOME_POSE
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in target}

    def _interp_segment(pose_a: dict[str, float], pose_b: dict[str, float], seg_s: float) -> None:
        seg_steps = max(int(seg_s * fps), 1)
        for step in range(1, seg_steps + 1):
            t = step / seg_steps
            interp = {k: pose_a[k] * (1.0 - t) + pose_b[k] * t for k in target}
            robot.send_action(interp)
            time.sleep(1.0 / fps)

    if lift_first and start.get("shoulder_lift.pos", -99.0) > -35.0:
        lift_waypoint = dict(start)
        lift_waypoint["shoulder_lift.pos"] = min(target["shoulder_lift.pos"], -45.0)
        lift_waypoint["elbow_flex.pos"] = target["elbow_flex.pos"]
        lift_waypoint["wrist_flex.pos"] = target["wrist_flex.pos"]
        phase1_s = max(duration_s * 0.35, 0.3)
        phase2_s = max(duration_s - phase1_s, 0.4)
        _interp_segment(start, lift_waypoint, phase1_s)
        _interp_segment(lift_waypoint, target, phase2_s)
    else:
        _interp_segment(start, target, duration_s)

    print(f"✅ Brazo colocado en {label}.", flush=True)


def move_robot_to_home(
    robot_port: str,
    robot_id: str,
    duration_s: float = 2.5,
    fps: int = 50,
    keep_torque: bool = True,
    target_pose: dict[str, float] | None = None,
) -> None:
    """Smoothly interpolate the SO-101 Follower to the dataset HOME pose."""
    _ensure_lerobot_on_path()
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

    cfg = SO101FollowerConfig(
        port=robot_port,
        id=robot_id,
        disable_torque_on_disconnect=not keep_torque,
        max_relative_target=15.0,
    )
    robot = SOFollower(cfg)
    robot.connect(calibrate=False)
    try:
        _move_connected_robot_to_pose(
            robot,
            target_pose=target_pose,
            duration_s=duration_s,
            fps=fps,
            label=f"HOME (keep_torque={keep_torque})",
        )
    finally:
        robot.disconnect()


def release_robot_torque(robot_port: str, robot_id: str) -> None:
    """Safely release servo torque when autonomous triage finishes."""
    _ensure_lerobot_on_path()
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

    cfg = SO101FollowerConfig(port=robot_port, id=robot_id, disable_torque_on_disconnect=True)
    robot = SOFollower(cfg)
    robot.connect(calibrate=False)
    robot.disconnect()


class PersistentTriageSession:
    """Keeps both ACT policies preloaded and SO-101 + cameras connected across triage cycles."""

    def __init__(
        self,
        *,
        policy_sano_path: Path,
        policy_contaminado_path: Path,
        robot_port: str,
        robot_id: str,
        robot_type: str,
        cam_top_idx: int,
        cam_wrist_idx: int,
        width: int = 640,
        height: int = 480,
        device: str = "mps",
        max_relative_target: float = 30.0,
        n_action_steps: int = 50,
        n_action_steps_contaminado: int = 60,
        post_home_action_steps: int = 0,
        temporal_ensemble_coeff: float | None = None,
    ) -> None:
        self.policy_paths = {"sano": policy_sano_path, "contaminado": policy_contaminado_path}
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.cam_top_idx = int(cam_top_idx)
        self.cam_wrist_idx = int(cam_wrist_idx)
        self.width = int(width)
        self.height = int(height)
        self.requested_device = device
        self.max_relative_target = float(max_relative_target)
        self.n_action_steps = int(n_action_steps)
        self.n_action_steps_contaminado = int(n_action_steps_contaminado)
        self.post_home_action_steps = int(post_home_action_steps)
        self.temporal_ensemble_coeff = temporal_ensemble_coeff

        self.robot: Any = None
        self.torch_device: Any = None
        self.dataset_features: dict[str, Any] = {}
        self.ordered_action_keys: list[str] = []
        self._models: dict[str, tuple[Any, Any, Any]] = {}

    def start(self) -> None:
        _ensure_lerobot_on_path()
        import torch
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

        self._torch = torch
        self._prepare_obs = prepare_observation_for_inference
        self._build_dataset_frame = build_dataset_frame
        self._obs_str = OBS_STR

        dev = self.requested_device
        if dev == "mps" and not torch.backends.mps.is_available():
            dev = "cpu"
        self.torch_device = torch.device(dev)

        for key, p_path in self.policy_paths.items():
            policy = ACTPolicy.from_pretrained(str(p_path))
            steps_for_key = (
                self.n_action_steps_contaminado
                if key == "contaminado" and self.n_action_steps_contaminado > 0
                else self.n_action_steps
            )
            if self.temporal_ensemble_coeff is not None and self.temporal_ensemble_coeff > 0:
                policy.config.temporal_ensemble_coeff = float(self.temporal_ensemble_coeff)
                policy.config.n_action_steps = 1
            else:
                policy.config.temporal_ensemble_coeff = None
                if steps_for_key > 0:
                    policy.config.n_action_steps = steps_for_key
            policy.to(self.torch_device)
            policy.eval()
            policy.reset()
            pre, post = make_pre_post_processors(
                policy.config,
                pretrained_path=str(p_path),
                preprocessor_overrides={"device_processor": {"device": dev}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
            self._models[key] = (policy, pre, post)

        cfg = SO101FollowerConfig(
            port=self.robot_port,
            id=self.robot_id,
            disable_torque_on_disconnect=True,
            max_relative_target=self.max_relative_target,
            cameras={
                "camera_top": OpenCVCameraConfig(
                    index_or_path=self.cam_top_idx,
                    width=self.width,
                    height=self.height,
                    fps=30,
                ),
                "camera_robot": OpenCVCameraConfig(
                    index_or_path=self.cam_wrist_idx,
                    width=self.width,
                    height=self.height,
                    fps=30,
                ),
            },
        )
        self.robot = SOFollower(cfg)
        self.robot.connect(calibrate=False)

        obs_hw = {
            k: v
            for k, v in self.robot.observation_features.items()
            if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
        }
        self.dataset_features = hw_to_dataset_features(obs_hw, OBS_STR)
        self.ordered_action_keys = [k for k in self.robot.action_features if k.endswith(".pos")]

        # Warm-up de MPS una sola vez al arrancar para evitar latencia en el primer chunk del Ciclo 1
        # Nota: prepare_observation_for_inference muta el dict in-place (ndarray -> Tensor + task/robot_type),
        # por lo que reconstruimos obs_frame0 en cada iteración del warm-up.
        obs0 = self.robot.get_observation()
        with torch.inference_mode():
            for policy, pre, post in self._models.values():
                obs_frame0 = build_dataset_frame(self.dataset_features, obs0, prefix=OBS_STR)
                batch0 = prepare_observation_for_inference(obs_frame0, self.torch_device, "", self.robot_type)
                batch0 = pre(batch0)
                _ = post(policy.select_action(batch0))
                policy.reset()
                pre.reset()
                post.reset()
        print(f"[PERSISTENT] Modelos act_sano + act_contaminado precargados en {dev} y SO-101 conectado.", flush=True)

    def go_home(self, duration_s: float = 1.6) -> None:
        _move_connected_robot_to_pose(self.robot, DEFAULT_HOME_POSE, duration_s=duration_s, label="HOME (persistente)")

    def dip_vaso_d(self, duration_down_s: float = 1.2, hold_s: float = 0.4, duration_up_s: float = 1.2) -> None:
        """Perform deterministic biosecurity dip into VASO D and return safely to HOME."""
        _move_connected_robot_to_pose(
            self.robot,
            DEFAULT_VASO_D_POSE,
            duration_s=duration_down_s,
            label="VASO D (desinfección)",
            lift_first=False,
        )
        time.sleep(hold_s)
        _move_connected_robot_to_pose(
            self.robot,
            DEFAULT_HOME_POSE,
            duration_s=duration_up_s,
            label="HOME (post-desinfección)",
            lift_first=True,
        )

    def capture_top_bgr(self, max_attempts: int = 5) -> np.ndarray:
        time.sleep(0.12)
        cam = self.robot.cameras["camera_top"]
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                _ = cam.async_read(timeout_ms=600)
                time.sleep(0.05)
                rgb = cam.async_read(timeout_ms=600)
            except TypeError:
                # Fallback for mocks/cameras whose async_read() takes no timeout_ms kwarg
                _ = cam.async_read()
                time.sleep(0.05)
                rgb = cam.async_read()
            except (TimeoutError, RuntimeError) as exc:
                last_exc = exc
                time.sleep(0.08 * (attempt + 1))
                continue
            if rgb is not None and isinstance(rgb, np.ndarray) and rgb.size > 0:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            time.sleep(0.08)
        if last_exc is not None:
            raise last_exc
        raise TimeoutError("camera_top no devolvió frame válido tras reintentos.")

    def run_policy(
        self,
        target_key: str,
        duration_s: float,
        fps: int = 30,
        live_frame_cb: Callable[[np.ndarray, dict[str, Any], float], bool] | None = None,
        state_cb: Callable[[dict[str, Any], float], bool] | None = None,
        live_every_n_ticks: int = 8,
    ) -> int:
        policy, pre, post = self._models[target_key]
        policy.reset()
        pre.reset()
        post.reset()

        period_s = 1.0 / float(fps)
        t_start = time.perf_counter()
        ticks = 0
        stop_event = threading.Event()
        worker_stop = threading.Event()
        slot_lock = threading.Lock()
        slot_event = threading.Event()
        latest_slot: list[tuple[np.ndarray, dict[str, Any], float] | None] = [None]

        def _live_worker() -> None:
            while not worker_stop.is_set():
                if not slot_event.wait(timeout=0.2):
                    continue
                slot_event.clear()
                with slot_lock:
                    item = latest_slot[0]
                    latest_slot[0] = None
                if item is None or live_frame_cb is None:
                    continue
                top_rgb, state_snap, el_s = item
                top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
                if live_frame_cb(top_bgr, state_snap, el_s):
                    stop_event.set()
                    break

        worker_thread: threading.Thread | None = None
        if live_frame_cb is not None:
            worker_thread = threading.Thread(target=_live_worker, name="live-opencv-hud", daemon=True)
            worker_thread.start()

        try:
            with self._torch.inference_mode():
                while (time.perf_counter() - t_start) < float(duration_s) and not stop_event.is_set():
                    tick_t0 = time.perf_counter()
                    elapsed_s = tick_t0 - t_start
                    obs = self.robot.get_observation()
                    if state_cb is not None:
                        obs_joints = {k: v for k, v in obs.items() if k.endswith(".pos")}
                        if state_cb(obs_joints, elapsed_s):
                            stop_event.set()
                            break

                    obs_frame = self._build_dataset_frame(self.dataset_features, obs, prefix=self._obs_str)
                    batch = self._prepare_obs(obs_frame, self.torch_device, "", self.robot_type)
                    batch = pre(batch)
                    action_queue = getattr(policy, "_action_queue", None)
                    refilled_now = action_queue is not None and len(action_queue) == 0
                    action_tensor = policy.select_action(batch)
                    if (
                        refilled_now
                        and self.temporal_ensemble_coeff is None
                        and self.post_home_action_steps > 0
                        and ticks > 0
                        and action_queue is not None
                    ):
                        # Chunk 0 usa n_action_steps completos para saltar la pausa en HOME;
                        # los siguientes re-observan cámaras ~2.3x más rápido durante el descenso y agarre.
                        keep_tail = max(self.post_home_action_steps - 1, 0)
                        while len(action_queue) > keep_tail:
                            action_queue.pop()
                    action_out = post(action_tensor).squeeze(0).cpu()
                    action_dict = {k: float(action_out[i].item()) for i, k in enumerate(self.ordered_action_keys)}
                    self.robot.send_action(action_dict)

                    if worker_thread is not None and (ticks % max(int(live_every_n_ticks), 1) == 0):
                        top_raw = obs.get("camera_top")
                        if isinstance(top_raw, np.ndarray):
                            obs_joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
                            with slot_lock:
                                latest_slot[0] = (top_raw.copy(), obs_joints, elapsed_s)
                            slot_event.set()

                    ticks += 1
                    elapsed_tick = time.perf_counter() - tick_t0
                    remain = period_s - elapsed_tick
                    if remain > 0:
                        time.sleep(remain)
        finally:
            worker_stop.set()
            slot_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=1.5)

        measured_s = max(time.perf_counter() - t_start, 1e-3)
        print(f"[ROLLOUT {target_key}] {ticks} ticks en {measured_s:.1f}s ({ticks / measured_s:.1f} Hz)", flush=True)
        return 0

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None


def build_rollout_command(
    policy_path: Path,
    duration_s: float,
    robot_port: str,
    robot_id: str,
    robot_type: str,
    cam_top_idx: int,
    cam_wrist_idx: int,
    device: str = "mps",
    max_relative_target: float = 30.0,
    n_action_steps: int = 50,
    temporal_ensemble_coeff: float | None = None,
    disable_torque_on_disconnect: bool = False,
) -> list[str]:
    """Build exact `lerobot-rollout` CLI command matching dataset camera keys (`camera_top`, `camera_robot`)."""
    cameras_arg = (
        "{ "
        f"camera_top: {{type: opencv, index_or_path: {int(cam_top_idx)}, width: 640, height: 480, fps: 30}}, "
        f"camera_robot: {{type: opencv, index_or_path: {int(cam_wrist_idx)}, width: 640, height: 480, fps: 30}} "
        "}"
    )
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_rollout",
        "--strategy.type=base",
        f"--robot.type={robot_type}",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        f"--robot.max_relative_target={max_relative_target}",
        f"--robot.disable_torque_on_disconnect={'true' if disable_torque_on_disconnect else 'false'}",
        f"--robot.cameras={cameras_arg}",
        f"--policy.path={policy_path}",
        f"--policy.device={device}",
        f"--device={device}",
        f"--duration={duration_s}",
        "--display_data=false",
        "--return_to_initial_position=true",
    ]
    if temporal_ensemble_coeff is not None and temporal_ensemble_coeff > 0:
        cmd.append(f"--policy.temporal_ensemble_coeff={temporal_ensemble_coeff}")
        cmd.append("--policy.n_action_steps=1")
    elif n_action_steps > 0:
        cmd.append(f"--policy.n_action_steps={int(n_action_steps)}")
    return cmd


class _StopRequested(BaseException):
    """Raised on SIGTERM/SIGINT so cleanup (child rollout stop + torque release) executes deterministically."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"signal {signum}")


def _terminate_process_cleanly(proc: subprocess.Popen[Any], wait_timeout_s: float = 0.6) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=wait_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass


def _default_rollout_runner(cmd: list[str]) -> int:
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    lerobot_src = str(REPO_ROOT / "lerobot" / "src")
    env["PYTHONPATH"] = f"{lerobot_src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else lerobot_src
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
    try:
        return int(proc.wait())
    except BaseException:
        _terminate_process_cleanly(proc)
        raise


def run_autonomous_supervisor(
    config: dict[str, Any],
    policy_sano: Path,
    policy_contaminado: Path,
    robot_port: str,
    robot_id: str,
    robot_type: str,
    cam_wrist_idx: int,
    duration_sano: float = 18.0,
    duration_contaminado: float = 27.0,
    max_retries: int = 2,
    max_cycles: int = 12,
    warmup_frames: int = 15,
    dry_run: bool = False,
    go_home: bool = False,
    max_relative_target: float = 30.0,
    n_action_steps: int = 50,
    n_action_steps_contaminado: int = 60,
    post_home_action_steps: int = 0,
    temporal_ensemble_coeff: float | None = None,
    remap_on_change: bool = True,
    remap_threshold_px: float = DEFAULT_REMAP_THRESHOLD_PX,
    pause_on_remap: bool = False,
    persistent: bool = True,
    scene_map_path: Path = DEFAULT_SCENE_MAP_PATH,
    scanner_fn: Callable[[], dict[str, Any]] | None = None,
    runner_fn: Callable[[list[str]], int] | None = None,
    hud_path: Path = DEFAULT_HUD_SNAPSHOT_PATH,
) -> int:
    """Closed-loop Clean-First triage orchestrator with macOS camera release, scene remapping, and auto-retry."""
    cam_top_idx = int(config["camera_top_index"])
    is_mocked = scanner_fn is not None or runner_fn is not None
    use_persistent = persistent and not dry_run and not is_mocked

    session: PersistentTriageSession | None = None
    if use_persistent:
        for label_k, p_path in (("sano", policy_sano), ("contaminado", policy_contaminado)):
            if not p_path.exists():
                print(f"[ERROR] Checkpoint no encontrado para '{label_k}': {p_path}", file=sys.stderr)
                return 2
        session = PersistentTriageSession(
            policy_sano_path=policy_sano,
            policy_contaminado_path=policy_contaminado,
            robot_port=robot_port,
            robot_id=robot_id,
            robot_type=robot_type,
            cam_top_idx=cam_top_idx,
            cam_wrist_idx=cam_wrist_idx,
            width=int(config["width"]),
            height=int(config["height"]),
            max_relative_target=max_relative_target,
            n_action_steps=n_action_steps,
            n_action_steps_contaminado=n_action_steps_contaminado,
            post_home_action_steps=post_home_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )

    def scan_scene(footer_hint: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if scanner_fn is not None:
            analysis = scanner_fn()
            return analysis, build_scene_map(analysis)
        if session is not None and session.robot is not None:
            frame = session.capture_top_bgr()
        else:
            frame = open_camera_and_capture(
                cam_index=cam_top_idx,
                width=int(config["width"]),
                height=int(config["height"]),
                warmup_frames=warmup_frames,
            )
        analysis = analyze_frame(frame, config)
        scene_map = build_scene_map(analysis)
        save_scene_map(scene_map, scene_map_path)
        hud = render_hud(frame, config, analysis, footer_hint=footer_hint)
        _save_hud_atomic(hud_path, hud)
        return analysis, scene_map

    execute_rollout = runner_fn or _default_rollout_runner
    retries_by_target = {"sano": 0, "contaminado": 0}
    lost_counts = {"sano": 0, "contaminado": 0}
    last_map: dict[str, Any] | None = None
    release_on_finally = (not dry_run and not is_mocked) and session is None

    is_main_thread = threading.current_thread() is threading.main_thread()
    prev_sigterm = signal.getsignal(signal.SIGTERM) if is_main_thread else None
    prev_sigint = signal.getsignal(signal.SIGINT) if is_main_thread else None

    def _on_stop_signal(signum: int, _frame: Any) -> None:
        raise _StopRequested(signum)

    if is_main_thread:
        signal.signal(signal.SIGTERM, _on_stop_signal)
        signal.signal(signal.SIGINT, _on_stop_signal)

    try:
        if session is not None:
            session.start()
            if go_home:
                session.go_home(duration_s=1.8)
        elif go_home and not dry_run and not is_mocked:
            move_robot_to_home(robot_port=robot_port, robot_id=robot_id, keep_torque=True)

        for cycle in range(1, max_cycles + 1):
            analysis, curr_map = scan_scene()
            if last_map is None and not is_mocked:
                coords = ", ".join(f"{o['color']}({o['centroid'][0]},{o['centroid'][1]})" for o in curr_map["objects"] if o["roi"] == "ZONA_A")
                print(f"[MAPA INICIAL] ZONA A -> {coords or 'vacía'} (guardado en {scene_map_path})")
            last_map = curr_map

            v_count = int(analysis["verdes_en_A"])
            r_count = int(analysis["rojos_en_A"])
            target = analysis["target"]

            print(f"[CICLO {cycle}] ZONA A: {v_count} Sanos (S) | {r_count} Contaminados (C) -> {analysis['next_order']}")

            if target == "completed":
                total_lost = lost_counts["sano"] + lost_counts["contaminado"]
                if total_lost > 0:
                    print(
                        "✅ MISIÓN COMPLETADA: ZONA A DESPEJADA "
                        f"(cubos perdidos fuera de la cinta omitidos: S={lost_counts['sano']}, C={lost_counts['contaminado']})"
                    )
                else:
                    print("✅ MISIÓN COMPLETADA: ZONA A DESPEJADA Y CLASIFICADA SIN CONTAMINACIÓN CRUZADA")
                return 0

            if target == "sano":
                policy_path = policy_sano
                duration_s = duration_sano
                prev_count = v_count
                count_key = "verdes_en_A"
                dest_key = "verdes_en_B_SUP"
            else:
                policy_path = policy_contaminado
                duration_s = duration_contaminado
                prev_count = r_count
                count_key = "rojos_en_A"
                dest_key = "rojos_en_B_INF"
            prev_dest_count = int(analysis.get(dest_key, 0))
            prev_dest_area = float(analysis.get("rojos_area_en_B_INF", 0.0)) if target == "contaminado" else 0.0

            if not dry_run and not policy_path.exists():
                print(f"[ERROR] Checkpoint no encontrado para '{target}': {policy_path}", file=sys.stderr)
                return 2

            rollout_steps = (
                n_action_steps_contaminado
                if target == "contaminado" and n_action_steps_contaminado > 0
                else n_action_steps
            )
            cmd = build_rollout_command(
                policy_path=policy_path,
                duration_s=duration_s,
                robot_port=robot_port,
                robot_id=robot_id,
                robot_type=robot_type,
                cam_top_idx=cam_top_idx,
                cam_wrist_idx=cam_wrist_idx,
                max_relative_target=max_relative_target,
                n_action_steps=rollout_steps,
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                disable_torque_on_disconnect=False,
            )

            if dry_run:
                print(f"[DRY-RUN] Ejecutaría: {' '.join(cmd)}")
                return 0

            tracker_ref: EarlyStopTracker | None = None
            if session is not None:
                tracker_counts = dict(curr_map["counts"])
                for extra_k in ("rojos_area_en_B_INF", "verdes_perdidos", "rojos_perdidos"):
                    if extra_k in analysis:
                        tracker_counts[extra_k] = analysis[extra_k]
                tracker = EarlyStopTracker(target_key=target, baseline_counts=tracker_counts)
                tracker_ref = tracker

                def _on_live_frame(
                    top_bgr: np.ndarray,
                    obs_state: dict[str, Any],
                    elapsed_s: float,
                    *,
                    _cycle: int = cycle,
                    _target: str = target,
                    _tracker: EarlyStopTracker = tracker,
                ) -> bool:
                    live_an = analyze_frame(top_bgr, config)
                    live_map = build_scene_map(live_an)
                    save_scene_map(live_map, scene_map_path)
                    hint = f"EN VIVO | Ciclo {_cycle} ({elapsed_s:.1f}s) | {_target}"
                    live_hud = render_hud(top_bgr, config, live_an, footer_hint=hint)
                    _save_hud_atomic(hud_path, live_hud)
                    return _tracker.update_vision(live_an, obs_state, elapsed_s)

                rc = session.run_policy(
                    target,
                    duration_s=duration_s,
                    live_frame_cb=_on_live_frame,
                    state_cb=tracker.observe_robot_state,
                )
                if tracker.stop_reason:
                    print(f"[EARLY-STOP {target}] {tracker.stop_reason}", flush=True)
            else:
                rc = execute_rollout(cmd)
            if rc != 0:
                print(f"[ERROR] lerobot-rollout devolvió código {rc}.", file=sys.stderr)
                return rc

            # Asegurar brazo en HOME despejado antes de re-mapear coordenadas tras el rollout
            if session is not None and go_home:
                session.go_home(duration_s=1.8)
            elif go_home and not is_mocked:
                move_robot_to_home(robot_port=robot_port, robot_id=robot_id, duration_s=1.8, keep_torque=True)
                time.sleep(0.25)

            post_analysis, post_map = scan_scene()
            scene_change = detect_scene_change(curr_map, post_map, move_threshold_px=remap_threshold_px)
            last_map = post_map

            if scene_change["changed"]:
                print(f"[MAPA] Escenario actualizado: {scene_change['summary']}")

            new_count = int(post_analysis[count_key])
            if new_count < prev_count:
                delta_left = prev_count - new_count
                post_dest_count = int(post_analysis.get(dest_key, 0))
                post_dest_area = float(post_analysis.get("rojos_area_en_B_INF", 0.0)) if target == "contaminado" else 0.0
                merged_dest_drop = (
                    target == "contaminado"
                    and prev_dest_count > 0
                    and post_dest_count >= 1
                    and post_dest_area >= prev_dest_area + 250.0
                )
                has_dest_counts = (dest_key in post_analysis) or (dest_key in analysis)
                evacuated_to_dest = (post_dest_count > prev_dest_count) or merged_dest_drop
                if evacuated_to_dest or not has_dest_counts:
                    print(f"[VERIFICACIÓN OK] {target}: {prev_count} -> {new_count} en ZONA_A.")
                else:
                    lost_counts[target] += delta_left
                    print(
                        f"⚠️ [CUBO PERDIDO] {target}: {delta_left} cubo(s) salió de la cinta "
                        f"({prev_count} -> {new_count} en ZONA_A, sin llegar a {dest_key}). "
                        "Se considera bueno/omitido y seguimos con el siguiente."
                    )
                    if target == "contaminado" and new_count == 0:
                        already_disinfected = tracker_ref is not None and tracker_ref.visited_vaso_d
                        if already_disinfected:
                            print("🧼 [BIOSEGURIDAD] Desinfección en VASO D ya completada tras el último rojo.")
                        elif session is not None and not dry_run and not is_mocked:
                            print("🧼 [BIOSEGURIDAD] Último cubo rojo salió de la cinta -> desinfectando en VASO D y terminando ciclo.")
                            session.dip_vaso_d()
                        elif not dry_run and not is_mocked:
                            print("🧼 [BIOSEGURIDAD] Último cubo rojo salió de la cinta -> ejecutando baño en VASO D.")
                            move_robot_to_home(robot_port=robot_port, robot_id=robot_id, duration_s=1.2, keep_torque=True, target_pose=DEFAULT_VASO_D_POSE)
                            time.sleep(0.4)
                            move_robot_to_home(robot_port=robot_port, robot_id=robot_id, duration_s=1.2, keep_torque=True, target_pose=DEFAULT_HOME_POSE)
                retries_by_target[target] = 0
            else:
                fail_hud_path = hud_path.with_name("debug_last_fail.jpg")
                if _archive_fail_snapshot(hud_path, fail_hud_path):
                    print(f"[DEBUG-SNAPSHOT] Fallo guardado en {fail_hud_path}")
                if remap_on_change and scene_change["moved_objects"]:
                    moves = ", ".join(
                        f"{m['color']} {tuple(m['from'])}->{tuple(m['to'])} (Δ={m['shift_px']}px)"
                        for m in scene_change["moved_objects"]
                    )
                    print(f"[REMAP] Objeto desplazado tras mal agarre ({moves}). Escenario re-mapeado en {scene_map_path}.")
                    if not is_mocked:
                        # Refrescar HUD con aviso de remapeo para el próximo ciclo
                        coords = ", ".join(
                            f"{o['color']}({o['centroid'][0]},{o['centroid'][1]})"
                            for o in post_map["objects"]
                            if o["roi"] == "ZONA_A"
                        )
                        print(f"[REMAP] Nuevas coordenadas ZONA A: {coords}")
                    if pause_on_remap:
                        resp = input("[PAUSA REMAP] Escenario re-mapeado. Pulsa Enter para reintentar (o 'q' + Enter para salir): ")
                        if resp.strip().lower() == "q":
                            return 0

                retries_by_target[target] += 1
                if retries_by_target[target] > max_retries:
                    print(
                        f"[FALLO CRÍTICO] Máximo de reintentos ({max_retries}) alcanzado para '{target}' sin evacuar cubo.",
                        file=sys.stderr,
                    )
                    return 3
                print(f"[AUTO-RECOVERY] Reintentando {target} ({retries_by_target[target]}/{max_retries})...")

        print("[AVISO] Límite de ciclos alcanzado sin despejar ZONA_A.", file=sys.stderr)
        return 4
    except _StopRequested as stop_exc:
        print(
            f"[STOP] Señal {stop_exc.signum} recibida -> rollout detenido; liberando torque limpiamente.",
            flush=True,
        )
        return 128 + stop_exc.signum
    finally:
        if is_main_thread:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                print(f"[AVISO] No se pudo cerrar sesión persistente limpiamente: {exc}", file=sys.stderr)
        elif release_on_finally:
            try:
                release_robot_torque(robot_port=robot_port, robot_id=robot_id)
            except Exception as exc:
                print(f"[AVISO] No se pudo liberar torque al salir: {exc}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrador y Supervisor OpenCV para Triage SO-101.")
    parser.add_argument("--mode", choices=["calibrate", "check", "map", "align", "home", "run"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scene-map", type=Path, default=DEFAULT_SCENE_MAP_PATH)
    parser.add_argument("--cam-top", type=int, default=int(os.environ.get("CAM_TOP_IDX", "1")))
    parser.add_argument("--cam-wrist", type=int, default=int(os.environ.get("CAM_WRIST_IDX", "0")))
    parser.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5B610343371"))
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "bilbao_follower"))
    parser.add_argument("--robot-type", default=os.environ.get("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--policy-sano", type=Path, default=DEFAULT_POLICY_SANO)
    parser.add_argument("--policy-contaminado", type=Path, default=DEFAULT_POLICY_CONTAMINADO)
    parser.add_argument("--duration-sano", type=float, default=18.0)
    parser.add_argument("--duration-contaminado", type=float, default=27.0)
    parser.add_argument("--max-relative-target", type=float, default=30.0, help="Límite relativo por paso (30.0 permite cierre rápido de pinza).")
    parser.add_argument("--n-action-steps", type=int, default=50, help="Pasos por chunk para act_sano (50 = ~2.0s).")
    parser.add_argument(
        "--n-action-steps-contaminado",
        type=int,
        default=60,
        help="Pasos por chunk para act_contaminado (60 = ~2.5s, supera la pausa inicial en HOME; 0 usa --n-action-steps).",
    )
    parser.add_argument(
        "--post-home-steps",
        type=int,
        default=0,
        help="Opcional: pasos por chunk tras salir de HOME (0 desactiva y usa --n-action-steps uniforme = 50).",
    )
    parser.add_argument("--temporal-ensemble", type=float, default=0.0, help="Coeficiente Temporal Ensembling ACT (0 desactivado).")
    parser.add_argument("--no-go-home", action="store_true", help="No mover suavemente al HOME del dataset antes/entre ciclos de --mode run.")
    parser.add_argument("--no-remap-on-change", action="store_true", help="Desactivar detección y re-mapeo automático si un objeto cambia de coordenadas.")
    parser.add_argument("--remap-threshold-px", type=float, default=DEFAULT_REMAP_THRESHOLD_PX, help="Desplazamiento mínimo en px para detectar cambio de escenario.")
    parser.add_argument("--pause-on-remap", action="store_true", help="Pausar para confirmación manual si un intento fallido desplazó un objeto.")
    parser.add_argument("--subprocess", action="store_true", help="Usar subproceso lerobot-rollout por cada ciclo en lugar de sesión persistente.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="En --mode check, captura un snapshot y sale.")
    parser.add_argument("--dry-run", action="store_true", help="En --mode run, escanea e imprime el rollout sin mover hardware.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "calibrate":
        return run_calibrate_mode(args.config, cam_top_idx=args.cam_top)
    if args.mode == "check":
        return run_check_mode(args.config, cam_top_idx=args.cam_top, once=args.once)
    if args.mode == "map":
        return run_map_mode(args.config, cam_top_idx=args.cam_top, map_path=args.scene_map)
    if args.mode == "align":
        return run_align_mode(args.config, cam_top_idx=args.cam_top)
    if args.mode == "home":
        move_robot_to_home(robot_port=args.port, robot_id=args.robot_id, keep_torque=True)
        return 0

    config = load_vision_config(args.config, cam_top_idx=args.cam_top)
    te_coeff = args.temporal_ensemble if args.temporal_ensemble and args.temporal_ensemble > 0 else None
    return run_autonomous_supervisor(
        config=config,
        policy_sano=args.policy_sano,
        policy_contaminado=args.policy_contaminado,
        robot_port=args.port,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        cam_wrist_idx=args.cam_wrist,
        duration_sano=args.duration_sano,
        duration_contaminado=args.duration_contaminado,
        max_retries=args.max_retries,
        warmup_frames=args.warmup_frames,
        dry_run=args.dry_run,
        go_home=not args.no_go_home,
        max_relative_target=args.max_relative_target,
        n_action_steps=args.n_action_steps,
        n_action_steps_contaminado=args.n_action_steps_contaminado,
        post_home_action_steps=args.post_home_steps,
        temporal_ensemble_coeff=te_coeff,
        remap_on_change=not args.no_remap_on_change,
        remap_threshold_px=args.remap_threshold_px,
        pause_on_remap=args.pause_on_remap,
        persistent=not args.subprocess,
        scene_map_path=args.scene_map,
    )


if __name__ == "__main__":
    raise SystemExit(main())
