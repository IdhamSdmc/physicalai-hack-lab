#!/usr/bin/env python3
"""
Sprint 4 — Plantilla de Arquitectura Ganadora: Orquestador + Multi-Policy ACT Local.

En un hackathon de 48h, entrenar una única política end-to-end para una tarea larga de varios pasos
suele fallar (compounding error). La estrategia consiste en:
1. Dividir el reto en 2 o 3 políticas ACT cortas y ultra-fiables:
   - policy_pick: acercarse y agarrar el objeto
   - policy_place: llevarlo a la cesta/zona destino
   - policy_recover: volver a posición segura y reintentar si la pinza está vacía
2. Usar una máquina de estados / VLM ligero que supervise qué política activa toca en cada momento
   y detecte atascos leyendo la telemetría del servo #6 (`gripper`, escala LeRobot `RANGE_0_100`).
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class TaskStage(Enum):
    SCAN_SCENE = "scan_scene"
    PICK_OBJECT = "pick_object"
    PLACE_OBJECT = "place_object"
    RECOVER = "recover"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RobotTelemetry:
    joint_positions: np.ndarray  # [6] ángulos en grados (cuerpo) y [0..100] en pinza
    gripper_pos: float           # Posición del servo #6 en escala LeRobot RANGE_0_100 (0..100)
    step_in_stage: int


class HackathonOrchestrator:
    """
    Orquestador híbrido que supervisa la telemetría del brazo SO-101 y conmuta
    entre checkpoints locales de ACT entrenados en el MacBook M5 o nube.
    """

    def __init__(
        self,
        checkpoints: dict[TaskStage, str],
        min_grasp_pos: float = 5.0,
        max_grasp_pos: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.checkpoints = checkpoints
        self.min_grasp_pos = min_grasp_pos
        self.max_grasp_pos = max_grasp_pos
        self.max_retries = max_retries
        self.stage = TaskStage.SCAN_SCENE
        self.stage_step = 0
        self.retries = 0

    def check_gripper_grasped(self, telemetry: RobotTelemetry) -> bool:
        """
        Truco medido en el SO-101 (LeRobot `MotorNormMode.RANGE_0_100`):
        - Pinza cerrada sin objeto: ~1.2..1.4
        - Pinza sujetando bloque/pieza: ~8.0..45.0 (ej. 13.8 con Lego)
        """
        return self.min_grasp_pos <= telemetry.gripper_pos <= self.max_grasp_pos

    def step_supervisor(self, telemetry: RobotTelemetry) -> TaskStage:
        """Advance the state machine by one 30 Hz control tick."""
        self.stage_step += 1

        if self.stage == TaskStage.SCAN_SCENE:
            print("[Orquestador] Escena validada -> Activando política ACT: PICK_OBJECT")
            self.stage = TaskStage.PICK_OBJECT
            self.stage_step = 0

        elif self.stage == TaskStage.PICK_OBJECT:
            # Tras 120 pasos (4 segundos a 30Hz), comprobamos si la pinza tiene el objeto
            if self.stage_step >= 120:
                if self.check_gripper_grasped(telemetry):
                    print("[Orquestador] Agarre confirmado por telemetría de pinza -> PLACE_OBJECT")
                    self.stage = TaskStage.PLACE_OBJECT
                    self.stage_step = 0
                else:
                    print("[Orquestador] Pinza cerrada en vacío (Fallo de agarre) -> RECOVER")
                    self.stage = TaskStage.RECOVER
                    self.stage_step = 0

        elif self.stage == TaskStage.RECOVER:
            if self.stage_step >= 60:
                self.retries += 1
                if self.retries > self.max_retries:
                    print(f"[Orquestador] Máximo de reintentos ({self.max_retries}) alcanzado -> FAILED")
                    self.stage = TaskStage.FAILED
                else:
                    print(f"[Orquestador] Brazo en Home seguro. Reintento #{self.retries} -> PICK_OBJECT")
                    self.stage = TaskStage.PICK_OBJECT
                self.stage_step = 0

        elif self.stage == TaskStage.PLACE_OBJECT:
            if self.stage_step >= 120:
                print("[Orquestador] Objeto depositado con éxito -> COMPLETED")
                self.stage = TaskStage.COMPLETED

        return self.stage


if __name__ == "__main__":
    print("Simulando bucle del Orquestador Híbrido (Multi-Policy + Detección de Agarre en Pinza)...")
    orchestrator = HackathonOrchestrator(
        checkpoints={
            TaskStage.PICK_OBJECT: "outputs/train/act_pick/checkpoints/last/pretrained_model",
            TaskStage.PLACE_OBJECT: "outputs/train/act_place/checkpoints/last/pretrained_model",
            TaskStage.RECOVER: "outputs/train/act_recover/checkpoints/last/pretrained_model",
        }
    )

    # Simulamos un primer intento donde la pinza cierra en vacío (1.3 en escala 0..100)
    # y un segundo intento exitoso sujetando el objeto (13.8 en escala 0..100)
    for step in range(450):
        simulated_gripper = 1.3 if step < 180 else 13.8
        telemetry = RobotTelemetry(
            joint_positions=np.zeros(6),
            gripper_pos=simulated_gripper,
            step_in_stage=orchestrator.stage_step,
        )
        current_stage = orchestrator.step_supervisor(telemetry)
        if current_stage in (TaskStage.COMPLETED, TaskStage.FAILED):
            break
    print(f"Simulación del Orquestador finalizada en estado: {orchestrator.stage.value}")
