#!/usr/bin/env python3
"""
Sprint 4 — Plantilla de Arquitectura Ganadora: Orquestador + Multi-Policy ACT Local.

En un hackathon de 48h, entrenar una única política end-to-end para una tarea larga de 4 pasos
suele fallar (compounding error). La estrategia de arquitectura ganadora consiste en:
1. Dividir el reto en 2 o 3 políticas ACT cortas y ultra-fiables entrenadas en tu M5:
   - policy_pick: acercarse y agarrar el objeto
   - policy_place: llevarlo a la cesta/zona destino
   - policy_recover: volver a posición segura y reintentar si la pinza está vacía
2. Usar una máquina de estados / VLM ligero que supervise qué política activa toca en cada momento
   y detecte atascos (si el brazo no avanza o el torque/corriente de la pinza indica fallo de agarre).
"""

from dataclasses import dataclass
from enum import Enum
import time
import numpy as np


class TaskStage(Enum):
    SCAN_SCENE = "scan_scene"
    PICK_OBJECT = "pick_object"
    PLACE_OBJECT = "place_object"
    RECOVER = "recover"
    COMPLETED = "completed"


@dataclass
class RobotTelemetry:
    joint_positions: np.ndarray  # [6] radianes o grados normalizados del SO-101
    gripper_pos: float           # Posición del servo #6 (pinza)
    step_in_stage: int


class HackathonOrchestrator:
    """
    Orquestador híbrido que supervisa la telemetría del brazo SO-101 y conmuta
    entre checkpoints locales de ACT entrenados en el MacBook M5.
    """

    def __init__(self, checkpoints: dict[TaskStage, str]) -> None:
        self.checkpoints = checkpoints
        self.stage = TaskStage.SCAN_SCENE
        self.stage_step = 0
        self.retries = 0

    def check_gripper_grasped(self, telemetry: RobotTelemetry) -> bool:
        """
        Truco de hackathon en el SO-101:
        Si ordenaste cerrar la pinza a 0.0 pero el servo se queda en ~0.18..0.45,
        significa que tiene un objeto físico atrapado entre los dedos.
        Si llega a <0.03, la pinza cerró en vacío (falló el agarre).
        """
        return 0.08 <= telemetry.gripper_pos <= 0.65

    def step_supervisor(self, telemetry: RobotTelemetry) -> TaskStage:
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

    # Simulamos un primer intento donde falla el agarre (gripper=0.01) y un segundo intento exitoso (gripper=0.25)
    for step in range(450):
        simulated_gripper = 0.01 if step < 180 else 0.25
        telemetry = RobotTelemetry(
            joint_positions=np.zeros(6),
            gripper_pos=simulated_gripper,
            step_in_stage=orchestrator.stage_step,
        )
        current_stage = orchestrator.step_supervisor(telemetry)
        if current_stage == TaskStage.COMPLETED:
            break
    print("Simulación del Orquestador finalizada correctamente.")
