"""Novel text prompt families for open-vocabulary HAR evaluation.

Three prompt families that are **completely disjoint** from the training
templates in ``task_templates.py`` (lines 97-153).  Each family uses a
single text description per class to test the minimal open-vocabulary
setting — no prompt ensembling, no overlap with training text.

Class-ID mapping (matches ``label_mapping.json``):
    0 = jumping, 1 = running, 2 = seated_breathing, 3 = walking, 4 = waving
"""

from __future__ import annotations

from typing import Dict, List

OPEN_VOCAB_HAR_FAMILIES: Dict[str, Dict[int, List[str]]] = {
    "domain_expert": {
        0: ["Bilateral lower-extremity plyometric activity with vertical displacement"],
        1: ["Sustained locomotion at high cadence with aerial phase between strides"],
        2: ["Resting tidal ventilation in a seated posture with minimal postural sway"],
        3: ["Bipedal ambulation at self-selected comfortable velocity"],
        4: ["Unilateral repetitive upper-extremity abduction-adduction in the coronal plane"],
    },
    "layperson": {
        0: ["Hopping on the spot like on a trampoline"],
        1: ["Going for a quick jog around the room"],
        2: ["Just chilling in a chair doing nothing"],
        3: ["Casually strolling from one side to the other"],
        4: ["Flapping a hand in the air to say hi"],
    },
    "sensor_grounded": {
        0: ["Periodic vertical body displacement causing sharp multipath length changes"],
        1: ["Rapid translational body movement inducing large Doppler spread across subcarriers"],
        2: ["Quasi-static scene with only subtle thoracic motion modulating the channel"],
        3: ["Moderate-speed translational displacement with regular bipedal gait periodicity"],
        4: ["Isolated limb oscillation superimposed on an otherwise stationary torso"],
    },
}

FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "domain_expert": "Clinical / biomechanics vocabulary",
    "layperson": "Simple everyday language",
    "sensor_grounded": "Physics of what the CSI signal measures",
}
