"""Novel text prompt families for open-vocabulary HAR evaluation.

All prompt families are **completely disjoint** from the training templates
in ``task_templates.py`` (lines 97-153).

Families are organized in three tiers:

1. **Minimal** (1 template / class, raw specialist text) — ``domain_expert``,
   ``layperson``, ``sensor_grounded``.
2. **Anchored** (1 template / class, specialist text wrapped in a natural-
   language frame) — ``domain_expert_anchored``, ``sensor_grounded_anchored``.
3. **Ensemble** (5 templates / class) — ``domain_expert_ensemble``,
   ``layperson_ensemble``, ``sensor_grounded_ensemble``.

Class-ID mapping (matches ``label_mapping.json``):
    0 = jumping, 1 = running, 2 = seated_breathing, 3 = walking, 4 = waving
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Tier 1 — Minimal (1 template per class, raw)
# ---------------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Tier 2 — Anchored (1 template per class, with "A person …" frame)
    # -------------------------------------------------------------------
    "domain_expert_anchored": {
        0: ["A person performing bilateral plyometric jumping with vertical displacement"],
        1: ["A person in sustained high-cadence running with aerial phases between strides"],
        2: ["A person seated with resting tidal ventilation and minimal postural sway"],
        3: ["A person in bipedal ambulation at comfortable walking velocity"],
        4: ["A person performing repetitive upper-extremity waving in the coronal plane"],
    },
    "sensor_grounded_anchored": {
        0: ["A person causing periodic vertical body displacement detected by the wireless sensor"],
        1: ["A person causing rapid translational movement inducing wireless signal fluctuations"],
        2: ["A person sitting still with only subtle breathing motion detectable by the sensor"],
        3: ["A person causing moderate-speed displacement with regular walking periodicity in the signal"],
        4: ["A person causing isolated arm oscillation detectable against an otherwise stationary body"],
    },

    # -------------------------------------------------------------------
    # Tier 3 — Ensemble (5 templates per class)
    # -------------------------------------------------------------------
    "domain_expert_ensemble": {
        0: [
            "A person performing bilateral plyometric jumping with vertical displacement",
            "Someone executing repetitive countermovement jumps in a stationary position",
            "An individual doing high-impact vertical hopping exercises",
            "A person engaged in standing jump training with cyclic takeoff and landing",
            "Someone practicing plyometric bounding with peak vertical excursion",
        ],
        1: [
            "A person in sustained high-cadence running with aerial phases between strides",
            "Someone performing vigorous running gait at elevated stride frequency",
            "An individual engaged in high-velocity bipedal locomotion with flight phase",
            "A person executing a running pattern with rapid footstrike cadence",
            "Someone demonstrating aerobic running with sustained bilateral flight intervals",
        ],
        2: [
            "A person seated with resting tidal ventilation and minimal postural sway",
            "Someone in quiet seated respiration at baseline metabolic rate",
            "An individual sitting still with only respiratory chest excursion visible",
            "A person maintaining sedentary posture with eupneic breathing rhythm",
            "Someone seated and breathing normally with negligible body movement",
        ],
        3: [
            "A person in bipedal ambulation at comfortable walking velocity",
            "Someone performing steady walking gait with reciprocal limb advancement",
            "An individual walking at self-selected preferred pace",
            "A person demonstrating habitual ambulatory locomotion at moderate speed",
            "Someone engaged in level-ground walking with symmetric stride pattern",
        ],
        4: [
            "A person performing repetitive upper-extremity waving in the coronal plane",
            "Someone executing rhythmic arm elevation with hand oscillation",
            "An individual making a deliberate hand-waving greeting gesture",
            "A person demonstrating periodic arm motion in a waving pattern",
            "Someone performing voluntary repetitive hand and arm waving",
        ],
    },
    "layperson_ensemble": {
        0: [
            "Hopping on the spot like on a trampoline",
            "Someone is bouncing up and down on both feet",
            "A person doing little hops in one place",
            "Somebody keeps jumping up and landing again",
            "A person leaping off the floor over and over",
        ],
        1: [
            "Going for a quick jog around the room",
            "Someone is dashing across the area at speed",
            "A person running fast from one end to the other",
            "Somebody is hurrying along at a run",
            "A person moving their legs really fast like a jogger",
        ],
        2: [
            "Just chilling in a chair doing nothing",
            "Someone is sitting down and relaxing quietly",
            "A person resting in a seat, barely moving at all",
            "Somebody is parked in a chair just breathing normally",
            "A person sitting totally still, just existing",
        ],
        3: [
            "Casually strolling from one side to the other",
            "Someone taking a calm walk across the room",
            "A person ambling along at a normal speed",
            "Somebody is wandering around at a relaxed pace",
            "A person walking slowly and steadily forward",
        ],
        4: [
            "Flapping a hand in the air to say hi",
            "Someone is waving their hand back and forth overhead",
            "A person swinging their arm to wave goodbye",
            "Somebody is shaking their hand in the air like a greeting",
            "A person moving their arm side to side to wave",
        ],
    },
    "sensor_grounded_ensemble": {
        0: [
            "A person causing periodic vertical body displacement detected by the wireless sensor",
            "Repeated sharp changes in wireless signal from someone bouncing vertically",
            "Cyclic body elevation and descent producing strong multipath signal variation",
            "Someone creating rhythmic vertical motion that disrupts the wireless channel periodically",
            "Rapid periodic signal amplitude changes from vertical whole-body displacement",
        ],
        1: [
            "A person causing rapid translational movement inducing wireless signal fluctuations",
            "Fast whole-body displacement generating large broadband wireless channel variations",
            "Someone moving quickly through the space causing rapid signal phase shifts",
            "High-speed body translation producing strong Doppler-like effects in the wireless signal",
            "Rapid continuous body movement detected as sustained high-variance signal fluctuations",
        ],
        2: [
            "A person sitting still with only subtle breathing motion detectable by the sensor",
            "Nearly static wireless channel with only tiny periodic chest motion variations",
            "Minimal body movement producing barely detectable signal modulation from respiration",
            "Someone causing almost no wireless signal change except faint breathing oscillations",
            "Quasi-static scene where only gentle thoracic motion slightly modulates the channel",
        ],
        3: [
            "A person causing moderate-speed displacement with regular walking periodicity in the signal",
            "Steady translational body movement producing periodic gait-related signal fluctuations",
            "Someone moving at moderate pace creating regular stride-frequency channel variations",
            "Consistent moderate-rate body displacement with characteristic walking signal pattern",
            "Periodic moderate signal changes from regular bipedal locomotion through the sensing area",
        ],
        4: [
            "A person causing isolated arm oscillation detectable against an otherwise stationary body",
            "Small rapid signal perturbation from limb movement while the torso remains static",
            "Localized upper-body motion detected as focused signal variation on a stable baseline",
            "Someone creating periodic arm motion sensed as isolated limb oscillation by the channel",
            "Partial body movement with only one arm producing detectable wireless signal changes",
        ],
    },
}

FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "domain_expert": "Clinical vocabulary (1 template, raw)",
    "layperson": "Everyday language (1 template)",
    "sensor_grounded": "Sensor-physics vocabulary (1 template, raw)",
    "domain_expert_anchored": "Clinical vocabulary (1 template, anchored)",
    "sensor_grounded_anchored": "Sensor-physics vocabulary (1 template, anchored)",
    "domain_expert_ensemble": "Clinical vocabulary (5 templates, anchored)",
    "layperson_ensemble": "Everyday language (5 templates)",
    "sensor_grounded_ensemble": "Sensor-physics vocabulary (5 templates, anchored)",
}
