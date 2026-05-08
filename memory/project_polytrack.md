---
name: Polytrack codebase overview
description: Architecture, key classes, and known issues for the Polytrack insect tracking software
type: project
---

Polytrack v5.0 is a Python insect tracking and pollination monitoring tool.

**Why:** Used for field ecology research, tracking unmarked freely foraging insects (bees, etc.) visiting flowers in outdoor video footage.

**How to apply:** When the user asks about Polytrack code, use this as context for understanding the architecture and design decisions.

**Architecture:**
- `src/main.py` – entry point; Config class (YAML→dot-notation object); TracknRecord orchestrates frame loop
- `src/insect_tracker.py` – InsectTracker (inherits DL_Detector + FGBG_Detector); hybrid detection pipeline
- `src/insect_recorder.py` – Recorder (inherits VideoWriter); writes per-insect CSV, verification CSV, trajectory plot
- `src/flower_tracker.py` – FlowerTracker (inherits DL_Flower_Detector + TrackingMethods)
- `src/flower_recorder.py` – FlowerRecorder; flower visitation events
- `src/tracking_methods.py` – TrackingMethods mixin; KalmanFilter; ExtendedKalmanFilter; Hungarian assignment; ABP

**Detection pipeline per frame:**
1. FGBG (MOG2 or FrameDifference) → blob candidates
2. YOLOv8 DL detector (run when missing/new insects detected, or FGBG disabled)
3. Hungarian/ABP assignment to existing predictions
4. Secondary YOLOv8 verification model (optional)
5. Relink missing tracks at relaxed distance threshold

**Data structures:**
- insect_tracks: list of [insect_id, start_frame, species_name, [[frame, x, y, flower_id, area, method, confidence, ...], ...]]
- Output: per-insect CSV, verification_Info CSV, flower CSVs, annotated video, track plot JPEG

**Config system:** YAML → nested Config objects with getattr() access; device auto-detected (CUDA > MPS > CPU)
