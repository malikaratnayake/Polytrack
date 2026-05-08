---
name: Polytrack refactor approach
description: What was changed in the March 2025 refactor and why
type: feedback
---

Refactored Polytrack in March 2025. Key changes made:

**Why:** User requested reliability improvements, documentation, and good coding practices.

**How to apply:** If making further changes, continue these patterns.

Bugs fixed:
- `tracking_methods.py`: bare `except:` → `except (TypeError, IndexError, AttributeError):`; added `self.actual_nframe = 0` init; `get_compression_details` raises FileNotFoundError with helpful message instead of crashing on missing sidecar CSV
- `insect_tracker.py`: bare `except:` fixed; added `self.full_frame_num/video_frame_num/actual_frame_num = []` guards for DL-only mode with compressed video; removed unused `math` import and `area` variable
- `insect_recorder.py`: `find_last_detected_frame` now returns 0 with warning instead of implicit None; added `_find_track_position(id)` helper replacing all unsafe `int(next(..., None))` patterns; None checks added throughout
- `flower_recorder.py`: added `_find_flower_position(id)` helper; None checks in `record_flower_detections` and `record_flower_visitations`; removed unused `pandas` import
- `flower_tracker.py`: hardcoded `max_interframe_travel_distance = 10` extracted to module constant `_FLOWER_MATCH_DISTANCE_PX`
- `event_logger.py`: file handler log level was hardcoded to `logging.INFO`; now uses `log_level_value` so DEBUG output is captured when configured

Docstrings/comments added to all source files (module-level and class-level).
