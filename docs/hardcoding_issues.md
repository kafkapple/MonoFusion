# Hardcoding Issues — casual_dataset.py & Pipeline

> Discovered during M5t2 PoC integration, 2026-03-29.
> casual_dataset.py is MonoFusion original code targeting Panoptic/Davis datasets.

---

## Critical (Blocks Training)

| # | Location | Hardcoded Value | M5t2 Impact | Fix Applied |
|---|----------|----------------|-------------|-------------|
| 1 | `casual_dataset.py:212` | `glb_step = 3` | 60 frames / 3 = 20 → camera count mismatch | `1 if 'm5t2' in video_name else 3` |
| 2 | `casual_dataset.py:367` | `range(0, 300, 3)` | IndexError at t=60 (only 60 frames) | `range(0, T, self.glb_step)` |
| 3 | `casual_dataset.py:436` | `path = f'_raw_data/{...}/Dy_train_meta.json'` (relative) | Fails unless CWD = data_root | Workaround: `cd data_root` before run |
| 4 | `casual_dataset.py:476` | `step=self.num_frames // 10` | query_idcs=[0,6,12,...] ≠ RAFT queries [0,15,30,45,59] | M5t2: `step = num_frames // 4` = 15 |
| 5 | `casual_dataset.py:365` | `c = int(self.seq_name[-1])` | Works for m5t2_cam0X (last char = digit) | No fix needed (compatible) |
| 6 | `generate_raft_tracks.py` | `expected_dist = 0.0` (was) | ALL tracks discarded by parse_tapir_track_info | Fixed to `CONFIDENT_DIST = -2.0` |

## Major (Affects Quality)

| # | Location | Issue | M5t2 Impact | Status |
|---|----------|-------|-------------|--------|
| 7 | `casual_dataset.py:225-231` | `video_name` branching for frame_names slicing | `_m5t2` falls to `else` branch (OK for now) | Monitor |
| 8 | `casual_dataset.py:69` | `depth_type` default = "modest" | M5t2 needs "moge" → set in monkey-patch | Fixed |
| 9 | `casual_dataset.py:71` | `track_2d_type` default = "bootstapir" | M5t2 uses "tapir" (RAFT symlink) | Fixed |
| 10 | images/ directory | 5-digit + 6-digit duplicates (120 files) | frame_names doubled → assertion fail | Deleted 5-digit files |

## Minor (Documentation / Robustness)

| # | Location | Issue | Status |
|---|----------|-------|--------|
| 11 | `train_m5t2.py:142` | `required = ["tapir"]` hardcoded | RAFT uses tapir/ symlink → works |
| 12 | `utils.py:134` | `mask == 1` exact float comparison | Boundary tracks may be lost (MonoFusion design) |
| 13 | DINOv2 | Docs say ViT-L/14(1024d), actual = ViT-S/14(384d) | Doc fixed |

---

## Fix Strategy

**Principle**: MonoFusion 원본 코드는 최소 수정. Fork 대신 conditional branch로 M5t2 지원.

Changes to `casual_dataset.py` (3 lines):
1. Line 212: `glb_step` conditional on video_name
2. Line 367: `range(0, T, self.glb_step)` (data-driven, not hardcoded)
3. Line 476: `_step` conditional for scene norm query spacing

Changes to own code:
- `generate_raft_tracks.py`: `CONFIDENT_DIST = -2.0`
- `train_m5t2.py`: monkey-patch sets correct kwargs

---

## Recommended Future Improvements

1. **CasualDataset subclass** — `M5t2Dataset(CasualDataset)` that overrides init/camera loading
2. **Config-driven query_idcs** — pass RAFT query frames as parameter, not compute from step
3. **Absolute path for camera JSON** — `root_dir / path` instead of relative
4. **Remove 5-digit image generation** from convert_m5t2.py (prevent future duplicates)

---

↑ MOC: `docs/README.md` | ↔ Related: `audit_pretraining.md`, `core_architecture.md`

*Hardcoding Issues | MonoFusion M5t2 PoC | 2026-03-29*
