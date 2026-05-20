from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_EVAL_DIR = Path("/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/tailgating_eval")
ROI_CONFIG_DIR = BASE_EVAL_DIR / "roi_configs"
MANUAL_GT_DIR = BASE_EVAL_DIR / "manual_ground_truth"
SYSTEM_OUTPUT_DIR = BASE_EVAL_DIR / "system_outputs"
ERROR_ANALYSIS_DIR = BASE_EVAL_DIR / "error_analysis"
REPORT_DIR = BASE_EVAL_DIR / "reports"


def ensure_eval_dirs() -> None:
    for path in [
        ROI_CONFIG_DIR,
        MANUAL_GT_DIR,
        SYSTEM_OUTPUT_DIR,
        ERROR_ANALYSIS_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def video_stem(video_path: str | Path) -> str:
    return Path(video_path).stem


def roi_config_path(video_path: str | Path) -> Path:
    return ROI_CONFIG_DIR / f"{video_stem(video_path)}__roi.json"


def manual_gt_path(video_path: str | Path) -> Path:
    return MANUAL_GT_DIR / f"{video_stem(video_path)}__manual_gt.csv"


def system_event_path(video_path: str | Path, tracker_name: str) -> Path:
    return SYSTEM_OUTPUT_DIR / f"{video_stem(video_path)}__{tracker_name}__events.csv"


def error_analysis_path(video_path: str | Path, tracker_name: str) -> Path:
    return ERROR_ANALYSIS_DIR / f"{video_stem(video_path)}__{tracker_name}__comparison.csv"


def save_roi_config(video_path: str | Path, entry: tuple[int, int, int, int], exit_roi: tuple[int, int, int, int], zone: list[tuple[int, int]]) -> Path:
    ensure_eval_dirs()
    payload = {
        "video_name": Path(video_path).name,
        "entry": list(entry),
        "exit": list(exit_roi),
        "zone": [list(pt) for pt in zone],
    }
    out_path = roi_config_path(video_path)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_roi_config(video_path: str | Path) -> dict:
    return json.loads(roi_config_path(video_path).read_text())


def build_manual_gt_template(video_path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "video_name",
            "event_index",
            "time_sec",
            "manual_label",
            "manual_notes",
            "failure_category",
        ]
    ).assign(video_name=Path(video_path).name)


def save_manual_gt_template(video_path: str | Path) -> Path:
    ensure_eval_dirs()
    out_path = manual_gt_path(video_path)
    build_manual_gt_template(video_path).to_csv(out_path, index=False)
    return out_path
