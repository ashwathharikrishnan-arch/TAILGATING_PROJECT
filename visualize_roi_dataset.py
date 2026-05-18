from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/pilot_roi_dataset")
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = DATASET_DIR / "visualizations"


def load_label(label_path: Path) -> dict:
    return json.loads(label_path.read_text())


def draw_box(image: np.ndarray, rect: list[int] | tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = map(int, rect)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_zone(image: np.ndarray, zone: list[list[int]] | list[tuple[int, int]]) -> None:
    pts = np.array(zone, dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 0))
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
    cv2.polylines(image, [pts], True, (255, 255, 0), 3)
    center_x = int(np.mean(pts[:, 0]))
    center_y = int(np.mean(pts[:, 1]))
    cv2.putText(
        image,
        "ZONE",
        (center_x - 30, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def visualize_pair(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    label = load_label(label_path)

    if label.get("entry"):
        draw_box(image, label["entry"], (0, 255, 0), "ENTRY")
    if label.get("exit"):
        draw_box(image, label["exit"], (0, 0, 255), "EXIT")
    if label.get("zone"):
        draw_zone(image, label["zone"])

    cv2.imwrite(str(output_path), image)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(LABEL_DIR.glob("*.json"))
    if not label_paths:
        raise RuntimeError(f"No label files found in {LABEL_DIR}")

    generated = 0
    missing = []

    for label_path in label_paths:
        stem = label_path.stem
        image_path = IMAGE_DIR / f"{stem}.jpg"
        if not image_path.exists():
            missing.append(stem)
            continue

        output_path = OUTPUT_DIR / f"{stem}_viz.jpg"
        visualize_pair(image_path, label_path, output_path)
        generated += 1
        print(f"Saved: {output_path}")

    print(f"\nGenerated {generated} visualization(s).")
    if missing:
        print("Missing matching images for:")
        for stem in missing:
            print(f"  - {stem}")


if __name__ == "__main__":
    main()
