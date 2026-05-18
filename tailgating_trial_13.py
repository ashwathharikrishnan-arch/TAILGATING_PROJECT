import cv2, numpy as np, time, warnings, math
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import deque
from difflib import get_close_matches
import json
from IPython.display import display as ipy_display
import ipywidgets as widgets
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

VIDEO_PATH            = "/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/TAILGATING_VIDEOS/tailgating_tiral_t9.mp4"
VIDEO_DIR             = Path("/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/TAILGATING_VIDEOS")
SAMPLE_VIDEO_SET      = sorted(str(p) for p in VIDEO_DIR.glob("*.mp4"))
EVAL_DIR              = Path("/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/evaluation_outputs")
ROI_TEMPLATE_PATH     = Path("/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/roi_template.json")
TRACKER_PRIORITY      = ['bytetrack', 'ocsort', 'botsort', 'strongsort']
DEFAULT_TRACKER       = 'bytetrack'

# Durations are defined in seconds and converted to frames at runtime.
AUTH_TIMEOUT_SEC      = 5.0
ZONE_DWELL_MIN_SEC    = 0.33
MIN_STABLE_SEC        = 0.20
SCANNER_COOLDOWN_SEC  = 1.25
AUTH_RETROACTIVE_SEC  = 0.75
GATE_EXIT_GRACE_SEC   = 0.20

# KEY: single shared constant for both setup AND runtime resize
DISPLAY_MAX_DIM = 1000

# Detection thresholds — tuned for overhead camera angle
YOLO_CONF    = 0.35
YOLO_IOU     = 0.35
RFDETR_CONF  = 0.35
MIN_HEIGHT_PX = 30
MIN_WIDTH_PX  = 20
MIN_ASPECT   = 0.5
MAX_ASPECT   = 4.0

# Gate
GATE_OVERLAP_MIN = 0.20
MAX_ZONE_OCCUPANCY_AT_SCAN = 1

# ID merger
ID_MERGE_DIST    = 80
ID_MERGE_MAX_GAP = 45

# Scanner assignment
SCANNER_ASSIGN_MAX_DIST = 180
SCAN_GATE_BLOCK_MARGIN  = 50

# Diagnostics
TRACK_HISTORY_MAXLEN    = 8
DEBUG_PRINT_LIMIT       = 20

TRACKER_NAMES = TRACKER_PRIORITY.copy()
ANNOTATION_VIDEO_SET = SAMPLE_VIDEO_SET.copy()
ANNOTATION_INDEX = 0

print('✓ Cell 1 — config loaded')
print(f'  DISPLAY_MAX_DIM = {DISPLAY_MAX_DIM} (must match setup AND runtime)')

# ==========CELL 2================

from ultralytics import YOLO

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print('⚠ rfdetr not installed — pip install rfdetr')

try:
    from boxmot import StrongSORT, BoTSORT, BYTETracker, OCSORT
    BOXMOT_AVAILABLE = True
    print('✓ boxmot loaded')
except ImportError as e:
    BOXMOT_AVAILABLE = False
    print(f'✗ boxmot import failed: {e}')

print('Loading YOLOv12n...')
yolo_model = YOLO('yolo12n.pt')
print('✓ YOLOv12n loaded')

rfdetr_model = None
if RFDETR_AVAILABLE:
    try:
        rfdetr_model = RFDETRBase()
        print('✓ RF-DETR loaded')
    except Exception as e:
        print(f'⚠ RF-DETR failed to load: {e}')
        print('  Continuing in YOLO-only mode')
else:
    print('⚠ Running in YOLO-only mode')

print('✓ Cell 2 — models loaded')

# =========CELL 3===================
REID_WEIGHTS = Path('osnet_x0_25_msmt17.pt')

TRACKER_BUILD_NOTES = {}

# ── Tracker factory ──────────────────────────────────────────
def tracker_supported(name):
    return name.lower() in {'strongsort', 'botsort', 'bytetrack', 'ocsort'}

def make_tracker(name, fps=15.0):
    if not BOXMOT_AVAILABLE:
        raise RuntimeError('boxmot is not available in this environment')
    n = name.lower()
    if n == 'strongsort':
        if not REID_WEIGHTS.exists():
            raise FileNotFoundError(
                f'Missing re-id weights: {REID_WEIGHTS}. '
                'StrongSORT needs an appearance model to stay stable.'
            )
        return StrongSORT(model_weights=REID_WEIGHTS, device='cpu', fp16=False)
    elif n == 'botsort':
        if not REID_WEIGHTS.exists():
            raise FileNotFoundError(
                f'Missing re-id weights: {REID_WEIGHTS}. '
                'BoTSORT needs an appearance model to stay stable.'
            )
        return BoTSORT(model_weights=REID_WEIGHTS, device='cpu', fp16=False, frame_rate=max(int(round(fps)), 1))
    elif n == 'bytetrack':
        return BYTETracker(frame_rate=max(int(round(fps)), 1))
    elif n == 'ocsort':
        return OCSORT(per_class=False)
    else:
        raise ValueError(f'Unknown tracker: {name}')

# ── Frame resize — uses shared DISPLAY_MAX_DIM ───────────────
def resize_frame(frame):
    """Resize so longest edge = DISPLAY_MAX_DIM. Used in BOTH setup and runtime."""
    if max(frame.shape[:2]) > DISPLAY_MAX_DIM:
        sc = DISPLAY_MAX_DIM / max(frame.shape[:2])
        frame = cv2.resize(frame, (int(frame.shape[1]*sc), int(frame.shape[0]*sc)))
    return frame

# ── Geometry helpers ─────────────────────────────────────────
def foot_point(bbox):
    """Bottom-center of bbox — physically where a person stands."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)

def filter_det(x1, y1, x2, y2):
    """Filter out non-person detections by size and aspect ratio."""
    w, h = x2 - x1, y2 - y1
    if w < MIN_WIDTH_PX or h < MIN_HEIGHT_PX:
        return False
    aspect = h / max(w, 1)
    return MIN_ASPECT <= aspect <= MAX_ASPECT

def bbox_in_polygon(bbox, poly):
    """
    Returns fraction of test points (center + 4 corners) inside polygon.
    Uses cv2.pointPolygonTest — correct for non-rectangular zones.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    pts = [(cx,cy), (x1,y1), (x2,y1), (x1,y2), (x2,y2)]
    hits = sum(
        cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0
        for px, py in pts
    )
    return hits / len(pts)

def gate_band(poly):
    """Bounding box of the crossing zone polygon."""
    return (int(poly[:,0].min()), int(poly[:,1].min()),
            int(poly[:,0].max()), int(poly[:,1].max()))

def near_gate(bbox, gb, m=40):
    """Quick pre-filter: is bbox close enough to gate to run RF-DETR?"""
    gx1, gy1, gx2, gy2 = gb
    x1, y1, x2, y2 = bbox
    return not (x2 < gx1-m or x1 > gx2+m or y2 < gy1-m or y1 > gy2+m)

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6)

def frames_for_seconds(seconds, fps, minimum=1):
    return max(int(round(seconds * max(fps, 1e-3))), minimum)

def point_dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

def foot_in_polygon(bbox, poly):
    fx, fy = foot_point(bbox)
    return cv2.pointPolygonTest(poly, (float(fx), float(fy)), False) >= 0

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# ── Detection ────────────────────────────────────────────────
def run_yolo(frame):
    res = yolo_model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                     classes=[0], imgsz=640, verbose=False)[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if filter_det(x1, y1, x2, y2):
            dets.append([x1, y1, x2, y2, float(b.conf[0]), 0])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0,6), dtype=np.float32)

def run_rfdetr(frame, gb):
    if rfdetr_model is None:
        return np.empty((0,6), dtype=np.float32)
    gx1, gy1, gx2, gy2 = gb
    m = 60;  h, w = frame.shape[:2]
    cx1, cy1 = max(0, gx1-m), max(0, gy1-m)
    cx2, cy2 = min(w, gx2+m), min(h, gy2+m)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.empty((0,6), dtype=np.float32)
    try:
        results = rfdetr_model.predict(crop, threshold=RFDETR_CONF)
        dets = []
        for d in results:
            if int(d[5]) != 0: continue
            x1 = int(d[0]) + cx1;  y1 = int(d[1]) + cy1
            x2 = int(d[2]) + cx1;  y2 = int(d[3]) + cy1
            if filter_det(x1, y1, x2, y2):
                dets.append([x1, y1, x2, y2, float(d[4]), 0])
        return np.array(dets, dtype=np.float32) if dets else np.empty((0,6), dtype=np.float32)
    except:
        return np.empty((0,6), dtype=np.float32)

def merge_dets(yd, rd, thr=0.5):
    if len(rd) == 0: return yd
    if len(yd) == 0: return rd
    kept = [y for y in yd if not any(_iou(y[:4], r[:4]) > thr for r in rd)]
    all_d = kept + list(rd)
    return np.array(all_d, dtype=np.float32) if all_d else np.empty((0,6), dtype=np.float32)

# ── Classes ──────────────────────────────────────────────────
class Scanner:
    def __init__(self, roi, name, crossing_zone=None):
        self.roi           = roi
        self.name          = name
        self.cooldown      = {}
        self.crossing_zone = crossing_zone
        self.cx            = (roi[0] + roi[2]) // 2
        self.cy            = (roi[1] + roi[3]) // 2

    def _person_already_in_zone(self, bbox):
        if self.crossing_zone is None:
            return False
        return foot_in_polygon(bbox, self.crossing_zone)

    def _count_people_near_gate(self, all_bboxes, margin=60):
        if self.crossing_zone is None:
            return 0
        gx1 = self.crossing_zone[:,0].min() - margin
        gy1 = self.crossing_zone[:,1].min() - margin
        gx2 = self.crossing_zone[:,0].max() + margin
        gy2 = self.crossing_zone[:,1].max() + margin
        count = 0
        for bbox in all_bboxes.values():
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                count += 1
        return count

    def check(self, feet, bboxes, trackers, fi, active_auths=None, scanner_cooldown_frames=1):
        x1, y1, x2, y2 = self.roi
        candidates = []
        for tid, (fx, fy) in feet.items():
            if tid in self.cooldown and fi - self.cooldown[tid] < scanner_cooldown_frames:
                continue

            foot_hit = x1 <= fx <= x2 and y1 <= fy <= y2

            bbox_hit = False
            if tid in bboxes:
                bx1, by1, bx2, by2 = bboxes[tid]
                bbox_hit = not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2)

            if foot_hit or bbox_hit:
                if tid in bboxes and self._person_already_in_zone(bboxes[tid]):
                    continue

                pt = trackers.get(tid)
                if pt is None:
                    continue

                stable_bonus = 0.0 if pt.stable() else 35.0
                foot_penalty = 0.0 if foot_hit else 55.0
                scan_dist = point_dist((fx, fy), (self.cx, self.cy))
                if scan_dist > SCANNER_ASSIGN_MAX_DIST:
                    continue

                if active_auths is not None:
                    pending_same_direction = [
                        a for a in active_auths
                        if a.direction == self.name
                        and a.active
                        and not a.used
                        and a.scanner_id != tid
                    ]
                    if pending_same_direction:
                        print(f'[SCANNER BLOCK] F{fi}: ID {tid} blocked — '
                              f'{len(pending_same_direction)} unused auth(s) pending '
                              f'for {self.name}')
                        continue

                score = scan_dist + stable_bonus + foot_penalty
                candidates.append((score, tid))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        best_tid = candidates[0][1]
        self.cooldown[best_tid] = fi
        return best_tid


class Authorization:
    def __init__(self, aid, direction, fi, sid):
        self.auth_id     = aid
        self.direction   = direction
        self.start_frame = fi
        self.scanner_id  = sid
        self.active      = True
        self.used        = False

    def expired(self, f, auth_timeout_frames):
        return f - self.start_frame > auth_timeout_frames

    def age(self, f, fps):
        return (f - self.start_frame) / fps


class PersonTracker:
    def __init__(self, tid):
        self.tid                  = tid
        self.bbox_history         = deque(maxlen=TRACK_HISTORY_MAXLEN)
        self.stable_frames        = 0
        self.pending_gate_frames  = 0
        self.is_in_gate           = False
        self.first_seen           = None
        self.has_crossed          = False
        self.used_auth            = False
        self.auth_id              = None
        self.pending_auth_check   = False
        self.last_seen_frame      = None
        self.last_foot            = None
        self.missed_gate_frames   = 0
        self.zone_entries         = 0
        self.decision_reason      = None

    def update(self, bbox, frame_idx):
        self.bbox_history.append(bbox)
        self.stable_frames += 1
        self.last_seen_frame = frame_idx
        self.last_foot = foot_point(bbox)

    def smoothed(self):
        if not self.bbox_history: return None
        return tuple(np.mean(np.array(self.bbox_history), axis=0).astype(int))

    def stable(self, min_stable_frames):
        return self.stable_frames >= min_stable_frames


class IDMerger:
    def __init__(self):
        self.aliases  = {}
        self.last_pos = {}

    def resolve(self, tid):
        return self.aliases.get(tid, tid)

    def update(self, tid, pos, f):
        self.last_pos[self.resolve(tid)] = (np.array(pos), f)

    def check(self, ids, positions, f):
        for tid in ids:
            if tid in self.aliases: continue
            pos = np.array(positions[tid])
            best, bd = None, ID_MERGE_DIST
            for oid, (op, lf) in self.last_pos.items():
                if oid in ids or f - lf > ID_MERGE_MAX_GAP: continue
                d = float(np.linalg.norm(pos - op))
                if d < bd:
                    bd, best = d, oid
            if best:
                self.aliases[tid] = best

print('✓ Cell 3 — all helpers loaded')

# ===========CELL 4a==================
from matplotlib.widgets import RectangleSelector

cap_s = cv2.VideoCapture(VIDEO_PATH)
ret, setup_frame = cap_s.read()
cap_s.release()

if not ret:
    raise RuntimeError(f'Cannot read video: {VIDEO_PATH}')

setup_frame = cv2.cvtColor(setup_frame, cv2.COLOR_BGR2RGB)
setup_frame = resize_frame(setup_frame)

roi_state = {
    'entry': None,
    'exit':  None,
    'zone':  [],
}

print(f'✓ Cell 4a — frame loaded: {setup_frame.shape[1]} x {setup_frame.shape[0]}')
print(f'  DISPLAY_MAX_DIM = {DISPLAY_MAX_DIM}')
print('  Proceed to Cell 4b')


def load_annotation_video(index=None, video_paths=None):
    """
    Load one video from the annotation queue into VIDEO_PATH/setup_frame/roi_state
    so you can annotate videos back-to-back without manually editing paths.
    """
    global VIDEO_PATH, setup_frame, roi_state, ANNOTATION_INDEX, ANNOTATION_VIDEO_SET

    if video_paths is not None:
        ANNOTATION_VIDEO_SET = [str(Path(v)) for v in video_paths]
    if not ANNOTATION_VIDEO_SET:
        raise RuntimeError('No videos available for annotation.')

    if index is not None:
        ANNOTATION_INDEX = int(index)

    if not (0 <= ANNOTATION_INDEX < len(ANNOTATION_VIDEO_SET)):
        raise IndexError(f'Annotation index out of range: {ANNOTATION_INDEX}')

    VIDEO_PATH = ANNOTATION_VIDEO_SET[ANNOTATION_INDEX]
    cap_s = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap_s.read()
    cap_s.release()
    if not ret:
        raise RuntimeError(f'Cannot read video: {VIDEO_PATH}')

    setup_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    setup_frame = resize_frame(setup_frame)
    roi_state = {'entry': None, 'exit': None, 'zone': []}

    print(f'Loaded [{ANNOTATION_INDEX + 1}/{len(ANNOTATION_VIDEO_SET)}]: {Path(VIDEO_PATH).name}')
    print(f'Frame size: {setup_frame.shape[1]} x {setup_frame.shape[0]}')
    return VIDEO_PATH


def save_current_annotation(
    label_dir="/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/pilot_roi_dataset/labels",
    image_dir="/Users/ashwath.harikrishnan/Documents/TAILGATING_PROJECT/pilot_roi_dataset/images",
):
    """
    Save the current ROI annotation as both JSON label and JPG reference image.
    """
    if roi_state['entry'] is None or roi_state['exit'] is None or len(roi_state['zone']) != 4:
        raise RuntimeError(
            f'ROI incomplete for {Path(VIDEO_PATH).name}: '
            f'entry={roi_state["entry"]}, exit={roi_state["exit"]}, zone={roi_state["zone"]}'
        )

    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(VIDEO_PATH).name
    video_stem = Path(VIDEO_PATH).stem

    label_data = {
        'video_name': video_name,
        'entry': list(roi_state['entry']),
        'exit': list(roi_state['exit']),
        'zone': [list(pt) for pt in roi_state['zone']],
    }

    label_path = label_dir / f'{video_stem}.json'
    label_path.write_text(json.dumps(label_data, indent=2))

    image_path = image_dir / f'{video_stem}.jpg'
    frame_bgr = cv2.cvtColor(setup_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), frame_bgr)

    print(f'Saved ROI label: {label_path}')
    print(f'Saved reference image: {image_path}')
    return label_path, image_path


def save_and_advance_annotation():
    """
    Save current annotation, then move to the next video in the queue.
    """
    global ANNOTATION_INDEX
    saved = save_current_annotation()
    if ANNOTATION_INDEX + 1 >= len(ANNOTATION_VIDEO_SET):
        print('All annotation videos completed.')
        return saved

    ANNOTATION_INDEX += 1
    print('\nMoving to next video...\n')
    load_annotation_video(ANNOTATION_INDEX)
    return saved

# ======== AUTO ROI HELPERS ========
def _norm_rect(rect, w, h):
    x1, y1, x2, y2 = rect
    return {
        'x1': x1 / w, 'y1': y1 / h,
        'x2': x2 / w, 'y2': y2 / h,
    }


def _denorm_rect(obj, w, h):
    return (
        int(round(obj['x1'] * w)),
        int(round(obj['y1'] * h)),
        int(round(obj['x2'] * w)),
        int(round(obj['y2'] * h)),
    )


def _norm_poly(points, w, h):
    return [{'x': x / w, 'y': y / h} for x, y in points]


def _denorm_poly(points, w, h):
    return [(int(round(p['x'] * w)), int(round(p['y'] * h))) for p in points]


def save_roi_template(entry_roi, exit_roi, zone_pts, frame_shape, template_path=ROI_TEMPLATE_PATH):
    """
    Save one manually reviewed ROI layout as normalized coordinates so it can be
    reused on other videos with similar framing.
    """
    h, w = frame_shape[:2]
    payload = {
        'reference_width': w,
        'reference_height': h,
        'entry': _norm_rect(entry_roi, w, h),
        'exit': _norm_rect(exit_roi, w, h),
        'zone': _norm_poly(zone_pts, w, h),
    }
    template_path.write_text(json.dumps(payload, indent=2))
    print(f'✓ ROI template saved to: {template_path}')


def load_roi_template(frame_shape, template_path=ROI_TEMPLATE_PATH):
    if not template_path.exists():
        raise FileNotFoundError(f'No ROI template found at {template_path}')
    h, w = frame_shape[:2]
    payload = json.loads(template_path.read_text())
    return {
        'entry': _denorm_rect(payload['entry'], w, h),
        'exit': _denorm_rect(payload['exit'], w, h),
        'zone': _denorm_poly(payload['zone'], w, h),
    }


def set_roi_from_template(frame_shape, template_path=ROI_TEMPLATE_PATH):
    auto_roi = load_roi_template(frame_shape, template_path=template_path)
    roi_state['entry'] = auto_roi['entry']
    roi_state['exit'] = auto_roi['exit']
    roi_state['zone'] = auto_roi['zone']
    print('✓ ROI state auto-loaded from template')
    print(f'  Entry : {roi_state["entry"]}')
    print(f'  Exit  : {roi_state["exit"]}')
    print(f'  Zone  : {roi_state["zone"]}')


def preview_roi_state(frame_rgb, roi_state_obj=None, title='ROI Preview'):
    state = roi_state if roi_state_obj is None else roi_state_obj
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(frame_rgb)

    er = state['entry']
    xr = state['exit']
    zp = np.array(state['zone'], dtype=np.int32) if state['zone'] else None

    if er:
        ax.add_patch(plt.Rectangle(
            (er[0], er[1]), er[2]-er[0], er[3]-er[1],
            edgecolor='green', facecolor='green', alpha=0.2, linewidth=2
        ))
        ax.text(er[0], max(er[1]-8, 10), 'ENTRY',
                color='green', fontsize=11, fontweight='bold')

    if xr:
        ax.add_patch(plt.Rectangle(
            (xr[0], xr[1]), xr[2]-xr[0], xr[3]-xr[1],
            edgecolor='red', facecolor='red', alpha=0.2, linewidth=2
        ))
        ax.text(xr[0], max(xr[1]-8, 10), 'EXIT',
                color='red', fontsize=11, fontweight='bold')

    if zp is not None and len(zp) == 4:
        ax.add_patch(plt.Polygon(
            zp, closed=True,
            facecolor='cyan', alpha=0.25, edgecolor='cyan', linewidth=2
        ))
        ax.text(zp[:,0].mean(), zp[:,1].mean(), 'ZONE',
                color='cyan', fontsize=13, fontweight='bold',
                ha='center', va='center')

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def apply_or_draw_rois(frame_rgb, prefer_template=True, template_path=ROI_TEMPLATE_PATH):
    """
    Convenience wrapper for notebooks:
      - If a saved ROI template exists, apply it and preview the result.
      - Otherwise fall back to the manual Cell 4b drawing flow.
    """
    if prefer_template and template_path.exists():
        set_roi_from_template(frame_rgb.shape, template_path=template_path)
        preview_roi_state(frame_rgb, title='ROI Preview — loaded from template')
        print('If these look correct, skip manual ROI drawing and proceed.')
        return True

    print('No ROI template found yet. Use the manual ROI drawing cells once, then save:')
    print("save_roi_template(roi_state['entry'], roi_state['exit'], roi_state['zone'], setup_frame.shape)")
    return False

# ========CELL 4b — Draw ENTRY Box====================
%matplotlib tk

fig_e, ax_e = plt.subplots(figsize=(12, 7))
ax_e.imshow(setup_frame)
ax_e.set_title('ENTRY scanner — drag to draw, then CLOSE this window',
               color='green', fontsize=12)
_entry_tmp = {}

def _on_entry_select(eclick, erelease):
    _entry_tmp['rect'] = (
        int(min(eclick.xdata, erelease.xdata)),
        int(min(eclick.ydata, erelease.ydata)),
        int(max(eclick.xdata, erelease.xdata)),
        int(max(eclick.ydata, erelease.ydata))
    )
    ax_e.set_title(f'ENTRY: {_entry_tmp["rect"]} — close to confirm',
                   color='green', fontsize=11)
    fig_e.canvas.draw()

def _on_entry_close(event):
    if 'rect' in _entry_tmp:
        roi_state['entry'] = _entry_tmp['rect']
        print(f'✓ Entry ROI saved: {roi_state["entry"]}')
    else:
        print('⚠ No box drawn — rerun this cell and drag before closing')

rs_entry = RectangleSelector(
    ax_e, _on_entry_select, useblit=True,
    props=dict(edgecolor='green', facecolor='green', alpha=0.3, fill=True),
    interactive=True
)
fig_e.canvas.mpl_connect('close_event', _on_entry_close)
plt.tight_layout()
plt.show()

# ======= EXIT scanner =====================
fig_x, ax_x = plt.subplots(figsize=(12, 7))
ax_x.imshow(setup_frame)
ax_x.set_title('EXIT scanner — drag to draw, then CLOSE this window',
               color='red', fontsize=12)
_exit_tmp = {}

def _on_exit_select(eclick, erelease):
    _exit_tmp['rect'] = (
        int(min(eclick.xdata, erelease.xdata)),
        int(min(eclick.ydata, erelease.ydata)),
        int(max(eclick.xdata, erelease.xdata)),
        int(max(eclick.ydata, erelease.ydata))
    )
    ax_x.set_title(f'EXIT: {_exit_tmp["rect"]} — close to confirm',
                   color='red', fontsize=11)
    fig_x.canvas.draw()

def _on_exit_close(event):
    if 'rect' in _exit_tmp:
        roi_state['exit'] = _exit_tmp['rect']
        print(f'✓ Exit ROI saved: {roi_state["exit"]}')
    else:
        print('⚠ No box drawn — rerun this cell and drag before closing')

rs_exit = RectangleSelector(
    ax_x, _on_exit_select, useblit=True,
    props=dict(edgecolor='red', facecolor='red', alpha=0.3, fill=True),
    interactive=True
)
fig_x.canvas.mpl_connect('close_event', _on_exit_close)
plt.tight_layout()
plt.show()

# =============== Crossing Zone ===========
_zone_tmp   = []
_poly_patch = [None]

fig_z, ax_z = plt.subplots(figsize=(12, 7))
ax_z.imshow(setup_frame)
ax_z.set_title('CROSSING ZONE — click 4 corners, then CLOSE this window',
               color='cyan', fontsize=12)
_scatter = ax_z.scatter([], [], c='cyan', s=80, zorder=5)

def _on_zone_click(event):
    if event.inaxes and len(_zone_tmp) < 4:
        _zone_tmp.append((int(event.xdata), int(event.ydata)))
        _scatter.set_offsets(np.array(_zone_tmp))
        if len(_zone_tmp) == 4:
            if _poly_patch[0]:
                _poly_patch[0].remove()
            _poly_patch[0] = plt.Polygon(
                _zone_tmp, closed=True,
                facecolor='cyan', alpha=0.25, edgecolor='cyan', linewidth=2
            )
            ax_z.add_patch(_poly_patch[0])
            ax_z.set_title('✓ 4 points set — close window to confirm',
                           color='cyan', fontsize=12)
        else:
            ax_z.set_title(f'ZONE: {len(_zone_tmp)}/4 — keep clicking',
                           color='cyan', fontsize=11)
        fig_z.canvas.draw()

def _on_zone_close(event):
    if len(_zone_tmp) == 4:
        roi_state['zone'] = _zone_tmp.copy()
        print(f'✓ Zone saved: {roi_state["zone"]}')
    else:
        print(f'⚠ Only {len(_zone_tmp)}/4 points — rerun and click all 4 before closing')

fig_z.canvas.mpl_connect('button_press_event', _on_zone_click)
fig_z.canvas.mpl_connect('close_event', _on_zone_close)
plt.tight_layout()
plt.show()

# %%
# ========CELL 4c====================
# =============== ROI state preview ===========
print('ROI state:')
print(f'  Entry : {roi_state["entry"]}')
print(f'  Exit  : {roi_state["exit"]}')
print(f'  Zone  : {roi_state["zone"]}')

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(setup_frame)

er = roi_state['entry']
xr = roi_state['exit']
zp = np.array(roi_state['zone'], dtype=np.int32) if roi_state['zone'] else None

if er:
    ax.add_patch(plt.Rectangle(
        (er[0], er[1]), er[2]-er[0], er[3]-er[1],
        edgecolor='green', facecolor='green', alpha=0.2, linewidth=2
    ))
    ax.text(er[0], max(er[1]-8, 10), 'ENTRY',
            color='green', fontsize=11, fontweight='bold')

if xr:
    ax.add_patch(plt.Rectangle(
        (xr[0], xr[1]), xr[2]-xr[0], xr[3]-xr[1],
        edgecolor='red', facecolor='red', alpha=0.2, linewidth=2
    ))
    ax.text(xr[0], max(xr[1]-8, 10), 'EXIT',
            color='red', fontsize=11, fontweight='bold')

if zp is not None and len(zp) == 4:
    ax.add_patch(plt.Polygon(
        zp, closed=True,
        facecolor='cyan', alpha=0.25, edgecolor='cyan', linewidth=2
    ))
    ax.text(zp[:,0].mean(), zp[:,1].mean(), 'ZONE',
            color='cyan', fontsize=13, fontweight='bold', ha='center', va='center')

missing = [k for k, v in roi_state.items() if v is None or v == []]
if missing:
    ax.set_title(f'⚠ Missing zones: {missing} — go back to 4b/4c/4d',
                 color='red', fontsize=12)
else:
    ax.set_title('✓ All zones set — if correct, proceed to Cell 5', fontsize=12)

plt.tight_layout()
plt.show()

# ========CELL 8===========

def frame_to_jpeg(frame_bgr, q=75):
    """Convert BGR frame to JPEG bytes for ipywidgets.Image display."""
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

def inspect_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        'video_path': str(video_path),
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_sec': total_frames / max(fps, 1e-6),
    }

def runtime_thresholds(fps):
    return {
        'auth_timeout_frames': frames_for_seconds(AUTH_TIMEOUT_SEC, fps),
        'zone_dwell_min_frames': frames_for_seconds(ZONE_DWELL_MIN_SEC, fps),
        'min_stable_frames': frames_for_seconds(MIN_STABLE_SEC, fps),
        'scanner_cooldown_frames': frames_for_seconds(SCANNER_COOLDOWN_SEC, fps),
        'auth_retroactive_frames': frames_for_seconds(AUTH_RETROACTIVE_SEC, fps),
        'gate_exit_grace_frames': frames_for_seconds(GATE_EXIT_GRACE_SEC, fps),
    }

def available_trackers(video_fps=15.0):
    available = []
    notes = {}
    for name in TRACKER_PRIORITY:
        try:
            tracker = make_tracker(name, fps=video_fps)
            available.append(name)
            notes[name] = 'ok'
            del tracker
        except Exception as e:
            notes[name] = str(e)
    return available, notes


def annotate(frame, persons_bboxes, person_trackers,
             entry_roi, exit_roi, crossing_zone, gband,
             active_auths, tracker_name, frame_idx, total_frames,
             fps_actual, total_auths, valid_crossings,
             tailgate_events, timeout_tailgating, people_in_zone,
             thresholds):
    """Draw all overlays onto BGR frame and return it."""
    f = frame.copy()

    # ── Zone overlays ──────────────────────────────────────────
    cv2.rectangle(f, (entry_roi[0], entry_roi[1]),
                  (entry_roi[2], entry_roi[3]), (0,255,0), 2)
    cv2.putText(f, 'ENTRY', (entry_roi[0], max(entry_roi[1]-8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.rectangle(f, (exit_roi[0], exit_roi[1]),
                  (exit_roi[2], exit_roi[3]), (255,0,0), 2)
    cv2.putText(f, 'EXIT', (exit_roi[0], max(exit_roi[1]-8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    ov = f.copy()
    cv2.fillPoly(ov, [crossing_zone], (0,255,255))
    cv2.addWeighted(ov, 0.2, f, 0.8, 0, f)
    cv2.polylines(f, [crossing_zone], True, (0,255,255), 3)

    gx1, gy1, gx2, gy2 = gband
    cv2.rectangle(f, (gx1,gy1), (gx2,gy2), (0,200,255), 1)

    # ── Person boxes ───────────────────────────────────────────
    for tid, bbox in persons_bboxes.items():
        pt = person_trackers.get(tid)
        if pt is None: continue

        stab = 'OK' if pt.stable(thresholds['min_stable_frames']) else '?'

        if pt.used_auth:
            color, lbl = (0,255,0),   f'ID:{tid} {stab} AUTH'
        elif pt.has_crossed:
            color, lbl = (0,0,255),   f'ID:{tid} {stab} TAIL'
        elif pt.pending_auth_check:
            color, lbl = (0,165,255), f'ID:{tid} {stab} WAIT'
        elif pt.is_in_gate:
            color, lbl = (0,255,255), f'ID:{tid} {stab} IN'
        else:
            color, lbl = (255,165,0), f'ID:{tid} {stab}'

        cv2.rectangle(f, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
        lsz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        ty  = max(bbox[1]-4, lsz[1]+4)
        cv2.rectangle(f, (bbox[0], ty-lsz[1]-4), (bbox[0]+lsz[0]+2, ty+2), color, -1)
        cv2.putText(f, lbl, (bbox[0]+1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

        fx, fy = foot_point(bbox)
        cv2.circle(f, (fx, fy), 4, color, -1)

    # ── Active auth list (top-left) ────────────────────────────
    STATS_TOP = f.shape[0] - 100
    y = 30
    for auth in active_auths[:5]:
        if y + 28 >= STATS_TOP: break
        fps_safe = max(fps_actual, 1)
        s = (f'{auth.auth_id} ({auth.direction}) '
             f'ID:{auth.scanner_id} [{auth.age(frame_idx, fps_safe):.1f}s]')
        cv2.rectangle(f, (10,y), (520,y+24), (40,40,40), -1)
        cv2.putText(f, s, (14,y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 1)
        y += 27

    # ── Stats panel (bottom) ───────────────────────────────────
    sy = f.shape[0] - 95
    cv2.rectangle(f, (0,sy), (f.shape[1], f.shape[0]), (0,0,0), -1)
    lines = [
        f'Tracker:{tracker_name.upper()}  Frame:{frame_idx}/{total_frames}  FPS:{fps_actual:.1f}',
        f'Auths:{total_auths}  Valid:{valid_crossings}  Tailgate:{tailgate_events}  Timeout:{timeout_tailgating}',
        f'In gate:{people_in_zone}  Active auths:{len(active_auths)}',
    ]
    for i, s in enumerate(lines):
        cv2.putText(f, s, (10, sy+22+i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,255), 1)

    return f

print('✓ Cell 5 — annotate() and frame_to_jpeg() defined')


def event_time_seconds(frame_idx, fps):
    """Convert a 1-based frame index into wall-clock seconds."""
    fps_safe = max(float(fps), 1e-6)
    return max((int(frame_idx) - 1) / fps_safe, 0.0)


def format_mmss(seconds):
    total_seconds = max(float(seconds), 0.0)
    minutes = int(total_seconds // 60)
    secs = total_seconds - minutes * 60
    return f'{minutes:02d}:{secs:06.3f}'


def canonical_event_label(label):
    text = str(label).strip().upper()
    if 'VALID' in text:
        return 'VALID'
    if 'TAILGATE' in text:
        return 'TAILGATE'
    return text or 'UNLABELED'


def failure_category_options():
    return [
        '',
        'roi_misaligned',
        'tracker_id_switch',
        'missed_detection',
        'scanner_assignment',
        'timing_window',
        'crowding_overlap',
        'manual_review_needed',
    ]


def prepare_event_log_df(event_log_df, video_path, video_fps, tracker_name):
    """Add comparison-friendly columns to the raw event log."""
    if event_log_df is None or event_log_df.empty:
        return pd.DataFrame(columns=[
            'video_name', 'tracker', 'event_index', 'frame', 'time_sec', 'time_mmss',
            'person_id', 'predicted_label', 'predicted_raw', 'auth', 'reason'
        ])

    df = event_log_df.copy().reset_index(drop=True)
    df['event_index'] = np.arange(1, len(df) + 1)
    df['video_name'] = Path(video_path).name
    df['tracker'] = tracker_name
    df['time_sec'] = df['frame'].apply(lambda f: round(event_time_seconds(f, video_fps), 3))
    df['time_mmss'] = df['time_sec'].apply(format_mmss)
    df['person_id'] = df['id']
    df['predicted_raw'] = df['result']
    df['predicted_label'] = df['result'].apply(canonical_event_label)
    ordered_cols = [
        'video_name', 'tracker', 'event_index', 'frame', 'time_sec', 'time_mmss',
        'person_id', 'predicted_label', 'predicted_raw', 'auth', 'reason'
    ]
    return df[ordered_cols]


def build_manual_review_template(video_path, system_events=None, include_blank_row=True):
    """
    Create a CSV-ready template for hand labeling.
    If system events are provided, prefill likely timestamps so review is faster.
    """
    video_name = Path(video_path).name
    rows = []

    if system_events is not None and not system_events.empty:
        for _, row in system_events.iterrows():
            rows.append({
                'video_name': video_name,
                'event_index': int(row['event_index']),
                'frame': int(row['frame']),
                'time_sec': float(row['time_sec']),
                'time_mmss': row['time_mmss'],
                'manual_label': '',
                'manual_notes': '',
                'manual_person_hint': '',
            })

    if include_blank_row or not rows:
        rows.append({
            'video_name': video_name,
            'event_index': pd.NA,
            'frame': pd.NA,
            'time_sec': pd.NA,
            'time_mmss': '',
            'manual_label': '',
            'manual_notes': '',
            'manual_person_hint': '',
        })

    return pd.DataFrame(rows)


def save_review_artifacts(result, output_dir=EVAL_DIR, include_template=True):
    """Persist system predictions and a hand-label template for one run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_name = result['tracker']
    video_path = result['video_path']
    video_stem = Path(video_path).stem
    system_events = prepare_event_log_df(
        result.get('event_log'),
        video_path=video_path,
        video_fps=result['video_fps'],
        tracker_name=tracker_name
    )

    system_csv = output_dir / f'{video_stem}__{tracker_name}__system_events.csv'
    system_events.to_csv(system_csv, index=False)

    paths = {'system_events_csv': system_csv}
    if include_template:
        template_df = build_manual_review_template(video_path, system_events=system_events)
        template_csv = output_dir / f'{video_stem}__manual_review_template.csv'
        template_df.to_csv(template_csv, index=False)
        paths['manual_template_csv'] = template_csv

    return paths


def load_manual_labels(csv_path):
    df = pd.read_csv(csv_path)
    expected = {'video_name', 'manual_label'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f'Manual labels CSV is missing columns: {sorted(missing)}')

    loaded = df.copy()
    loaded['manual_label'] = loaded['manual_label'].fillna('').map(canonical_event_label)
    if 'time_sec' in loaded.columns:
        loaded['time_sec'] = pd.to_numeric(loaded['time_sec'], errors='coerce')
    if 'frame' in loaded.columns:
        loaded['frame'] = pd.to_numeric(loaded['frame'], errors='coerce')
    return loaded


def _best_video_name_match(video_name, candidates):
    if video_name in candidates:
        return video_name
    matches = get_close_matches(video_name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def compare_manual_vs_system(system_events, manual_labels, time_tolerance_sec=1.0):
    """
    Match each predicted event to the closest manual event in time and summarize
    whether the system agreed with your review.
    """
    system_df = system_events.copy().reset_index(drop=True)
    manual_df = manual_labels.copy().reset_index(drop=True)

    if system_df.empty and manual_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    manual_df = manual_df[manual_df['manual_label'] != 'UNLABELED'].copy()
    if 'manual_notes' not in manual_df.columns:
        manual_df['manual_notes'] = ''
    if 'failure_category' not in manual_df.columns:
        manual_df['failure_category'] = ''

    system_df['matched_manual_index'] = pd.NA
    system_df['comparison_status'] = 'UNMATCHED_SYSTEM_EVENT'
    system_df['manual_label'] = ''
    system_df['manual_notes'] = ''
    system_df['failure_category'] = ''

    matched_manual = set()

    for sys_idx, sys_row in system_df.iterrows():
        same_video = manual_df[manual_df['video_name'] == sys_row['video_name']]
        if same_video.empty:
            matched_name = _best_video_name_match(sys_row['video_name'], set(manual_df['video_name']))
            if matched_name is not None:
                same_video = manual_df[manual_df['video_name'] == matched_name]

        if same_video.empty or pd.isna(sys_row['time_sec']):
            continue

        available = same_video[~same_video.index.isin(matched_manual)].copy()
        available = available[available['time_sec'].notna()]
        if available.empty:
            continue

        available['abs_dt'] = (available['time_sec'] - sys_row['time_sec']).abs()
        best_idx = available['abs_dt'].idxmin()
        best_row = available.loc[best_idx]
        if best_row['abs_dt'] > time_tolerance_sec:
            continue

        matched_manual.add(best_idx)
        system_df.at[sys_idx, 'matched_manual_index'] = int(best_idx)
        system_df.at[sys_idx, 'manual_label'] = best_row['manual_label']
        system_df.at[sys_idx, 'manual_notes'] = best_row.get('manual_notes', '')
        system_df.at[sys_idx, 'failure_category'] = best_row.get('failure_category', '')
        system_df.at[sys_idx, 'comparison_status'] = (
            'MATCH'
            if best_row['manual_label'] == sys_row['predicted_label']
            else 'LABEL_MISMATCH'
        )

    missed_rows = manual_df.loc[~manual_df.index.isin(matched_manual)].copy()
    if not missed_rows.empty:
        missed_rows['comparison_status'] = 'MISSED_BY_SYSTEM'

    summary_rows = [
        ('system_events', int(len(system_df))),
        ('manual_events', int(len(manual_df))),
        ('matches', int((system_df['comparison_status'] == 'MATCH').sum())),
        ('label_mismatches', int((system_df['comparison_status'] == 'LABEL_MISMATCH').sum())),
        ('unmatched_system_events', int((system_df['comparison_status'] == 'UNMATCHED_SYSTEM_EVENT').sum())),
        ('missed_manual_events', int(len(missed_rows))),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=['metric', 'value'])

    return system_df, summary_df


# ===========CELL 9=======

def run_tracker(tracker_name, video_path=VIDEO_PATH, show_every_n=5, display=True, return_event_log=False):
    """
    Process the full video with the given tracker.

    Auth decision flow (retroactive window):
      1. Person's foot enters crossing zone → pending_auth_check = True
      2. Every subsequent frame while pending:
           - If scanner fired for this ID → VALID (immediate)
           - If another person's auth exists but not this person's → TAILGATE (immediate)  [FIX #3]
           - If AUTH_RETROACTIVE_FRAMES elapsed with no scanner → TAILGATE
      3. If person exits zone before decision:
           - If scanner fired during their time in zone → VALID
           - Otherwise → TAILGATE (caught on exit)

    FIXES applied vs original:
      FIX #1  Zone entry uses >= instead of == so detection gaps don't skip the check.
      FIX #2  AUTH_RETROACTIVE_FRAMES reduced 45 → 20 so fast tailgaters are caught
              before they exit the zone.
      FIX #3  Immediate tailgate flag when an unmatched auth exists in active_auths —
              this is the primary fix for the female-follows-male scenario.
      FIX #4  Stability check is skipped for persons already inside the gate so
              new tracks that appear mid-zone are still evaluated.
    """
    # ── Validate ROIs ─────────────────────────────────────────
    if roi_state['entry'] is None or roi_state['exit'] is None or len(roi_state['zone']) != 4:
        raise RuntimeError(
            f'ROIs not fully set:\n'
            f'  entry={roi_state["entry"]}\n'
            f'  exit={roi_state["exit"]}\n'
            f'  zone={roi_state["zone"]}\n'
            f'Go back to Cell 4 and draw all 3 zones.'
        )

    entry_roi     = roi_state['entry']
    exit_roi      = roi_state['exit']
    crossing_zone = np.array(roi_state['zone'], dtype=np.int32)
    gband         = gate_band(crossing_zone)
    video_meta    = inspect_video(video_path)
    fps_nominal   = video_meta['fps']
    thresholds    = runtime_thresholds(fps_nominal)

    # ── Init ──────────────────────────────────────────────────
    mot          = make_tracker(tracker_name, fps=fps_nominal)
    e_sc         = Scanner(entry_roi, 'ENTRY', crossing_zone=crossing_zone)
    x_sc         = Scanner(exit_roi,  'EXIT',  crossing_zone=crossing_zone)
    idm          = IDMerger()
    ptrackers    = {}
    active_auths = []
    auth_log     = []
    debug_notes  = []
    total_auths = valid_crossings = tailgate_events = timeout_tailgating = 0

    cap          = cv2.VideoCapture(str(video_path))
    total_frames = video_meta['total_frames']
    fi           = 0
    t0           = time.time()

    # ── Display widgets ───────────────────────────────────────
    img_w = lbl_w = None
    if display:
        img_w = widgets.Image(format='jpeg', width=780)
        lbl_w = widgets.Label(value=f'Starting {tracker_name.upper()}...')
        ipy_display(widgets.VBox([lbl_w, img_w]))
    pbar  = tqdm(total=total_frames, desc=tracker_name, leave=True)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = resize_frame(frame)
        fi   += 1
        pbar.update(1)

        # ── Detection ──────────────────────────────────────────
        yd = run_yolo(frame)
        any_near = any(
            near_gate((int(d[0]),int(d[1]),int(d[2]),int(d[3])), gband)
            for d in yd
        )
        rd   = run_rfdetr(frame, gband) if any_near else np.empty((0,6), dtype=np.float32)
        dets = merge_dets(yd, rd)

        # ── Tracking ───────────────────────────────────────────
        tracks = mot.update(dets, frame)
        pb = {};  pf = {}
        for t in tracks:
            x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])
            if not filter_det(x1,y1,x2,y2): continue
            cid  = idm.resolve(tid)
            bbox = (x1,y1,x2,y2)
            pb[cid] = bbox
            pf[cid] = foot_point(bbox)
            if cid not in ptrackers:
                ptrackers[cid] = PersonTracker(cid)
            ptrackers[cid].update(bbox, fi)

        idm.check(set(pf.keys()), pf, fi)
        for tid, foot in pf.items():
            idm.update(tid, foot, fi)

        # ── Scanner activations ────────────────────────────────
        for sc_obj, direction in [(e_sc,'ENTRY'), (x_sc,'EXIT')]:
            act = sc_obj.check(
                pf, pb, ptrackers, fi,
                active_auths=active_auths,
                scanner_cooldown_frames=thresholds['scanner_cooldown_frames']
            )
            if act:
                auth = Authorization(f'A{total_auths}', direction, fi, act)
                active_auths.append(auth)
                total_auths += 1
                if len(debug_notes) < DEBUG_PRINT_LIMIT:
                    debug_notes.append(
                        f'F{fi}: {direction} scan assigned to ID {act}'
                    )

        # ── Gate crossing logic ────────────────────────────────
        piz = 0
        for tid in list(pb.keys()):
            pt = ptrackers.get(tid)
            if pt is None: continue

            sbbox   = pt.smoothed() or pb[tid]
            overlap_in_gate = bbox_in_polygon(sbbox, crossing_zone) >= GATE_OVERLAP_MIN
            foot_inside_gate = foot_in_polygon(sbbox, crossing_zone)
            in_gate = overlap_in_gate or foot_inside_gate

            # Track gate dwell frames
            if in_gate:
                piz += 1
                pt.pending_gate_frames += 1
                pt.missed_gate_frames = 0
            else:
                pt.missed_gate_frames += 1
                if pt.missed_gate_frames >= thresholds['gate_exit_grace_frames']:
                    pt.pending_gate_frames = 0

            # FIX #4: Skip stability check for persons already in the gate.
            # New tracks appearing mid-zone must still be evaluated — not
            # silently ignored because they haven't accumulated MIN_STABLE_FRAMES.
            if not pt.stable(thresholds['min_stable_frames']) and not in_gate:
                continue

            # ── ZONE ENTRY: start retroactive auth window ──────
            # FIX #1: Use >= instead of == so a detection gap (e.g. frames
            # jump from 4→6) cannot permanently skip the entry trigger.
            if (
                pt.pending_gate_frames >= thresholds['zone_dwell_min_frames']
                and not pt.is_in_gate
                and not pt.has_crossed
            ):
                pt.is_in_gate         = True
                pt.first_seen         = fi
                pt.pending_auth_check = True
                pt.zone_entries      += 1

            # ── AUTH CHECK: runs every frame while pending ─────
            if pt.pending_auth_check and not pt.has_crossed:
                auth_found = False

                # Look for an auth created by this specific person
                for auth in active_auths:
                    if auth.active and not auth.used and auth.scanner_id == tid:
                        auth.used             = True
                        auth.active           = False
                        pt.used_auth          = True
                        pt.has_crossed        = True
                        pt.auth_id            = auth.auth_id
                        pt.pending_auth_check = False
                        pt.decision_reason    = 'matched scanner auth while in gate'
                        valid_crossings      += 1
                        active_auths.remove(auth)
                        auth_log.append({
                            'frame': fi, 'id': tid,
                            'result': 'VALID', 'auth': auth.auth_id,
                            'reason': pt.decision_reason
                        })
                        auth_found = True
                        break

                if not auth_found:
                    # FIX #3: IMMEDIATE tailgate flag.
                    # If there are active auths in active_auths but NONE
                    # belong to this person, someone else scanned — this
                    # person is piggybacking. Flag immediately; do not wait
                    # for the retroactive window to expire.
                    # This is the primary fix for the female-follows-male case:
                    #   male scans → his auth is created → male crosses → his
                    #   auth is consumed. Female enters zone: active_auths is
                    #   now empty, so she falls through to the window check.
                    #   BUT if she enters while his auth is still active (he
                    #   hasn't crossed yet), we catch her here instantly.
                    unmatched_auths_exist = any(
                        a.active and not a.used and a.scanner_id != tid
                        for a in active_auths
                    )
                    if unmatched_auths_exist:
                        tailgate_events      += 1
                        pt.has_crossed        = True
                        pt.pending_auth_check = False
                        pt.decision_reason    = 'entered gate while another active auth existed'
                        auth_log.append({
                            'frame': fi, 'id': tid,
                            'result': 'TAILGATE (immediate — unmatched auth)',
                            'auth': '--',
                            'reason': pt.decision_reason
                        })
                        print(f'[TAILGATE IMMEDIATE] F{fi}: ID {tid} flagged — '
                              f'entered zone while another person\'s auth is active')
                    else:
                        # No auths at all — use the retroactive window.
                        frames_waiting = fi - (pt.first_seen or fi)
                        if frames_waiting >= thresholds['auth_retroactive_frames']:
                            tailgate_events      += 1
                            pt.has_crossed        = True
                            pt.pending_auth_check = False
                            pt.decision_reason    = 'no scan observed during retroactive gate window'
                            auth_log.append({
                                'frame': fi, 'id': tid,
                                'result': 'TAILGATE', 'auth': '--',
                                'reason': pt.decision_reason
                            })

            # ── ZONE EXIT ──────────────────────────────────────
            elif (not in_gate) and pt.is_in_gate and pt.missed_gate_frames >= thresholds['gate_exit_grace_frames']:
                pt.is_in_gate          = False
                pt.pending_gate_frames = 0
                pt.first_seen          = None

                if pt.pending_auth_check and not pt.has_crossed:
                    pt.pending_auth_check = False

                    auth_found = False
                    for auth in active_auths:
                        if auth.active and not auth.used and auth.scanner_id == tid:
                            auth.used      = True
                            auth.active    = False
                            pt.used_auth   = True
                            pt.has_crossed = True
                            pt.auth_id     = auth.auth_id
                            pt.decision_reason = 'matched scanner auth on zone exit'
                            valid_crossings += 1
                            active_auths.remove(auth)
                            auth_log.append({
                                'frame': fi, 'id': tid,
                                'result': 'VALID (on exit)', 'auth': auth.auth_id,
                                'reason': pt.decision_reason
                            })
                            auth_found = True
                            break

                    if not auth_found:
                        tailgate_events += 1
                        pt.has_crossed   = True
                        pt.decision_reason = 'left gate before any scan was matched'
                        auth_log.append({
                            'frame': fi, 'id': tid,
                            'result': 'TAILGATE (fast)', 'auth': '--',
                            'reason': pt.decision_reason
                        })

        # ── Auth timeouts ──────────────────────────────────────
        for auth in list(active_auths):
            if auth.expired(fi, thresholds['auth_timeout_frames']):
                if not auth.used:
                    timeout_tailgating += 1
                active_auths.remove(auth)

        # ── Inline display every N frames ──────────────────────
        if display and fi % show_every_n == 0:
            fps_a = fi / max(time.time() - t0, 0.001)
            ann   = annotate(
                frame, pb, ptrackers,
                entry_roi, exit_roi, crossing_zone, gband,
                active_auths, tracker_name,
                fi, total_frames, fps_a,
                total_auths, valid_crossings,
                tailgate_events, timeout_tailgating, piz,
                thresholds
            )
            img_w.value = frame_to_jpeg(ann)
            lbl_w.value = (
                f'{tracker_name.upper()} | Frame {fi}/{total_frames} | '
                f'FPS {fps_a:.1f} | Valid:{valid_crossings}  Tailgate:{tailgate_events}'
            )

    pbar.close()
    cap.release()
    elapsed = time.time() - t0

    result = {
        'tracker':            tracker_name,
        'video_path':         str(video_path),
        'fps':                round(fi / max(elapsed, 0.001), 1),
        'video_fps':          round(fps_nominal, 2),
        'duration_sec':       round(video_meta['duration_sec'], 2),
        'total_auths':        total_auths,
        'valid_crossings':    valid_crossings,
        'tailgate_events':    tailgate_events,
        'timeout_tailgating': timeout_tailgating,
        'id_merges':          len(idm.aliases),
        'debug_notes':        debug_notes,
    }

    print(f'\n✓ {tracker_name.upper()} done')
    print(f'  FPS:{result["fps"]}  Auths:{total_auths}  Valid:{valid_crossings}  '
          f'Tailgate:{tailgate_events}  Timeout:{timeout_tailgating}  '
          f'ID merges:{result["id_merges"]}')

    if auth_log:
        print('\nEvent log:')
        ipy_display(pd.DataFrame(auth_log))

    if return_event_log:
        result['event_log'] = pd.DataFrame(auth_log)

    return result

print('✓ Cell 6 — run_tracker() defined')


# ======CELL 10==========

def benchmark_trackers(video_path=VIDEO_PATH, tracker_names=None, show_every_n=5, display=True):
    tracker_names = tracker_names or TRACKER_NAMES
    all_results = []
    for name in tracker_names:
        print(f'\n{"━"*55}')
        print(f'  Running: {name.upper()} on {Path(video_path).name}')
        print(f'{"━"*55}')
        try:
            r = run_tracker(
                name,
                video_path=video_path,
                show_every_n=show_every_n,
                display=display,
                return_event_log=False
            )
            all_results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ✗ {name} failed: {e}')
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)
    return df

def benchmark_sample_set(video_paths=None, tracker_name=DEFAULT_TRACKER, show_every_n=15):
    video_paths = video_paths or SAMPLE_VIDEO_SET
    rows = []
    for vp in video_paths:
        print(f'\n{"="*60}')
        print(f'Analyzing sample video: {Path(vp).name}')
        print(f'{"="*60}')
        try:
            result = run_tracker(
                tracker_name,
                video_path=vp,
                show_every_n=show_every_n,
                display=False,
                return_event_log=False
            )
            rows.append(result)
        except Exception as e:
            rows.append({
                'tracker': tracker_name,
                'video_path': str(vp),
                'error': str(e)
            })
            print(f'  ✗ failed: {e}')
    return pd.DataFrame(rows)


def run_video_audit(video_path=VIDEO_PATH, tracker_name=DEFAULT_TRACKER, show_every_n=15,
                    output_dir=EVAL_DIR, display=False):
    """
    Run the detector on one video and save review CSVs for manual auditing.
    """
    result = run_tracker(
        tracker_name,
        video_path=video_path,
        show_every_n=show_every_n,
        display=display,
        return_event_log=True
    )
    artifact_paths = save_review_artifacts(result, output_dir=output_dir, include_template=True)
    print('\nSaved review artifacts:')
    for label, path in artifact_paths.items():
        print(f'  {label}: {path}')
    return result, artifact_paths


def review_accuracy(manual_labels_csv, result=None, system_events_csv=None, time_tolerance_sec=1.0):
    """
    Compare hand labels against either an in-memory run result or a saved system CSV.
    """
    if result is None and system_events_csv is None:
        raise ValueError('Pass either result=... or system_events_csv=...')

    if result is not None:
        system_events = prepare_event_log_df(
            result.get('event_log'),
            video_path=result['video_path'],
            video_fps=result['video_fps'],
            tracker_name=result['tracker']
        )
    else:
        system_events = pd.read_csv(system_events_csv)

    manual_labels = load_manual_labels(manual_labels_csv)
    comparison_df, summary_df = compare_manual_vs_system(
        system_events=system_events,
        manual_labels=manual_labels,
        time_tolerance_sec=time_tolerance_sec
    )

    print('Accuracy summary:')
    ipy_display(summary_df)

    mismatches = comparison_df[
        comparison_df['comparison_status'].isin(['LABEL_MISMATCH', 'UNMATCHED_SYSTEM_EVENT'])
    ]
    if not mismatches.empty:
        print('\nSystem-side mismatches:')
        ipy_display(mismatches)

    return comparison_df, summary_df

def diagnostic_questions():
    questions = [
        'Are scanner ROIs positioned before and after the gate, with minimal overlap into the crossing zone?',
        'Do the same people keep the same track ID from scanner approach through gate exit, or do IDs reset mid-crossing?',
        'Are missed detections happening when two people overlap, when a person turns sideways, or under motion blur?',
        'Is the system labeling a person as tailgating because no scan was seen, or because another person still had an active auth?',
        'Do thresholds need to be calibrated for 15 fps video instead of the original 30 fps assumptions?',
        'Is the benchmark measuring against hand-labeled ground truth, or only reporting internal event counters?',
        'Are false positives coming from scanner assignment errors, zone geometry mistakes, or tracker fragmentation?',
        'Are some videos harder because the gate width, camera zoom, or crowd density differs from the single-video tuning setup?'
    ]
    print('Accuracy review checklist:')
    for i, q in enumerate(questions, start=1):
        print(f'{i}. {q}')

print('ROI check:')
print(f'  Entry : {roi_state["entry"]}')
print(f'  Exit  : {roi_state["exit"]}')
print(f'  Zone  : {roi_state["zone"]}')

video_meta = inspect_video(VIDEO_PATH)
usable_trackers, tracker_notes = available_trackers(video_meta['fps'])
print(f'\nVideo: {Path(VIDEO_PATH).name}')
print(f'  Resolution: {video_meta["width"]} x {video_meta["height"]}')
print(f'  FPS: {video_meta["fps"]:.2f}')
print(f'  Duration: {video_meta["duration_sec"]:.1f}s')
print('\nTracker availability:')
for tname in TRACKER_PRIORITY:
    print(f'  {tname}: {tracker_notes.get(tname, "not checked")}')

print('\nAccuracy review prompts:')
diagnostic_questions()

missing = [k for k, v in roi_state.items() if v is None or v == []]
if missing:
    print(f'\n⚠ Missing: {missing} — go back to Cell 4 and draw these zones')
elif not usable_trackers:
    print('\n⚠ No compatible trackers are available in the current environment.')
else:
    print('\n✓ All ROIs set — starting benchmark...\n')
    df = benchmark_trackers(
        video_path=VIDEO_PATH,
        tracker_names=usable_trackers,
        show_every_n=5,
        display=True
    )

    if not df.empty:
        df = df.set_index('tracker')
        df.index.name = 'Tracker'

        print('\n' + '='*55)
        print('  BENCHMARK RESULTS')
        print('='*55)
        ipy_display(
            df.style
            .highlight_max(subset=['fps', 'valid_crossings'], color='lightgreen')
            .highlight_min(subset=['tailgate_events','timeout_tailgating','id_merges'],
                           color='lightgreen')
            .format({'fps': '{:.1f}'})
        )

        df['score'] = df['fps']*0.3 - df['id_merges']*5 - df['timeout_tailgating']*2
        best = df['score'].idxmax()
        print(f'\n★ Recommended tracker: {best.upper()}')
    else:
        print('No results — all trackers failed')
