import os
import csv
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from ultralytics import YOLO

# ============================ 2D/3D Classifier (RGB + 10-d features) ============================
class RebarNet(nn.Module):
    def __init__(self, feat_dim: int = 10):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()  # 512-d
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512 + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, img, feat):
        f_img = self.backbone(img)           # (B, 512)
        f = torch.cat([f_img, feat], dim=1)  # (B, 512+feat_dim)
        return self.classifier(f)            # (B, 2)


# ============================ Geometry features from lines ============================
def angle_of_line(line: List[List[float]]) -> float:
    (x1, y1), (x2, y2) = line
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    theta = abs(theta)
    if theta > 180:
        theta -= 180
    return theta


def compute_mean_angle_diff(angles: List[float]) -> float:
    if len(angles) < 2:
        return 0.0
    diffs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            diffs.append(d)
    return float(np.mean(diffs))


def compute_top_bottom_diff(lines: List[List[List[float]]], angles: List[float]) -> float:
    if not lines:
        return 0.0
    y_mids = [(l[0][1] + l[1][1]) / 2.0 for l in lines]
    split_y = float(np.median(y_mids))

    top_angles, bottom_angles = [], []
    for y_mid, angle in zip(y_mids, angles):
        if y_mid < split_y:
            top_angles.append(angle)
        else:
            bottom_angles.append(angle)

    if not top_angles or not bottom_angles:
        return 0.0

    d = abs(float(np.mean(top_angles)) - float(np.mean(bottom_angles)))
    return float(min(d, 180 - d))


# ============================ Lines from YOLO nodes (PCA-neighbor) ============================
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def vector_aligned_with_pc(p: np.ndarray, q: np.ndarray, pc: np.ndarray, tolerance_deg: float) -> bool:
    v = q - p
    if np.linalg.norm(v) < 1e-6:
        return False
    v_u = _unit(v)
    pc_u = _unit(pc)
    cosv = float(np.clip(np.dot(v_u, pc_u), -1.0, 1.0))
    ang = math.degrees(math.acos(abs(cosv)))  # allow +/- pc direction
    return ang <= tolerance_deg


def dedup_lines(lines: List[List[List[float]]], round_ndigits: int = 2) -> List[List[List[float]]]:
    seen = set()
    out = []
    for l in lines:
        a = (round(l[0][0], round_ndigits), round(l[0][1], round_ndigits))
        b = (round(l[1][0], round_ndigits), round(l[1][1], round_ndigits))
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        out.append([[float(l[0][0]), float(l[0][1])], [float(l[1][0]), float(l[1][1])]])
    return out


def mode_angle_outlier_filter(lines: List[List[List[float]]], threshold_deg: float = 10.0) -> List[List[List[float]]]:
    if not lines or threshold_deg <= 0:
        return lines

    radians = []
    for l in lines:
        p1 = np.array(l[0], dtype=np.float32)
        p2 = np.array(l[1], dtype=np.float32)
        v = p2 - p1
        radians.append(float(np.arctan2(v[1], v[0])))

    bins = 36
    hist, bin_edges = np.histogram(radians, bins=bins, range=(-np.pi, np.pi))
    max_bin = int(np.argmax(hist))
    mode = float(bin_edges[max_bin] + (np.pi / bins))

    thr = threshold_deg * np.pi / 180.0

    def norm_radian(r: float) -> float:
        while r <= -np.pi:
            r += 2 * np.pi
        while r > np.pi:
            r -= 2 * np.pi
        return r

    keep = []
    for l, r in zip(lines, radians):
        if abs(norm_radian(r - mode)) <= thr:
            keep.append(l)
    return keep


def get_bounding_boxes_yolo(model: Any, image_path: str, conf: float = 0.25) -> List[List[float]]:
    res = model.predict(source=image_path, conf=conf, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    return xyxy.astype(np.float32).tolist()


def get_vertice_from_box(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def generate_lines_from_nodes(
    image_path: str,
    yolo_model: Any,
    conf: float = 0.25,
    tolerance_deg: float = 30.0,
    prune_angle_deg: float = 10.0,
) -> Dict[str, Any]:
    boxes = get_bounding_boxes_yolo(yolo_model, image_path, conf=conf)
    if len(boxes) < 2:
        return {"shapes": [], "num_nodes": len(boxes)}

    vertices = np.array([get_vertice_from_box(b) for b in boxes], dtype=np.float32)
    mean = np.mean(vertices, axis=0)
    vc = vertices - mean

    _, _, Vh = np.linalg.svd(vc, full_matrices=False)
    pc1 = Vh[0]
    pc2 = Vh[1] if Vh.shape[0] > 1 else np.array([0.0, 1.0], dtype=np.float32)

    pc1_lines, pc2_lines = [], []
    n = len(vertices)

    for i in range(n):
        best1_j, best1_d = None, float("inf")
        best2_j, best2_d = None, float("inf")
        for j in range(n):
            if i == j:
                continue
            p = vertices[i]
            q = vertices[j]
            d = float(np.linalg.norm(q - p))
            if d < 1e-6:
                continue

            if vector_aligned_with_pc(p, q, pc1, tolerance_deg) and d < best1_d:
                best1_j, best1_d = j, d
            if vector_aligned_with_pc(p, q, pc2, tolerance_deg) and d < best2_d:
                best2_j, best2_d = j, d

        if best1_j is not None:
            pc1_lines.append([vertices[i].tolist(), vertices[best1_j].tolist()])
        if best2_j is not None:
            pc2_lines.append([vertices[i].tolist(), vertices[best2_j].tolist()])

    pc1_lines = mode_angle_outlier_filter(dedup_lines(pc1_lines), threshold_deg=prune_angle_deg)
    pc2_lines = mode_angle_outlier_filter(dedup_lines(pc2_lines), threshold_deg=prune_angle_deg)

    shapes = []
    if pc1_lines:
        shapes.append({"lines": pc1_lines})
    if pc2_lines:
        shapes.append({"lines": pc2_lines})

    return {"shapes": shapes, "num_nodes": len(boxes)}


def flatten_lines(line_json: Dict[str, Any]) -> List[List[List[float]]]:
    lines = []
    for s in line_json.get("shapes", []):
        lines.extend(s.get("lines", []))
    return lines


# ============================ Depth features ============================
def find_matching_depth(depth_dir: str, base: str) -> Optional[str]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".npy", ".JPG", ".PNG", ".JPEG"]
    for ext in exts:
        p = os.path.join(depth_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def read_depth_any(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(np.float32)


def depth_global_features(depth: np.ndarray) -> Tuple[float, float, float, float, float]:
    h, w = depth.shape[:2]
    valid = depth > 0
    valid_ratio = float(valid.mean()) if h * w > 0 else 0.0

    if valid.any():
        vals = depth[valid]
        med = float(np.median(vals))
        std = float(np.std(vals))
    else:
        med, std = 0.0, 0.0

    mid = h // 2
    top_v = valid[:mid, :]
    bot_v = valid[mid:, :]
    if top_v.any() and bot_v.any():
        top_med = float(np.median(depth[:mid, :][top_v]))
        bot_med = float(np.median(depth[mid:, :][bot_v]))
        top_bot_diff = float(abs(top_med - bot_med))
    else:
        top_bot_diff = 0.0

    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(dx, dy)
    grad_mean = float(np.mean(g[valid])) if valid.any() else float(np.mean(g))

    return valid_ratio, med, std, top_bot_diff, grad_mean


# 交點相關（給 ROI）
def segment_intersection(p1, p2, p3, p4) -> Optional[Tuple[float, float]]:
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4

    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den

    def within(a, b, c):
        return min(a, b) - 1e-6 <= c <= max(a, b) + 1e-6

    if within(x1, x2, px) and within(y1, y2, py) and within(x3, x4, px) and within(y3, y4, py):
        return float(px), float(py)
    return None


def dedup_points(points: List[Tuple[float, float]], grid: int = 5) -> List[Tuple[int, int]]:
    s = set()
    out = []
    for x, y in points:
        xi = int(round(x / grid) * grid)
        yi = int(round(y / grid) * grid)
        if (xi, yi) in s:
            continue
        s.add((xi, yi))
        out.append((xi, yi))
    return out


def roi_depth_features(
    depth: np.ndarray,
    lines: List[List[List[float]]],
    roi_half: int = 10,
    ring_thick: int = 6,
) -> Tuple[float, float, float]:
    if depth is None or depth.size == 0 or not lines:
        return 0.0, 0.0, 0.0

    pts = []
    segs = [((l[0][0], l[0][1]), (l[1][0], l[1][1])) for l in lines]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            p = segment_intersection(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            if p is not None:
                pts.append(p)

    centers = dedup_points(pts, grid=5)
    if not centers:
        return 0.0, 0.0, 0.0

    h, w = depth.shape[:2]
    valid = depth > 0

    roi_valid_ratios = []
    roi_stds = []
    roi_ring_diffs = []

    outer_half = roi_half + ring_thick

    for cx, cy in centers:
        x1 = max(0, cx - roi_half); x2 = min(w, cx + roi_half + 1)
        y1 = max(0, cy - roi_half); y2 = min(h, cy + roi_half + 1)
        roi = depth[y1:y2, x1:x2]
        roi_v = valid[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_valid_ratios.append(float(roi_v.mean()))

        if roi_v.any():
            roi_vals = roi[roi_v]
            roi_stds.append(float(np.std(roi_vals)))
            roi_med = float(np.median(roi_vals))
        else:
            roi_stds.append(0.0)
            roi_med = 0.0

        ox1 = max(0, cx - outer_half); ox2 = min(w, cx + outer_half + 1)
        oy1 = max(0, cy - outer_half); oy2 = min(h, cy + outer_half + 1)
        outer = depth[oy1:oy2, ox1:ox2]
        outer_v = valid[oy1:oy2, ox1:ox2]

        ring_mask = outer_v.copy()
        ix1 = (x1 - ox1); ix2 = (x2 - ox1)
        iy1 = (y1 - oy1); iy2 = (y2 - oy1)
        ring_mask[iy1:iy2, ix1:ix2] = False

        if ring_mask.any():
            ring_vals = outer[ring_mask]
            ring_med = float(np.median(ring_vals))
            roi_ring_diffs.append(float(abs(roi_med - ring_med)))
        else:
            roi_ring_diffs.append(0.0)

    if not roi_valid_ratios:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(roi_valid_ratios)),
        float(np.mean(roi_stds)) if roi_stds else 0.0,
        float(np.mean(roi_ring_diffs)) if roi_ring_diffs else 0.0,
    )


# ============================ Preprocess RGB ============================
def preprocess_rgb(img_path: str, img_size: int = 224) -> torch.Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # CHW


def load_state_dict_flexible(model: nn.Module, state: dict):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--folder", default=script_dir, help="RGB 圖所在資料夾（預設=程式同層）")
    ap.add_argument("--depth_dir", default="Depth", help="深度圖資料夾名稱（預設=Depth，位於 folder 內）")

    ap.add_argument("--node_weights", default="best.pt", help="YOLO node 權重（best.pt）")
    ap.add_argument("--cls_weights", default="2d3d_with_depth.pt", help="2D/3D 分類權重（state_dict）")

    ap.add_argument("--out_csv", default="pred_strengthened.csv", help="輸出 CSV")
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tol_deg", type=float, default=30.0)
    ap.add_argument("--prune_deg", type=float, default=10.0)

    # Two-stage threshold（強化原本版本）
    ap.add_argument("--low", type=float, default=0.30, help="P3D <= low -> 直接判 2D")
    ap.add_argument("--high", type=float, default=0.70, help="P3D >= high -> 直接判 3D")

    # Depth gate 門檻（只在灰區介入）
    ap.add_argument("--min_depth_valid", type=float, default=0.20)
    ap.add_argument("--thr_depth_topbot", type=float, default=0.15)  # (0~1) after /255
    ap.add_argument("--thr_depth_grad", type=float, default=0.20)    # (0~1) after /255
    ap.add_argument("--thr_roi_ring", type=float, default=0.15)      # (0~1) after /255

    ap.add_argument("--roi_half", type=int, default=10)
    ap.add_argument("--ring_thick", type=int, default=6)

    args = ap.parse_args()

    folder = args.folder
    depth_dir = args.depth_dir if os.path.isabs(args.depth_dir) else os.path.join(folder, args.depth_dir)
    node_w = args.node_weights if os.path.isabs(args.node_weights) else os.path.join(folder, args.node_weights)
    cls_w  = args.cls_weights  if os.path.isabs(args.cls_weights)  else os.path.join(folder, args.cls_weights)
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(folder, args.out_csv)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"Depth folder not found: {depth_dir}")
    if not os.path.exists(node_w):
        raise FileNotFoundError(f"node_weights not found: {node_w}")
    if not os.path.exists(cls_w):
        raise FileNotFoundError(f"cls_weights not found: {cls_w}")

    # load YOLO node model
    node_model = YOLO(node_w)

    # load classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_model = RebarNet(feat_dim=10).to(device)
    state = torch.load(cls_w, map_location=device)
    load_state_dict_flexible(cls_model, state)
    cls_model.eval()

    # enumerate RGB images in same folder
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    if not images:
        print("[WARN] 找不到 RGB 圖（jpg/png/bmp）")
        return

    rows = []
    miss_depth = 0

    for fname in images:
        img_path = os.path.join(folder, fname)
        base = os.path.splitext(fname)[0]

        depth_path = find_matching_depth(depth_dir, base)
        depth_found = 1 if depth_path is not None else 0
        if depth_path is None:
            miss_depth += 1

        # 1) nodes -> lines
        line_json = generate_lines_from_nodes(
            img_path, node_model,
            conf=args.conf,
            tolerance_deg=args.tol_deg,
            prune_angle_deg=args.prune_deg,
        )
        lines = flatten_lines(line_json)

        # 2) RGB geom features
        if lines:
            angles = [angle_of_line(l) for l in lines]
            mean_diff = compute_mean_angle_diff(angles)
            top_bottom = compute_top_bottom_diff(lines, angles)
        else:
            mean_diff = 0.0
            top_bottom = 0.0

        # normalize /90
        mean_n = mean_diff / 90.0
        top_n  = top_bottom / 90.0

        # 3) depth + roi features
        if depth_found:
            depth = read_depth_any(depth_path)
            d_valid_ratio, d_med, d_std, d_topbot, d_grad = depth_global_features(depth)
            roi_vr, roi_std, roi_ring = roi_depth_features(
                depth, lines, roi_half=args.roi_half, ring_thick=args.ring_thick
            )
        else:
            d_valid_ratio = d_med = d_std = d_topbot = d_grad = 0.0
            roi_vr = roi_std = roi_ring = 0.0

        # 4) model inference
        img_t = preprocess_rgb(img_path, args.img_size).unsqueeze(0).to(device)

        feat_t = torch.tensor([[
            float(mean_n),
            float(top_n),
            float(d_valid_ratio),
            float(d_med),
            float(d_std),
            float(d_topbot),
            float(d_grad),
            float(roi_vr),
            float(roi_std),
            float(roi_ring),
        ]], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = cls_model(img_t, feat_t)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            p2d = float(prob[0])
            p3d = float(prob[1])

        # =========================
        # ✅ Two-stage 강화版判斷（重點在這裡）
        # =========================
        reason = ""
        if p3d >= args.high:
            pred = 1
            reason = f"model_conf_high(p3d={p3d:.3f}>=high={args.high})"
        elif p3d <= args.low:
            pred = 0
            reason = f"model_conf_low(p3d={p3d:.3f}<=low={args.low})"
        else:
            # 灰區：Depth gate
            # 你的 depth 是 jpg -> 0~255 強度，所以做 /255 正規化再比較門檻
            d_topbot_n = float(np.clip(d_topbot, 0, 255) / 255.0)
            d_grad_n   = float(np.clip(d_grad,   0, 255) / 255.0)
            roi_ring_n = float(np.clip(roi_ring, 0, 255) / 255.0)

            gate = (d_valid_ratio >= args.min_depth_valid) and (
                (d_topbot_n >= args.thr_depth_topbot) or
                (d_grad_n   >= args.thr_depth_grad) or
                (roi_ring_n >= args.thr_roi_ring)
            )

            if gate:
                pred = 1
                reason = (
                    f"depth_gate(p3d_in_gray={p3d:.3f}, "
                    f"valid={d_valid_ratio:.3f}, "
                    f"topbotN={d_topbot_n:.3f}, gradN={d_grad_n:.3f}, ringN={roi_ring_n:.3f})"
                )
            else:
                pred = 0
                reason = (
                    f"gray_no_gate(p3d={p3d:.3f}, "
                    f"valid={d_valid_ratio:.3f}, "
                    f"topbotN={d_topbot_n:.3f}, gradN={d_grad_n:.3f}, ringN={roi_ring_n:.3f})"
                )

        pred_label = "2D" if pred == 0 else "3D"

        print(f"[OK] {fname} -> {pred_label} | p3d={p3d:.3f} | {reason}")

        rows.append([
            fname,
            depth_path if depth_path else "",
            pred_label,
            f"{p2d:.5f}",
            f"{p3d:.5f}",
            reason,
            f"{mean_n:.5f}",
            f"{top_n:.5f}",
            f"{d_valid_ratio:.5f}",
            f"{d_med:.5f}",
            f"{d_std:.5f}",
            f"{d_topbot:.5f}",
            f"{d_grad:.5f}",
            f"{roi_vr:.5f}",
            f"{roi_std:.5f}",
            f"{roi_ring:.5f}",
            str(depth_found),
            str(line_json.get("num_nodes", 0)),
            str(len(lines)),
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "depth_path", "pred", "p_2d", "p_3d", "reason",
            "mean_diff_norm", "top_bottom_norm",
            "depth_valid_ratio", "depth_median", "depth_std", "depth_top_bottom_diff", "depth_grad_mean",
            "roi_valid_ratio_mean", "roi_z_std_mean", "roi_vs_ring_diff_mean",
            "depth_found", "num_nodes", "num_lines"
        ])
        w.writerows(rows)

    print(f"\n[DONE] {len(rows)} images -> {out_csv}")
    if miss_depth > 0:
        print(f"[WARN] 有 {miss_depth} 張 RGB 找不到對應深度圖（Depth/ 必須同 basename）")


if __name__ == "__main__":
    main()
