import cv2
import numpy as np
from frames import FRAME_SIZE

# Resolution of the search (docs/needle_hold.md)
PLANE_ROTATIONS = 64
ARC_POINTS = 16
MASK_POINTS = 32
RADII = 16 * 2 ** (np.arange(17) / 4)
STITCH_FRAMES = 16
CENTRAL_ANGLES = np.radians(np.arange(10, 351, 10))
COARSE_RATIOS = np.arange(1, 33) / 32
FINE_RATIOS = np.arange(1, 129) / 128


def only_holder(frame):
    """The only instrument with grasp, or None"""
    holders = [i for i in frame["instruments"] if i["grasp"] is not None]
    return holders[0] if len(holders) == 1 else None


def ratio_frame(frame):
    """Observations of a frame for the registration, or None if it lacks the tip, the holder or the needle mask"""
    holder = only_holder(frame)
    tip, tail = frame["needle"]["needle_tip"], frame["needle"]["needle_tail"]
    polygons = [p for p in frame["needle_polygons"] if len(p) >= 3]
    if tip is None or holder is None or not polygons:
        return None
    width, height = FRAME_SIZE
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.array(p, np.int32) for p in polygons], 1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    index = np.linspace(0, len(xs) - 1, min(MASK_POINTS, len(xs))).astype(int)
    jaws = np.zeros((height, width), np.uint8)
    corners = [holder[k] for k in ("clasper_1", "clasper_2", "head")]
    if all(c is not None for c in corners):
        cv2.fillPoly(jaws, [np.array(corners, np.int32)], 1)
    return {
        "tip": np.array(tip, float),
        "grasp": np.array(holder["grasp"], float),
        "tail": None if tail is None else np.array(tail, float),
        "distance": cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3),
        "mask_points": np.stack([xs[index], ys[index]], 1).astype(float),
        "jaws": jaws,
    }


def frame_costs(frame, radius, central_angle, ratios):
    """Registration cost of a frame for each grasp ratio; inf where the ratio is impossible"""
    tip, grasp = frame["tip"], frame["grasp"]
    chord = grasp - tip
    chord_length = np.linalg.norm(chord)
    grasp_angles = ratios * central_angle
    # Chord length in 3D of the arc from the tip to the grasp point
    lengths = 2 * radius * np.sin(grasp_angles / 2)
    costs = np.full(len(ratios), np.inf)
    feasible = lengths >= chord_length
    if not feasible.any() or chord_length < 1:
        return costs
    angles, grasp_angles = (
        np.linspace(0, central_angle, ARC_POINTS),
        grasp_angles[feasible],
    )
    count = len(grasp_angles)
    # Arc points relative to the tip, in the chord direction and perpendicular to it within the needle plane
    arc = radius * np.stack([np.cos(angles) - 1, np.sin(angles)], 1)
    chord_dirs = np.stack([np.cos(grasp_angles) - 1, np.sin(grasp_angles)], 1)
    chord_dirs /= np.linalg.norm(chord_dirs, axis=1, keepdims=True)
    along = chord_dirs @ arc.T
    across = np.stack([-chord_dirs[:, 1], chord_dirs[:, 0]], 1) @ arc.T
    depth = np.sqrt(lengths[feasible] ** 2 - chord_length**2)
    rotations = np.linspace(0, 2 * np.pi, PLANE_ROTATIONS, endpoint=False)
    best = np.full(count, np.inf)
    height, width = frame["distance"].shape
    for sign in (1, -1):
        u = np.concatenate(
            [np.broadcast_to(chord, (count, 2)), sign * depth[:, None]], 1
        )
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        helper = np.where(
            np.abs(u[:, 2:]) < 0.9, np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        )
        e1 = np.cross(u, helper)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(u, e1)
        w = (
            np.cos(rotations)[None, :, None] * e1[:, None]
            + np.sin(rotations)[None, :, None] * e2[:, None]
        )
        # Image of the arc (ratio, rotation, point, xy) under weak perspective
        points = (
            tip
            + along[:, None, :, None] * u[:, None, None, :2]
            + across[:, None, :, None] * w[:, :, None, :2]
        )
        x = np.clip(np.round(points[..., 0]).astype(int), 0, width - 1)
        y = np.clip(np.round(points[..., 1]).astype(int), 0, height - 1)
        visible = frame["jaws"][y, x] == 0
        to_mask = (frame["distance"][y, x] * visible).sum(-1) / np.maximum(
            visible.sum(-1), 1
        )
        # Squared distances from the mask points to the arc points, (ratio, rotation, arc point, mask point)
        mask_points = frame["mask_points"]
        squared = (
            (points**2).sum(-1)[..., None]
            - 2 * points @ mask_points.T
            + (mask_points**2).sum(-1)
        )
        to_arc = np.sqrt(np.maximum(squared.min(2), 0)).mean(-1)
        cost = to_mask + to_arc
        if frame["tail"] is not None:
            cost = cost + np.linalg.norm(points[:, :, -1] - frame["tail"], axis=-1)
        best = np.minimum(best, cost.min(1))
    costs[feasible] = best
    return costs


def stitch_costs(stitch_frames, radius, central_angle, ratios):
    return np.mean(
        [frame_costs(f, radius, central_angle, ratios) for f in stitch_frames], 0
    )


def needle_hold_ratio(frames, windows):
    """Needle Hold Ratio of each stitch: (grasp ratio or None, number of frames)"""
    observed = []
    for start, entry, _ in windows:
        stitch_frames = [
            ratio_frame(f) for i, f in sorted(frames.items()) if start <= i <= entry
        ]
        stitch_frames = [f for f in stitch_frames if f is not None]
        # Evenly spaced frames of the window
        if len(stitch_frames) > STITCH_FRAMES:
            index = np.linspace(0, len(stitch_frames) - 1, STITCH_FRAMES)
            stitch_frames = [stitch_frames[i] for i in index.astype(int)]
        observed.append(stitch_frames)
    measured = [o for o in observed if o]
    if not measured:
        return [(None, 0)] * len(windows)
    # Needle shape of the video: the radius and central angle with the least total cost of the stitches
    totals = {
        (radius, central_angle): sum(
            stitch_costs(o, radius, central_angle, COARSE_RATIOS).min()
            for o in measured
        )
        for radius in RADII
        for central_angle in CENTRAL_ANGLES
    }
    radius, central_angle = min(totals, key=totals.get)
    if np.isinf(totals[radius, central_angle]):
        return [(None, 0)] * len(windows)
    out = []
    for o in observed:
        costs = stitch_costs(o, radius, central_angle, FINE_RATIOS) if o else [np.inf]
        if np.isinf(np.min(costs)):
            out.append((None, 0))
            continue
        out.append((float(FINE_RATIOS[np.argmin(costs)]), len(o)))
    return out


def ratio_score(value):
    """good if the grasp ratio is between 1/2 (1:1 hold) and 3/4 (3/4:1/4 hold), otherwise poor"""
    if value is None:
        return None
    return "good" if 0.5 <= value <= 0.75 else "poor"
