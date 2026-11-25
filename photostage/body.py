"""Prepare a photo's body rig: the person's own pixels, set up so the player can breathe, sway and gesture with them.

Run with the FlashHead interpreter; it needs no GPU. Nothing is generated: every pixel the player draws is a pixel of the
photo (moved a little, never invented), except for a patch of "plate" behind a hand that gestures, which is the cloth or
skin around the hand carried over the spot the hand uncovers.

 1. Breathing and sway: a smooth weight field over the person (the whole person above the shoulders, fading to nothing at
    the hips or at the picture's lower edge, and towards the silhouette so that no background is dragged along). The
    player moves the photo by that weight times a small, slowly changing offset, and the live head follows it.
 2. Hands: a hand whose picture is clean enough is cut out as its own layer with a plate behind it, so that the player
    can turn it at the wrist and slide it. A hand that cannot be cut out cleanly keeps still: the rig renders every hand
    at its extreme poses here and measures what a viewer would see (a ghost of the old hand, a rim of old cloth, a seam at
    the cuff, a patch of the plate without the cloth's grain, a hand stretched out of shape) before it allows the hand to
    move, and it gives the hand the largest share of its range that passes. Two hands that could pass over one another
    are not both allowed to move.

Everything is measured in the photo's own pixels and in face heights, so it does not depend on the photo's size or on
what the person wears.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# The player draws at most this many pixels per side, so nothing is analysed or stored finer than that.
WORK_SIDE = 1920
FIELD_SIDE = 512

# Offsets of the body at full weight, in face heights: sideways sway and the lift of a breath. About what a person
# standing for a camera does; the warp stays clean at several times this.
SWAY, LIFT = 0.04, 0.02
# The weight is 1 down to the shoulders and fades to 0 at the hips (or the lower edge); towards the silhouette it ramps up
# over EDGE_RAMP face heights (EDGE_FLOOR is the distance outside the silhouette where it starts), so the weights of the
# background and of the outline stay near 0 and the outline hardly moves against the background.
EDGE_RAMP, EDGE_FLOOR = 0.2, 0.01
# The face itself moves as one piece: the weight is 1 inside an ellipse of these radii (in face heights) that fades out over BLOB_FADE.
BLOB_RADII, BLOB_FADE = (0.62, 0.7), 0.15
# Where the shoulders sit under the face's bottom edge when the pose finder does not see them, in face heights.
SHOULDERS_BELOW = 0.5
# A person matte covering less or more of the picture than this is not believed.
MATTE_AREA = (0.03, 0.9)
POSE_VISIBLE = 0.5


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def disc(diameter):
    size = max(1, int(round(diameter))) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class Segmenter:
    """MediaPipe's person probability. Its model sees 256 px, so a large frame is read in overlapping squares."""

    def __init__(self):
        import mediapipe as mp

        self.model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def __call__(self, bgr):
        height, width = bgr.shape[:2]
        tall = height >= width
        side, length = (width, height) if tall else (height, width)
        starts = list(range(0, length - side + 1, max(1, side // 2)))
        if starts[-1] != length - side:
            starts.append(length - side)
        total = np.zeros((height, width), np.float32)
        weight = np.zeros_like(total)
        for start in starts:
            tile = bgr[start:start + side] if tall else bgr[:, start:start + side]
            probability = self.model.process(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).segmentation_mask
            # Trust a tile most at its centre, where the model sees the person whole.
            ramp = np.clip(1.0 - np.abs(np.arange(side) + 0.5 - side / 2) / (side / 2), 0.02, 1.0).astype(np.float32)
            ramp = ramp[:, None] if tall else ramp[None, :]
            window = (slice(start, start + side), slice(None)) if tall else (slice(None), slice(start, start + side))
            total[window] += probability * ramp
            weight[window] += ramp
        return total / weight


def refine(probability, guide):
    """Snap a coarse person probability to the image's edges with a guided filter."""
    filtered = cv2.ximgproc.guidedFilter(guide, probability.astype(np.float32), 8, 600.0)
    return smoothstep(0.3, 0.7, filtered)


def find_pose(bgr):
    """The pose finder's landmarks as {name: (x, y, visibility)} in pixels, or {} when it sees nobody."""
    import mediapipe as mp

    names = mp.solutions.pose.PoseLandmark
    height, width = bgr.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.4) as pose:
        found = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).pose_landmarks
    if not found:
        return {}
    marks = found.landmark
    return {name.lower(): (marks[names[name]].x * width, marks[names[name]].y * height, marks[names[name]].visibility)
            for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_WRIST", "RIGHT_WRIST")}


def pose_row(pose, names, height):
    """The mean row of two landmarks that are both seen inside the picture, else None."""
    marks = [pose.get(name) for name in names]
    if any(mark is None or mark[2] < POSE_VISIBLE or not 0 <= mark[1] < height for mark in marks):
        return None
    return float(np.mean([mark[1] for mark in marks]))


def geometry(face, size, pose):
    """Where the shoulders are and where the body is anchored (no motion at or below it), in pixels of the work image."""
    width, height = size
    x1, y1, x2, y2 = face
    face_height = y2 - y1
    shoulders = pose_row(pose, ("left_shoulder", "right_shoulder"), height)
    if shoulders is None or not y2 <= shoulders <= y2 + 2.0 * face_height:
        shoulders = y2 + SHOULDERS_BELOW * face_height
    hips = pose_row(pose, ("left_hip", "right_hip"), height)
    anchor = hips if hips is not None and hips > shoulders + 0.5 * face_height else float(height)
    return {"face_height": face_height, "shoulders": float(shoulders), "anchor": float(anchor),
            "centre": ((x1 + x2) / 2, (y1 + y2) / 2)}


def stand_in_matte(face, size, geo):
    """A person-shaped region from the face alone, for the photo whose real matte cannot be trusted."""
    width, height = size
    face_height = geo["face_height"]
    cx, cy = geo["centre"]
    matte = np.zeros((height, width), np.uint8)
    cv2.ellipse(matte, (int(cx), int(cy)), (int(0.75 * face_height), int(0.95 * face_height)), 0, 0, 360, 255, -1)
    top, bottom = int(geo["shoulders"]), int(min(height, geo["anchor"]))
    half_top, half_bottom = 1.25 * face_height, 1.05 * face_height
    shape = np.array([[cx - half_top, top], [cx + half_top, top], [cx + half_bottom, bottom], [cx - half_bottom, bottom]], np.int32)
    cv2.fillConvexPoly(matte, shape, 255)
    return cv2.GaussianBlur(matte, (0, 0), 0.04 * face_height).astype(np.float32) / 255.0


def person_matte(bgr, face, size, geo, segmenter=None):
    """(matte, trusted): the person as a 0..1 matte, and whether it came from the picture and not from the face's geometry."""
    try:
        matte = refine((segmenter or Segmenter())(bgr), bgr)
    except Exception:  # a failing segmenter must not cost the avatar its breathing
        matte = None
    if matte is not None:
        inside = matte > 0.5
        cx, cy = (int(round(v)) for v in geo["centre"])
        if MATTE_AREA[0] <= inside.mean() <= MATTE_AREA[1] and inside[min(cy, inside.shape[0] - 1), min(cx, inside.shape[1] - 1)]:
            return matte, True
    return stand_in_matte(face, size, geo), False


def field_weights(matte, face, geo):
    """The sway and breathing weight of every pixel, 0..1: the vertical profile times the closeness to the person's inside."""
    height, width = matte.shape
    face_height = geo["face_height"]
    rows = np.arange(height, dtype=np.float32)[:, None]
    span = max(geo["anchor"] - geo["shoulders"], 1.0)
    profile = 1.0 - smoothstep(0.0, 1.0, (rows - geo["shoulders"]) / span)
    ramp = max(EDGE_RAMP * face_height, 2.0)
    floor = EDGE_FLOOR * face_height
    inside = (matte > 0.5).astype(np.uint8)
    # The picture's border is not a silhouette: a person cut by the frame goes on beyond it.
    pad = int(np.ceil(ramp)) + 2
    padded = cv2.copyMakeBorder(inside, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    inner = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    outer = cv2.distanceTransform(1 - padded, cv2.DIST_L2, 5)
    signed = (inner - outer)[pad:-pad, pad:-pad]
    edge = smoothstep(0.0, 1.0, (signed + floor) / (ramp + floor))
    cx, cy = geo["centre"]
    columns = np.arange(width, dtype=np.float32)[None, :]
    reach = np.hypot((columns - cx) / (BLOB_RADII[0] * face_height), (rows - cy) / (BLOB_RADII[1] * face_height))
    blob = 1.0 - smoothstep(1.0, 1.0 + BLOB_FADE / BLOB_RADII[0], reach)
    return (profile * np.maximum(edge, blob * (matte > 0.2))).astype(np.float32)


# --- Hands -----------------------------------------------------------------------------------------------------------

# A hand's size (its bounding box's long side) against the face's height, and the detector's confidence.
HAND_SIZE = (0.3, 1.5)
HAND_CONFIDENCE = 0.5
# The farthest a wrist may be from the pose finder's wrist for the two to be the same hand, in face heights.
HAND_MATCH = 0.6
# Below this chroma distance (CIELAB a/b) between skin and what surrounds it, colour cannot tell them apart.
MIN_SEPARATION = 9.0


def find_hands(bgr, pose, face_height):
    """The hands MediaPipe sees as (21, 2) landmark arrays in pixels. Small hands, which the full picture hides from its
    detector, are searched for again in a window around the pose finder's wrists."""
    import mediapipe as mp

    height, width = bgr.shape[:2]
    found = []

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=HAND_CONFIDENCE) as hands:
        def detect(image, origin, scale):
            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rows, columns = image.shape[:2]
            return [np.array([[p.x * columns, p.y * rows] for p in marks.landmark], np.float32) / scale + origin
                    for marks in result.multi_hand_landmarks or []]

        found.extend(detect(bgr, np.zeros(2, np.float32), 1.0))
        for name in ("left_wrist", "right_wrist"):
            mark = pose.get(name)
            if mark is None or mark[2] < POSE_VISIBLE or not (0 <= mark[0] < width and 0 <= mark[1] < height):
                continue
            wrist = np.array(mark[:2], np.float32)
            if any(np.linalg.norm(hand[0] - wrist) < HAND_MATCH * face_height for hand in found):
                continue
            half = int(round(1.1 * face_height))
            left, top = max(0, int(wrist[0]) - half), max(0, int(wrist[1]) - half)
            right, bottom = min(width, int(wrist[0]) + half), min(height, int(wrist[1]) + half)
            window = bgr[top:bottom, left:right]
            if min(window.shape[:2]) < 32:
                continue
            scale = max(1.0, 320.0 / max(window.shape[:2]))
            if scale > 1.0:
                window = cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for points in detect(window, np.array([left, top], np.float32), scale):
                if np.linalg.norm(points[0] - wrist) < HAND_MATCH * face_height:
                    found.append(points)
    unique = []
    for points in sorted(found, key=lambda marks: -np.ptp(marks, axis=0).max()):
        if all(np.linalg.norm(points.mean(0) - kept.mean(0)) > 0.3 * face_height for kept in unique):
            unique.append(points)
    return unique


def otsu(values, bins=64):
    """The threshold that best splits a two-peaked set of numbers into two groups."""
    histogram, edges = np.histogram(values, bins=bins)
    histogram = histogram.astype(np.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    below = np.cumsum(histogram)
    above = below[-1] - below
    mean_below = np.cumsum(histogram * centres) / np.maximum(below, 1)
    mean_above = (np.cumsum((histogram * centres)[::-1]) / np.maximum(np.cumsum(histogram[::-1]), 1))[::-1]
    between = below[:-1] * above[:-1] * (mean_below[:-1] - mean_above[1:]) ** 2
    return float(centres[np.argmax(between)])


def skin_score(lab, cloth, skin):
    """Each pixel's chroma along the line from the surroundings' chroma (0) to the hand's (1)."""
    direction = skin - cloth
    separation = float(np.linalg.norm(direction))
    return ((lab[..., 1:] - cloth) @ (direction / separation)) / separation, separation


def hand_matte(bgr, points):
    """The hand's skin matte in a window around it, as a dict, or None when colour cannot separate it from its surroundings.

    The landmarks only seed the search: their hull misses the fingertips. The matte is every pixel whose chroma lies
    nearer the hand's than the surroundings', which is exact where the cloth is another colour and fails visibly (and
    detectably) where it is not."""
    height, width = bgr.shape[:2]
    low, high = points.min(0), points.max(0)
    size = float((high - low).max())
    pad = int(round(0.6 * size))
    left, top = max(0, int(low[0]) - pad), max(0, int(low[1]) - pad)
    right, bottom = min(width, int(high[0]) + pad + 1), min(height, int(high[1]) + pad + 1)
    window = bgr[top:bottom, left:right]
    lab = cv2.cvtColor(window, cv2.COLOR_BGR2LAB).astype(np.float32)
    local = points - [left, top]
    hull = np.zeros(window.shape[:2], np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(local).astype(np.int32)), 1)
    core = cv2.erode(hull, disc(0.2 * size)) > 0
    near = cv2.dilate(hull, disc(0.5 * size)) > 0
    outside = ~near
    if core.sum() < 50 or outside.sum() < 0.05 * outside.size:
        return None
    cloth, skin = np.median(lab[outside][:, 1:], axis=0), np.median(lab[core][:, 1:], axis=0)
    if np.linalg.norm(skin - cloth) < MIN_SEPARATION:
        return None
    score, separation = skin_score(lab, cloth, skin)
    threshold = 0.5 * otsu(np.clip(score[near], -0.5, 1.5))
    reach = cv2.dilate(hull, disc(0.6 * size)) > 0
    # Chroma decides only where it is clear. Between "surely skin" and "surely something else" the boundary is the strongest
    # edge in the picture, so skin-coloured surroundings (a desk, a painting's backdrop) cannot leak in without crossing one.
    markers = np.zeros(window.shape[:2], np.int32)
    markers[(score < threshold) | ~reach] = 1
    markers[core & (score > 2 * threshold)] = 2
    # The fingers' own skeleton is skin even where the fingers are as pale as the table they rest on.
    import mediapipe as mp

    bones = np.zeros(window.shape[:2], np.uint8)
    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(bones, tuple(np.round(local[a]).astype(int)), tuple(np.round(local[b]).astype(int)), 1,
                 max(2, int(round(0.03 * size))))
    markers[(bones > 0) & (score > 0.5 * threshold)] = 2
    smooth = cv2.GaussianBlur(window, (0, 0), max(1.0, 0.006 * size))
    cv2.watershed(smooth, markers)
    binary = (markers == 2).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disc(max(3.0, 0.015 * size)))
    count, labels = cv2.connectedComponents(binary)
    if count < 2:
        return None
    overlap = np.bincount(labels[core], minlength=count)
    overlap[0] = 0
    binary = (labels == overlap.argmax()).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disc(0.06 * size))
    # Rings and watches are inside the hand.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 1, -1)
    area = int(binary.sum())
    outline = cv2.convexHull(np.concatenate(contours)) if contours else None
    solidity = area / max(cv2.contourArea(outline), 1.0) if outline is not None else 0.0
    # How much of the landmark skeleton the matte covers, and how long the matte runs into the limit of the search.
    grown = cv2.dilate(binary, disc(0.06 * size)) > 0
    rows, columns = grown.shape
    covered = float(np.mean([grown[min(max(int(round(y)), 0), rows - 1), min(max(int(round(x)), 0), columns - 1)] for x, y in local]))
    limit = (cv2.dilate(binary, disc(5)) > 0) & ~(cv2.erode(reach.astype(np.uint8), disc(5)) > 0)
    return {"window": (left, top, right, bottom), "binary": binary, "score": score, "separation": separation,
            "threshold": float(threshold), "size": size, "area_ratio": area / max(int(hull.sum()), 1), "solidity": solidity,
            "covered": covered, "limit": limit, "cloth": cloth, "skin": skin, "local": local}


# --- A hand as a layer ---------------------------------------------------------------------------------------------------

# How far a hand may move: a fingertip travels at most REACH face heights, and it turns at most TURN_MAX radians (about 26
# degrees, as far as a wrist bends in a gesture). The whole hand slides at most SLIDE face heights and at most SLIDE_RAMP of
# its wrist's ramp: a slide is blended in along the ramp, which stretches the hand by about 1.5 times the slide over the
# ramp's length, and a hand that slid a third of its ramp was stretched 3 to 10 times more in one direction than in another
# (see distortion_of) in the corner poses, where it also turned. A turn is the clean part of a gesture: it shears the hand
# by about its own angle, whatever the ramp.
REACH, SLIDE = 0.28, 0.05
TURN_MAX = 0.45
SLIDE_RAMP = 0.1
# The shares of that range a hand may have, the largest first: a hand gets the first that what it uncovers can be hidden at.
AMOUNTS = (1.0, 0.75, 0.5, 0.35)
# A hand's turn builds up from the wrist over this share of the hand's length, so the wrist bends instead of tearing.
BAND = 0.5
# A hand's layer is its matte shaved by SHAVE and softened over EDGE, in hand sizes (at least 1 and 2 px): the matte's rim
# is a blend of skin and the cloth beside it, and a layer that carried it along would draw a pale line around the hand.
# Beside the matte, skin the colour test passed over is taken in up to SKIN_REACH hand sizes out.
SHAVE, EDGE, SKIN_REACH = 0.006, 0.01, 0.04
# The plate's cloth is read from a band that starts CLOTH_NEAR px outside the layer (past the shadow of the hand's edge)
# and ends CLOTH_FAR hand sizes from it.
CLOTH_NEAR, CLOTH_FAR = 3.0, 0.2
# The plate's grain is copied from cloth without structure of its own. A source is plain where the picture's gradient stays
# within PLAIN times what is usual around the hand (the looser tolerances are only tried for what the strict one left), and
# it is the picture less its blur of DETAIL_BLUR px that is copied. A copy has to cover MIN_GAIN of the layer to be worth
# making and there are at most COPIES of them; a copy comes from at least NEAREST and at most FARTHEST hand sizes away.
PLAIN, DETAIL_BLUR = (1.6, 2.5, 4.0), 8.0
MIN_GAIN, COPIES = 0.02, 8
NEAREST, FARTHEST = 0.12, 1.0
# Cloth whose grain is weaker than this (in levels) has none worth copying, and the band of cloth read for the plate must
# have at least MIN_CLOTH times the layer's area.
GRAIN_FLOOR, MIN_CLOTH = 1.0, 0.2
# The skin score from which a pixel beside the matte is not cloth.
NOT_CLOTH = 0.25
# What a hand must pass to be allowed to move (see check_hand).
LIMITS = {"covered": 0.9, "solidity": 0.8, "cut": 0.05, "leftover": 0.005, "ghost": 0.005, "halo": 14.0, "flat": 0.01,
          "distortion": 2.0}
# The metrics that fall with the share of the range the hand is given; a hand that fails any other cannot be helped.
SCALES = ("ghost", "flat", "distortion")
# A patch of plate without grain shows from this width (in hand sizes); narrower gaps are specks that the weave hides.
# (Real twill measures up to 0.006 here, strong fine stripes 0.013 and more.)
FLAT_WIDTH = 0.03
# The extreme poses the hand is judged at: multiples of its largest turn and slide, along each axis and in each corner.
POSES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
         *((turn, slide_x, slide_y) for turn in (-1, 1) for slide_x in (-1, 1) for slide_y in (-1, 1)))
# The poses a hand is drawn at to find the ground it can cover: these multiples of its largest turn, and of its largest
# slide along either axis.
SWEEP_TURNS, SWEEP_SLIDES = (-1.0, -0.5, 0.0, 0.5, 1.0), (-1.0, 0.0, 1.0)


def ramp_weight(shape, pivot, axis, ramp):
    """How much of the hand's motion applies at each pixel: 0 at and behind the wrist, 1 beyond the ramp."""
    rows, columns = shape
    along = ((np.arange(columns, dtype=np.float32)[None, :] + 0.5 - pivot[0]) * axis[0]
             + (np.arange(rows, dtype=np.float32)[:, None] + 0.5 - pivot[1]) * axis[1])
    return smoothstep(0.0, ramp, along)


def source_map(shape, rig, turn, slide):
    """Where each pixel of a hand's window takes its colour from when the hand has turned and slid: the rigid motion's
    inverse, blended in by the wrist's ramp. This is the player's math. Pixel centres are at integers, as cv2.remap wants."""
    rows, columns = shape
    x = np.arange(columns, dtype=np.float32)[None, :] + 0.5
    y = np.arange(rows, dtype=np.float32)[:, None] + 0.5
    weight = ramp_weight(shape, rig["pivot"], rig["axis"], rig["ramp"])
    cos, sin = np.cos(turn), np.sin(turn)
    rx, ry = x - rig["pivot"][0] - slide[0], y - rig["pivot"][1] - slide[1]
    source_x = x + weight * (rig["pivot"][0] + cos * rx + sin * ry - x)
    source_y = y + weight * (rig["pivot"][1] - sin * rx + cos * ry - y)
    return (source_x - 0.5).astype(np.float32), (source_y - 0.5).astype(np.float32)


def distortion_of(shape, rig, turn, slide, shown):
    """How unevenly a pose stretches the hand's picture: over the part of the hand that `shown` (the moved alpha) covers, the
    95th percentile of the ratio between the most and the least that the warp scales it, at a pixel, in some direction. A hand
    that turns and slides as a whole gives 1; the wrist's ramp, where the motion is blended in, is where it gives more, and
    where the blend folds over (a slide that is too large for the ramp) it gives a great deal."""
    map_x, map_y = source_map(shape, rig, turn, slide)
    (x_by_y, x_by_x), (y_by_y, y_by_x) = np.gradient(map_x), np.gradient(map_y)
    # The singular values of the warp's Jacobian: their squares sum to `square` and multiply to `determinant` squared.
    square = x_by_x ** 2 + x_by_y ** 2 + y_by_x ** 2 + y_by_y ** 2
    determinant = x_by_x * y_by_y - x_by_y * y_by_x
    root = np.sqrt(np.maximum(square ** 2 - 4.0 * determinant ** 2, 0.0))
    ratio = np.sqrt((square + root) / np.maximum(square - root, 1e-12))
    covered = shown > 0.5
    return float(np.percentile(ratio[covered], 95)) if covered.any() else 1.0


def hand_pose(photo, alpha, plate, rig, turn, slide):
    """(composite, moved alpha): the hand's window as the player draws it. The layer is the photo's own pixels, taken
    from where the hand has moved from, over the plate, which shows only where the hand has left."""
    map_x, map_y = source_map(alpha.shape, rig, turn, slide)
    layer = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    moved = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    uncovered = np.clip((alpha - moved) / np.maximum(alpha, 1e-3), 0.0, 1.0)[..., None]
    opaque = moved[..., None]
    return opaque * layer + (1.0 - opaque) * (photo + uncovered * (plate - photo)), moved


def layer_alpha(binary, score, size):
    """1 inside the matte shaved by SHAVE, falling smoothly to 0 over EDGE beyond that, in the 8-bit steps it is stored in.
    The matte first takes in the skin beside it that the colour test passed over (a finger's shaded underside): left in
    the picture, it would stay behind as a sliver when the hand moves."""
    shave, edge = max(1.0, SHAVE * size), max(2.0, EDGE * size)
    near = cv2.dilate(binary, disc(2.0 * SKIN_REACH * size)) > 0
    binary = (binary > 0) | (near & (score > NOT_CLOTH))
    binary = binary.astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    # The matte's edge is half a pixel from its outermost pixels' centres.
    distance = np.where(binary > 0, 0.5 - inside, outside - 0.5)
    alpha = cv2.GaussianBlur((1.0 - smoothstep(-shave, edge - shave, distance)).astype(np.float32), (0, 0), 0.8)
    return np.round(alpha * 255.0).astype(np.float32) / 255.0


def push_pull(image, weight):
    """`image` spread smoothly over the whole window from where `weight` is above 0: averaged down a pyramid, then back up,
    keeping what is known at each level."""
    floor = 1e-3
    levels = [(image * weight[..., None], weight)]
    while min(levels[-1][1].shape) > 6:
        levels.append((cv2.pyrDown(levels[-1][0]), cv2.pyrDown(levels[-1][1])))
    top, top_weight = levels[-1]
    result = np.where(top_weight[..., None] > floor, top / np.maximum(top_weight, floor)[..., None],
                      top.sum((0, 1)) / max(float(top_weight.sum()), floor))
    for fine, fine_weight in reversed(levels[:-1]):
        up = cv2.pyrUp(result, dstsize=(fine_weight.shape[1], fine_weight.shape[0]))
        have = np.clip(fine_weight * 4.0, 0.0, 1.0)[..., None]
        result = have * (fine / np.maximum(fine_weight, floor)[..., None]) + (1.0 - have) * up
    return result


def copy_grain(window, support, hole, others, score, size):
    """(grain, left): detail for the pixels of `support`, copied from cloth elsewhere in the window, and the pixels of it
    that no cloth could supply. Detail is the picture less its blur, so it adds texture and no tone, and every pixel
    gets it from one of a few constant offsets: each is the offset that supplies the most of what is still missing from
    plain cloth, so the copied pieces are real cloth in one piece, not stitched from scraps."""
    rows, columns = hole.shape
    picture = window.astype(np.float32)
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.5)
    energy = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    cloth = (hole == 0) & ~others & (score < 0.1)
    nearby = cloth & (cv2.dilate(hole, disc(FARTHEST * size)) > 0)
    grain = np.zeros_like(picture)
    left = support.copy()
    if nearby.sum() < 50 or not support.any():
        return grain, left
    usual = float(np.median(energy[nearby]))
    # The detail is taken from the cloth's own blur, which a nearby hand must not tint.
    known = cloth.astype(np.float32)
    blur = cv2.GaussianBlur(picture * known[..., None], (0, 0), DETAIL_BLUR) / np.maximum(
        cv2.GaussianBlur(known, (0, 0), DETAIL_BLUR), 1e-3)[..., None]
    detail = picture - blur
    plain = nearby & (energy <= PLAIN[0] * usual)
    if plain.sum() < 50 or float(detail.mean(2)[plain].std()) < GRAIN_FLOOR:
        return grain, np.zeros_like(support)
    pad = int(np.ceil(FARTHEST * size))
    padded_detail = np.pad(detail, ((pad, pad), (pad, pad), (0, 0)))
    ys, xs = np.nonzero(support)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    total = np.zeros_like(picture)
    weight = np.zeros((rows, columns), np.float32)
    area = float(support.sum())
    used = 0
    for tolerance in PLAIN:
        bad = (energy > tolerance * usual) | (hole > 0) | others | (score > 0.1)
        valid = cv2.dilate(bad.astype(np.uint8), disc(0.035 * size)) == 0
        padded_valid = np.pad(valid, pad, constant_values=False)
        while used < COPIES and left.any():
            # Every offset's gain at once: how many of the missing pixels have plain cloth that far away.
            wanted = left[y0:y1, x0:x1].astype(np.float32)
            gain = cv2.matchTemplate(padded_valid.astype(np.float32), wanted, cv2.TM_CCORR)
            dy = np.arange(gain.shape[0], dtype=np.float32)[:, None] - pad - y0
            dx = np.arange(gain.shape[1], dtype=np.float32)[None, :] - pad - x0
            far = dx * dx + dy * dy
            gain[(far < (NEAREST * size) ** 2) | (far > (FARTHEST * size) ** 2)] = 0
            best = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[best] < MIN_GAIN * area:
                break
            shift_y, shift_x = int(dy[best[0], 0]), int(dx[0, best[1]])
            window_y, window_x = pad + shift_y, pad + shift_x
            copied = padded_detail[window_y:window_y + rows, window_x:window_x + columns]
            takes = padded_valid[window_y:window_y + rows, window_x:window_x + columns] & left
            share = cv2.GaussianBlur(takes.astype(np.float32), (0, 0), 3.0)
            total += copied * share[..., None]
            weight += share
            left &= ~takes
            used += 1
    grain = total / np.maximum(weight, 1e-3)[..., None] * np.clip(weight, 0.0, 1.0)[..., None]
    return grain, left


def make_plate(window, alpha, size, score, others):
    """(plate, bare), or None when there is too little cloth around the hand to read. `window` is the hand's window of the
    picture, 8-bit BGR. The plate is what the hand covers:
    the cloth around it carried over it as tone, spread smoothly from a band of cloth around the layer, and as grain,
    copied from plain cloth in the window. `bare` marks the pixels that got no grain though the cloth has some. Whatever
    is not cloth beside the matte (a wrist cut off by the reach limit, the shaded edge of a finger) is in the hole too,
    so that the fill never borrows its colour. A weave's grain is carried over; a bold pattern is not (only plain cloth
    is copied), so the plate under a hand on plaid is plain."""
    support = alpha > 0.001
    hole = cv2.dilate((support | (score > NOT_CLOTH)).astype(np.uint8), disc(7))
    outside = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 5)
    reach = CLOTH_FAR * size
    band = (outside > CLOTH_NEAR) & (outside < reach) & (hole == 0) & ~others
    if band.sum() < MIN_CLOTH * support.sum():
        return None
    # Nearer cloth counts more.
    weight = (band * (1.0 - 0.6 * np.clip(outside / reach, 0.0, 1.0))).astype(np.float32)
    tone = push_pull(window.astype(np.float32), weight)
    grain, bare = copy_grain(window, support, hole, others, score, size)
    return np.clip(tone + grain, 0, 255).astype(np.float32), bare


def hand_rig(matte, face_height):
    """What moves a hand: the pivot at its wrist, the axis along it, the length of the wrist's ramp, and how far it may
    turn and slide. Positions are in the matte's window."""
    local = matte["local"]
    axis = local[[5, 9, 13, 17]].mean(0) - local[0]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
    rows, columns = np.nonzero(matte["binary"])
    points = np.stack([columns, rows], 1).astype(np.float32) + 0.5
    along = (points - local[0]) @ axis
    length = max(float(along.max() - along.min()), 1.0)
    pivot = points[along <= along.min() + 0.1 * length].mean(0)
    reach = max(float(np.linalg.norm(points - pivot, axis=1).max()), 1.0)
    return {"pivot": pivot, "axis": axis, "ramp": BAND * length,
            "turn": min(REACH * face_height / reach, TURN_MAX), "slide": min(SLIDE * face_height, SLIDE_RAMP * BAND * length)}


def lab_distance(first, second):
    """CIE76 colour difference between two float BGR images in 0..255."""
    a = cv2.cvtColor(np.clip(first, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(np.clip(second, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return np.linalg.norm(a - b, axis=2)


def check_hand(photo, alpha, plate, bare, matte, rig, others, amount):
    """Measures what a viewer would object to once the hand moves, by drawing it at its extreme poses.

    leftover: skin-coloured pixels beside the hand that its matte missed, which would stay behind as a ghost;
    ghost:    skin-coloured pixels where the hand was and no longer is, which a bad plate, a hole in the layer or a sliver
              just outside it leaves;
    halo:     how far the colours the layer carries beyond the skin (shadow, cloth) are from the plate under them;
    cut:      how much of the matte's hard edge, where it ran into the limit of the search (a wrist, a cuff), is in the part
              of the hand that moves: an edge that moves is a seam;
    flat:     the share of the hand's area that it uncovers where the plate has no grain, over a patch wider than FLAT_WIDTH
              hand sizes, though the cloth has grain;
    distortion: how unevenly the poses stretch the hand's picture (see distortion_of): a hand that is drawn twice as long
              at one place as at another looks like rubber."""
    size, cloth, skin = matte["size"], matte["cloth"], matte["skin"]
    support = (alpha > 0.01).astype(np.uint8)
    area = max(float(support.sum()), 1.0)
    moving = ramp_weight(alpha.shape, rig["pivot"], rig["axis"], rig["ramp"]) > 0.25
    cut = float((matte["limit"] & moving).sum()) / max(size, 1.0) / 5.0
    beside = (cv2.dilate(support, disc(0.5 * size)) > 0) & ~(cv2.dilate(support, disc(7)) > 0) & ~others & moving
    leftover = float(((matte["score"] > 0.6) & beside).sum()) / area
    watched = (alpha > 0.02) | ((cv2.dilate(support, disc(2.0 * SKIN_REACH * size)) > 0) & moving)
    patches = cv2.morphologyEx(bare.astype(np.uint8), cv2.MORPH_OPEN, disc(FLAT_WIDTH * size)) > 0
    ghost = flat = 0.0
    distortion = 1.0
    for turn, slide_x, slide_y in POSES:
        angle, slide = amount * turn * rig["turn"], (amount * slide_x * rig["slide"], amount * slide_y * rig["slide"])
        shown, moved = hand_pose(photo, alpha, plate, rig, angle, slide)
        lab = cv2.cvtColor(np.clip(shown, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        skinlike = skin_score(lab, cloth, skin)[0] > 0.6
        covered = cv2.dilate((moved > 0.5).astype(np.uint8), disc(7)) > 0
        ghost = max(ghost, float((skinlike & watched & ~others & ~covered).sum()) / area)
        uncovered = (alpha - moved) > 0.5 * alpha
        flat = max(flat, float((patches & uncovered & (alpha > 0.01)).sum()) / area)
        distortion = max(distortion, distortion_of(alpha.shape, rig, angle, slide, moved))
    carried = (alpha > 0.03) & (alpha < 0.97) & (matte["score"] < 0.15)
    halo = float(np.percentile(lab_distance(photo, plate)[carried], 90)) if carried.sum() > 20 else 0.0
    return {"covered": matte["covered"], "solidity": matte["solidity"], "cut": cut, "leftover": leftover,
            "ghost": ghost, "halo": halo, "flat": flat, "distortion": distortion}


def failures(metrics):
    """The metrics that are past their limit."""
    return [name for name, limit in LIMITS.items() if (metrics[name] < limit if name in ("covered", "solidity") else metrics[name] > limit)]


# --- The rig ---------------------------------------------------------------------------------------------------------

def overlap_mask(window, others):
    """A mask over `window` of the other hands' cut-outs, grown a little, so that their skin is not counted as a leftover."""
    left, top, right, bottom = window
    mask = np.zeros((bottom - top, right - left), np.uint8)
    for (other_left, other_top, other_right, other_bottom), binary in others:
        x0, y0, x1, y1 = max(left, other_left), max(top, other_top), min(right, other_right), min(bottom, other_bottom)
        if x1 > x0 and y1 > y0:
            mask[y0 - top:y1 - top, x0 - left:x1 - left] |= binary[y0 - other_top:y1 - other_top, x0 - other_left:x1 - other_left]
    return cv2.dilate(mask, disc(7)) > 0


def boxes_meet(first, second, margin=0.0):
    return (first[0] - margin < second[2] and second[0] - margin < first[2]
            and first[1] - margin < second[3] and second[1] - margin < first[3])


def swept_area(alpha, rig, window, margin):
    """(mask, rectangle): every place a hand's layer can be drawn, over the hand's window grown by `margin` px on each side
    (a hand can leave its window), and that grown window's rectangle in the work image."""
    padded = np.pad(alpha, margin)
    placed = dict(rig, pivot=rig["pivot"] + margin)
    area = np.zeros(padded.shape, bool)
    for turn in SWEEP_TURNS:
        for slide_x in SWEEP_SLIDES:
            for slide_y in SWEEP_SLIDES:
                map_x, map_y = source_map(padded.shape, placed, turn * rig["turn"],
                                          (slide_x * rig["slide"], slide_y * rig["slide"]))
                area |= cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.02
    left, top, right, bottom = window
    return area.astype(np.uint8), (left - margin, top - margin, right + margin, bottom + margin)


def reaches(sweep, other):
    """Whether a hand's swept area (as swept_area returns it) comes within a few pixels of the place of `other`, a hand
    candidate. The player draws its hands one over the other, and where a hand has left it draws the plate: it would
    paint over a hand that had moved into that place. (A hand that only passes in front of another is no harm.)"""
    mask, rectangle = sweep
    place = (other["matte"]["window"], (other["alpha"] > 0.001).astype(np.uint8))
    return bool((overlap_mask(rectangle, [place]) & (mask > 0)).any())


def build_hands(work, face_height, crop, pose):
    """(layers, report): the hands that may move, each with its rig and its alpha and plate over a box of the work image,
    and what became of every hand that was found."""
    candidates, report = [], []
    margin = int(np.ceil((REACH + SLIDE) * face_height)) + 2
    for number, points in enumerate(find_hands(work, pose, face_height)):
        entry = {"hand": number, "kept": False}
        report.append(entry)
        matte = hand_matte(work, points)
        if matte is None:
            entry["reason"] = "its colour cannot be told from its surroundings"
            continue
        left, top = matte["window"][:2]
        rows, columns = np.nonzero(matte["binary"])
        box = (left + columns.min(), top + rows.min(), left + columns.max() + 1, top + rows.max() + 1)
        entry["size"] = round(matte["size"] / face_height, 2)
        if not HAND_SIZE[0] <= entry["size"] <= HAND_SIZE[1]:
            entry["reason"] = "its size is not a hand's"
        elif boxes_meet(box, crop, (REACH + SLIDE) * face_height):
            entry["reason"] = "it is next to the face"
        elif matte["covered"] < LIMITS["covered"] or matte["solidity"] < LIMITS["solidity"]:
            entry["reason"] = "its outline is not a hand's"
        else:
            rig = hand_rig(matte, face_height)
            alpha = layer_alpha(matte["binary"], matte["score"], matte["size"])
            candidates.append({"matte": matte, "entry": entry, "area": int(matte["binary"].sum()), "rig": rig, "alpha": alpha,
                               "sweep": swept_area(alpha, rig, matte["window"], margin)})
    # Two hands, one of which could move into the other's place, would have to be cut apart; the larger one keeps its turn.
    chosen = []
    for candidate in sorted(candidates, key=lambda item: -item["area"]):
        if any(reaches(candidate["sweep"], kept) or reaches(kept["sweep"], candidate) for kept in chosen):
            candidate["entry"]["reason"] = "it could move into another hand's place"
        else:
            chosen.append(candidate)
    layers = []
    for candidate in sorted(chosen, key=lambda item: item["entry"]["hand"]):
        matte, entry, rig, alpha = candidate["matte"], candidate["entry"], candidate["rig"], candidate["alpha"]
        left, top, right, bottom = matte["window"]
        window = work[top:bottom, left:right]
        others = overlap_mask(matte["window"], [(other["matte"]["window"], other["matte"]["binary"])
                                                for other in chosen if other is not candidate])
        made = make_plate(window, alpha, matte["size"], matte["score"], others)
        if made is None:
            entry["reason"] = "there is too little cloth around it to carry over it"
            continue
        plate, bare = made
        # The largest range first; a smaller one only where what fails is what a smaller range uncovers less of.
        for amount in AMOUNTS:
            metrics = check_hand(window.astype(np.float32), alpha, plate, bare, matte, rig, others, amount)
            bad = failures(metrics)
            if not bad or any(name not in SCALES for name in bad):
                break
        entry["metrics"] = {name: round(float(value), 4) for name, value in metrics.items()}
        if bad:
            entry["reason"] = "it would show " + ", ".join(bad)
            continue
        rows, columns = np.nonzero(alpha > 0.002)
        x0, y0 = max(columns.min() - 2, 0), max(rows.min() - 2, 0)
        x1, y1 = min(columns.max() + 3, alpha.shape[1]), min(rows.max() + 3, alpha.shape[0])
        layer_alpha_crop = alpha[y0:y1, x0:x1].copy()
        # The texture is clamped at its border, so the border must be empty.
        layer_alpha_crop[0, :] = layer_alpha_crop[-1, :] = layer_alpha_crop[:, 0] = layer_alpha_crop[:, -1] = 0.0
        origin = np.array([left, top], np.float32)
        layers.append({"box": (left + x0, top + y0, left + x1, top + y1), "alpha": layer_alpha_crop, "plate": plate[y0:y1, x0:x1],
                       "pivot": rig["pivot"] + origin, "axis": rig["axis"], "ramp": rig["ramp"],
                       "turn": rig["turn"] * amount, "slide": rig["slide"] * amount})
        entry["kept"], entry["amount"] = True, amount
    return layers, report


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write {path}")


def clean(folder):
    for name in ("body.json", "body-field.png", *(path.name for path in folder.glob("hand-*.png")),
                 *(path.name for path in folder.glob("plate-*.png"))):
        (folder / name).unlink(missing_ok=True)


def build(folder):
    """Writes the body rig into an avatar's folder and returns it: body-field.png, body.json and, for every hand that
    may move, hand-K.png (its alpha) and plate-K.png. It reads photo.png and rig.json, and never fails because of what is in
    the photo: whatever cannot be analysed is left still, and in the worst case the rig is empty, which the player treats
    as no body motion."""
    folder = Path(folder)
    meta = json.loads((folder / "rig.json").read_text())
    width, height = meta["size"]
    face_height_photo = meta["face"][3] - meta["face"][1]
    rig = {"version": 1, "size": [width, height], "field": None, "sway": 0.0, "lift": 0.0, "hands": []}
    report = {"person": None, "pose": False, "hands": []}
    clean(folder)
    try:
        photo = cv2.imread(str(folder / "photo.png"), cv2.IMREAD_COLOR)
        if photo is None or photo.shape[:2] != (height, width):
            raise ValueError("photo.png is missing or does not match rig.json")
        scale = min(1.0, WORK_SIDE / max(width, height))
        work = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else photo
        size = (work.shape[1], work.shape[0])
        face = [value * scale for value in meta["face"]]
        crop = [value * scale for value in meta["crop"]]
        try:
            pose = find_pose(work)
        except Exception:
            pose = {}
        report["pose"] = bool(pose)
        geo = geometry(face, size, pose)
        matte, trusted = person_matte(work, face, size, geo)
        report["person"] = "photo" if trusted else "stand-in"
        field = field_weights(matte, face, geo)
        factor = FIELD_SIDE / max(field.shape)
        small = cv2.resize(field, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        write_image(folder / "body-field.png", np.clip(small * 255.0 + 0.5, 0, 255).astype(np.uint8))
        amplitude = 1.0 if trusted else 0.5
        rig.update(field="body-field.png", sway=SWAY * face_height_photo * amplitude, lift=LIFT * face_height_photo * amplitude)
        try:
            layers, report["hands"] = build_hands(work, geo["face_height"], crop, pose)
        except Exception as error:
            layers, report["hands"] = [], [{"error": f"{type(error).__name__}: {error}"}]
        for number, layer in enumerate(layers):
            write_image(folder / f"hand-{number}.png", np.clip(layer["alpha"] * 255.0 + 0.5, 0, 255).astype(np.uint8))
            write_image(folder / f"plate-{number}.png", np.clip(layer["plate"] + 0.5, 0, 255).astype(np.uint8))
            rig["hands"].append({
                "box": [round(float(value) / scale, 2) for value in layer["box"]],
                "pivot": [round(float(value) / scale, 2) for value in layer["pivot"]],
                "axis": [round(float(value), 5) for value in layer["axis"]],
                "ramp": round(float(layer["ramp"]) / scale, 2),
                "turn": round(float(layer["turn"]), 5),
                "slide": round(float(layer["slide"]) / scale, 2),
                "alpha": f"hand-{number}.png",
                "plate": f"plate-{number}.png"})
    except Exception as error:
        clean(folder)
        rig.update(field=None, sway=0.0, lift=0.0, hands=[])
        report["error"] = f"{type(error).__name__}: {error}"
    rig["report"] = report
    pending = folder / "body.json.tmp"
    pending.write_text(json.dumps(rig, indent=1))
    pending.replace(folder / "body.json")
    return rig


def main(argv):
    if len(argv) != 2:
        print("usage: python -m photostage.body <avatar folder>", file=sys.stderr)
        return 2
    # MediaPipe's solutions open an EGL context on the first GPU the driver offers, which on a shared machine may be
    # someone else's, although their models run on the CPU. With no EGL vendor there is no context and no GPU at all.
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
    rig = build(argv[1])
    report = rig["report"]
    print(f"body rig: person from {report['person']}, {len(rig['hands'])} of {len(report['hands'])} hands move"
          + (f", error: {report['error']}" if "error" in report else ""))
    for entry in report["hands"]:
        print("  hand", entry.get("hand"), "moves" if entry.get("kept") else "stays still: " + entry.get("reason", entry.get("error", "?")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
