import math

import cv2
import numpy as np

MIN_SIZE = 4
SNAP_ANGLE = 15

HANDLES = {
    "nw": (-1, -1), "n": (0, -1), "ne": (1, -1), "e": (1, 0),
    "se": (1, 1), "s": (0, 1), "sw": (-1, 1), "w": (-1, 0),
}
ROTATE = "rotate"
MOVE = "move"


class Placement:
    def __init__(self, rgb: np.ndarray, alpha: np.ndarray, center, size=None):
        self.rgb = rgb
        self.alpha = alpha
        height, width = alpha.shape
        self.source_size = (width, height)
        self.cx, self.cy = center
        self.width, self.height = size if size is not None else (width, height)
        self.angle = 0.0
        self._drag = None

    # Geometry
    def _cos_sin(self):
        a = math.radians(self.angle)
        return math.cos(a), math.sin(a)

    def to_image(self, lx: float, ly: float, origin=None):
        ox, oy = origin if origin is not None else (self.cx, self.cy)
        c, s = self._cos_sin()
        return ox + lx * c - ly * s, oy + lx * s + ly * c

    def to_local(self, x: float, y: float, origin=None):
        ox, oy = origin if origin is not None else (self.cx, self.cy)
        c, s = self._cos_sin()
        dx, dy = x - ox, y - oy
        return dx * c + dy * s, -dx * s + dy * c

    def corners(self):
        w, h = self.width / 2, self.height / 2
        return [self.to_image(-w, -h), self.to_image(w, -h), self.to_image(w, h), self.to_image(-w, h)]

    def handle_points(self, rotate_distance: float) -> dict:
        points = {name: self.to_image(sx * self.width / 2, sy * self.height / 2)
                  for name, (sx, sy) in HANDLES.items()}
        points[ROTATE] = self.to_image(0, -self.height / 2 - rotate_distance)
        return points

    def hit_test(self, x: float, y: float, tolerance: float, rotate_distance: float):
        points = self.handle_points(rotate_distance)
        for name in [ROTATE, "nw", "ne", "se", "sw", "n", "e", "s", "w"]:
            hx, hy = points[name]
            if math.hypot(x - hx, y - hy) <= tolerance:
                return name
        lx, ly = self.to_local(x, y)
        if abs(lx) <= self.width / 2 and abs(ly) <= self.height / 2:
            return MOVE
        return None

    def handle_direction(self, name: str) -> float:
        sx, sy = HANDLES[name]
        return math.degrees(math.atan2(sy, sx)) + self.angle

    def transform_matrix(self) -> np.ndarray:
        src_w, src_h = self.source_size
        return self._matrix(self.width / src_w, self.height / src_h, src_w, src_h)

    def _matrix(self, scale_x, scale_y, src_w, src_h) -> np.ndarray:
        c, s = self._cos_sin()
        a = np.array([[c * scale_x, -s * scale_y], [s * scale_x, c * scale_y]])
        offset = np.array([self.cx, self.cy]) - a @ np.array([src_w / 2, src_h / 2])
        return np.hstack([a, offset[:, None]])


    # Mouse
    def start_drag(self, kind: str, x: float, y: float) -> None:
        start = dict(kind=kind, mouse=(x, y), center=(self.cx, self.cy),
                     size=(self.width, self.height), angle=self.angle)
        if kind in HANDLES:
            sx, sy = HANDLES[kind]
            start["anchor"] = self.to_image(-sx * self.width / 2, -sy * self.height / 2)
        self._drag = start

    def drag_to(self, x: float, y: float, shift: bool = False) -> None:
        drag = self._drag
        if drag is None:
            return
        kind = drag["kind"]
        if kind == MOVE:
            (mx, my), (cx, cy) = drag["mouse"], drag["center"]
            self.cx, self.cy = cx + x - mx, cy + y - my
        elif kind == ROTATE:
            cx, cy = self.cx, self.cy
            mx, my = drag["mouse"]
            turn = math.degrees(math.atan2(y - cy, x - cx) - math.atan2(my - cy, mx - cx))
            angle = drag["angle"] + turn
            if shift:
                angle = round(angle / SNAP_ANGLE) * SNAP_ANGLE
            self.angle = (angle + 180) % 360 - 180
        else:
            self._resize(drag, x, y, free=shift)

    def _resize(self, drag, x, y, free: bool) -> None:
        sx, sy = HANDLES[drag["kind"]]
        start_w, start_h = drag["size"]
        anchor = drag["anchor"]
        vx, vy = self.to_local(x, y, origin=anchor)
        if sx and sy:
            if free:
                width, height = vx * sx, vy * sy
            else:
                k = (vx * sx * start_w + vy * sy * start_h) / (start_w ** 2 + start_h ** 2)
                width, height = start_w * k, start_h * k
                smallest = min(width, height)
                if smallest < MIN_SIZE:
                    k = MIN_SIZE / min(start_w, start_h)
                    width, height = start_w * k, start_h * k
        elif sx:
            width, height = vx * sx, start_h
        else:
            width, height = start_w, vy * sy
        self.width, self.height = max(MIN_SIZE, width), max(MIN_SIZE, height)
        self.cx, self.cy = self.to_image(sx * self.width / 2, sy * self.height / 2, origin=anchor)

    def end_drag(self) -> None:
        self._drag = None

    def is_dragging(self) -> bool:
        return self._drag is not None


    # Final pixels
    def apply_to(self, layer_rgb: np.ndarray, layer_alpha: np.ndarray):
        height, width = layer_alpha.shape
        xs, ys = zip(*self.corners())
        left, top = max(0, math.floor(min(xs))), max(0, math.floor(min(ys)))
        right, bottom = min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys)))
        rgb, alpha = layer_rgb.copy(), layer_alpha.copy()
        if left >= right or top >= bottom:
            return rgb, alpha

        # For large size reductions, use INTER_AREA to shrink the image first for a cleaner result
        src_rgb, src_alpha = self.rgb, self.alpha
        src_w, src_h = self.source_size
        new_w, new_h = max(1, round(self.width)), max(1, round(self.height))
        if new_w < src_w and new_h < src_h:
            src_rgb = cv2.resize(src_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            src_alpha = cv2.resize(src_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
            src_w, src_h = new_w, new_h
        matrix = self._matrix(self.width / src_w, self.height / src_h, src_w, src_h)
        matrix[:, 2] += matrix[:, :2] @ np.array([0.5, 0.5]) - 0.5 - np.array([left, top])

        a = src_alpha.astype(np.float32) / 255
        premultiplied = src_rgb.astype(np.float32) * a[:, :, None]
        size = (right - left, bottom - top)
        warp = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        piece = cv2.warpAffine(premultiplied, matrix, size, **warp)
        piece_a = cv2.warpAffine(a, matrix, size, **warp)
        if piece.ndim == 2:
            piece = piece[:, :, None]

        below_rgb = rgb[top:bottom, left:right].astype(np.float32)
        below_a = alpha[top:bottom, left:right].astype(np.float32) / 255
        out_a = piece_a + below_a * (1 - piece_a)
        out_rgb = piece + below_rgb * (below_a * (1 - piece_a))[:, :, None]
        safe = np.where(out_a > 0, out_a, 1)[:, :, None]
        out_rgb = np.where(out_a[:, :, None] > 0, out_rgb / safe, below_rgb)

        rgb[top:bottom, left:right] = np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8)
        alpha[top:bottom, left:right] = np.clip(np.rint(out_a * 255), 0, 255).astype(np.uint8)
        return rgb, alpha
