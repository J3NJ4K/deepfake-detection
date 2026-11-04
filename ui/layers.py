"""
Layer system for the editor (no Qt in here, only NumPy, so it can be tested alone).

Each layer has:
    rgb     -> H x W x 3 uint8 (same format as everywhere else in the project)
    alpha   -> H x W uint8, 0 = transparent, 255 = opaque
    visible -> bool
    opacity -> float between 0 and 1

All layers always have the same size. Layer 0 is the bottom one ("Background").
The image shown on screen (and saved, and analysed) is the composite of all
visible layers, drawn from bottom to top over a white background.

Like the rest of the code, arrays are never modified in place: an edit replaces
layer.rgb / layer.alpha with new arrays. This is what makes snapshots for
undo/redo cheap (they share the arrays and only copy the small Layer objects).
"""

from __future__ import annotations

import numpy as np


class Layer:
    def __init__(self, name: str, rgb: np.ndarray, alpha: np.ndarray,
                 visible: bool = True, opacity: float = 1.0):
        self.name = name
        self.rgb = rgb
        self.alpha = alpha
        self.visible = visible
        self.opacity = opacity

    def copy(self) -> "Layer":
        # Shallow copy on purpose: arrays are shared (they are never modified in place)
        return Layer(self.name, self.rgb, self.alpha, self.visible, self.opacity)

    def is_opaque(self) -> bool:
        """True if every pixel is fully opaque (alpha = 255)."""
        return bool(self.alpha.min() == 255)


class LayerStack:
    def __init__(self, layers: list[Layer], active: int = 0, next_number: int = 1):
        if not layers:
            raise ValueError("A layer stack needs at least one layer")
        self.layers = layers          # index 0 = bottom
        self.active = active          # index of the layer the tools draw on
        self._next_number = next_number
        # Cached merges of the layers below / above the active one (see composite())
        self._below_cache = (None, None)
        self._above_cache = (None, None)
        self._full_cache = (None, None)

    # ------------------------------------------------------------------
    # Creation / copies
    # ------------------------------------------------------------------
    @classmethod
    def from_image(cls, image: np.ndarray) -> "LayerStack":
        """New document: a single opaque "Background" layer containing the image."""
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return cls([Layer("Background", image, alpha)])

    def snapshot(self) -> "LayerStack":
        """Copy used by the undo/redo history."""
        return LayerStack([layer.copy() for layer in self.layers], self.active, self._next_number)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active]

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) of every layer."""
        return self.layers[0].rgb.shape[:2]

    def __len__(self) -> int:
        return len(self.layers)

    # ------------------------------------------------------------------
    # Add / delete
    # ------------------------------------------------------------------
    def add_layer(self, name: str | None = None) -> Layer:
        """Add an empty, fully transparent layer just above the active one and select it."""
        height, width = self.size
        if name is None:
            name = f"Layer {self._next_number}"
            self._next_number += 1
        layer = Layer(name,
                      np.zeros((height, width, 3), dtype=np.uint8),
                      np.zeros((height, width), dtype=np.uint8))
        self.active += 1
        self.layers.insert(self.active, layer)
        return layer

    def add_layer_on_top(self, name: str, rgb: np.ndarray, alpha: np.ndarray) -> Layer:
        """Add a layer with given content at the very top, without changing the active layer."""
        if rgb.shape[:2] != self.size or alpha.shape != self.size:
            raise ValueError("The new layer must have the same size as the image")
        layer = Layer(name, rgb, alpha)
        self.layers.append(layer)
        return layer

    def delete_layer(self, index: int) -> None:
        if len(self.layers) == 1:
            raise ValueError("Cannot delete the last layer")
        del self.layers[index]
        if self.active >= index and self.active > 0:
            self.active -= 1

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------
    def composite(self, override: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """
        Merge the visible layers into one RGB uint8 image (over a white background).
        `override` = (rgb, alpha) temporarily replaces the active layer, used for live previews.

        To stay fast while drawing, the layers below and above the active layer are
        merged once and kept in a cache: as long as they do not change, each mouse move
        only has to blend the active layer between these two cached images.
        """
        layer = self.active_layer
        rgb, alpha = override if override is not None else (layer.rgb, layer.alpha)

        # Fast path: one opaque layer -> no computation at all (same speed as without layers)
        if (len(self.layers) == 1 and layer.visible and layer.opacity >= 1.0
                and (alpha is layer.alpha and layer.is_opaque() or alpha.min() == 255)):
            return rgb

        full = self._full()
        if override is None:
            return full

        # Preview: only the rectangle that the drawing changed has to be recomputed
        changed = (alpha != layer.alpha) | (rgb[:, :, 0] != layer.rgb[:, :, 0]) \
            | (rgb[:, :, 1] != layer.rgb[:, :, 1]) | (rgb[:, :, 2] != layer.rgb[:, :, 2])
        rows = np.flatnonzero(changed.any(axis=1))
        if rows.size == 0:
            return full
        cols = np.flatnonzero(changed.any(axis=0))
        region = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        out = full.copy()
        out[region] = self._blend(region, rgb[region], alpha[region])
        return out

    def _full(self) -> np.ndarray:
        """Merge of all layers in their current state (cached until something changes)."""
        state = self._state(self.layers) + [self.active]
        cached_state, cached = self._full_cache
        if cached_state is not None and cached_state[-1] == self.active \
                and self._same_state(cached_state[:-1], state[:-1]):
            return cached
        layer = self.active_layer
        everything = (slice(None), slice(None))
        result = self._blend(everything, layer.rgb, layer.alpha)
        self._full_cache = (state, result)
        return result

    def remember_composite(self, image: np.ndarray) -> None:
        """
        Tell the stack that `image` is the merge of the layers in their current state
        (the brush already computed it while drawing), so it is not recomputed.
        """
        self._full_cache = (self._state(self.layers) + [self.active], image)

    def _blend(self, region, rgb, alpha) -> np.ndarray:
        """Merge (below) + active layer (rgb, alpha) + (above) inside `region`."""
        layer = self.active_layer
        if self.active == 0:
            # Nothing below the bottom layer: just the white background
            out = np.full(rgb.shape, 255.0, dtype=np.float32)
        else:
            out = self._below()[region]                        # float32
        if layer.visible and layer.opacity > 0:
            a = alpha.astype(np.float32)[:, :, None] * (layer.opacity / 255.0)
            out = out + (rgb.astype(np.float32) - out) * a     # = out*(1-a) + rgb*a
        above = self._above()
        if above is not None:
            color, coverage = above
            out = out * (1.0 - coverage[region]) + color[region]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def _state(layers) -> list:
        # Keeps references to the arrays themselves, so the comparison with `is` is reliable
        return [(l.rgb, l.alpha, l.visible, l.opacity) for l in layers]

    @staticmethod
    def _same_state(a, b) -> bool:
        return (a is not None and len(a) == len(b)
                and all(x[0] is y[0] and x[1] is y[1] and x[2] == y[2] and x[3] == y[3]
                        for x, y in zip(a, b)))

    def _below(self) -> np.ndarray:
        """Merge of the layers under the active one, over white (cached)."""
        state = self._state(self.layers[:self.active])
        cached_state, cached = self._below_cache
        if self._same_state(cached_state, state):
            return cached
        height, width = self.size
        out = np.full((height, width, 3), 255.0, dtype=np.float32)
        for layer in self.layers[:self.active]:
            if layer.visible and layer.opacity > 0:
                a = layer.alpha.astype(np.float32)[:, :, None] * (layer.opacity / 255.0)
                out = out + (layer.rgb.astype(np.float32) - out) * a
        self._below_cache = (state, out)
        return out

    def _above(self):
        """
        Layers above the active one merged into (color, coverage), cached, so that
        final = image * (1 - coverage) + color. None if there is nothing visible above.
        """
        layers = [l for l in self.layers[self.active + 1:] if l.visible and l.opacity > 0]
        if not layers:
            return None
        state = self._state(self.layers[self.active + 1:])
        cached_state, cached = self._above_cache
        if self._same_state(cached_state, state):
            return cached
        height, width = self.size
        color = np.zeros((height, width, 3), dtype=np.float32)
        coverage = np.zeros((height, width, 1), dtype=np.float32)
        for layer in layers:
            a = layer.alpha.astype(np.float32)[:, :, None] * (layer.opacity / 255.0)
            color = color * (1.0 - a) + layer.rgb.astype(np.float32) * a
            coverage = coverage * (1.0 - a) + a
        self._above_cache = (state, (color, coverage))
        return color, coverage

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------
    def edit_active(self, operation, args=(), erase: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (new_rgb, new_alpha) of the active layer after a drawing operation,
        WITHOUT changing the stack (so it can also be used for previews).

        operation(image, *args) is any function of the project that takes and returns
        an RGB image (brush stroke, shape, flood fill...).

        To know which pixels the operation painted, it is run a second time on the
        inverted image: a painted pixel gets the same colour in both results, while an
        untouched pixel keeps its (different) original value. Painted pixels become
        opaque, or transparent if `erase` is True (eraser tool).
        """
        layer = self.active_layer
        result = operation(layer.rgb, *args)

        # Fast path: drawing on a fully opaque layer -> nothing to change in alpha
        if not erase and layer.is_opaque():
            return result, layer.alpha

        probe = operation(255 - layer.rgb, *args)
        # An untouched pixel differs from its inverse in every channel, so one channel is enough
        painted = result[:, :, 0] == probe[:, :, 0]
        if erase:
            return layer.rgb, np.where(painted, 0, layer.alpha).astype(np.uint8)
        return result, np.where(painted, 255, layer.alpha).astype(np.uint8)

    def edit_region(self, operation, region, erase: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Same as edit_active, but only on a rectangle of the active layer
        (region = (row slice, column slice)); operation receives the cropped image,
        so its coordinates must be relative to the top-left corner of the region.
        Much faster for previews, since a shape only covers a small part of the image.
        """
        layer = self.active_layer
        rgb = np.ascontiguousarray(layer.rgb[region])
        alpha = layer.alpha[region]
        result = operation(rgb)
        if not erase and alpha.min() == 255:
            return result, alpha
        probe = operation(255 - rgb)
        painted = result[:, :, 0] == probe[:, :, 0]
        if erase:
            return rgb, np.where(painted, 0, alpha).astype(np.uint8)
        return result, np.where(painted, 255, alpha).astype(np.uint8)

    def blend_region(self, region, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        What the screen shows inside `region` if the active layer had (rgb, alpha) there
        (rgb and alpha are already cropped to the region). Used by the live previews.
        """
        return self._blend(region, rgb, alpha)

    def apply_to_all(self, operation, args=()) -> None:
        """
        Apply a geometric operation (rotate, flip, resize, crop...) to every layer,
        so that all layers keep the same size and stay aligned.
        """
        new_layers = []
        for layer in self.layers:
            rgb = operation(layer.rgb, *args)
            alpha3 = np.dstack([layer.alpha] * 3)       # core functions expect 3 channels
            alpha = np.ascontiguousarray(operation(alpha3, *args)[:, :, 0])
            new_layer = layer.copy()
            new_layer.rgb, new_layer.alpha = rgb, alpha
            new_layers.append(new_layer)
        self.layers = new_layers


def overlay_from_drawing(draw, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a transparent layer containing only what `draw(image) -> image` draws
    (used to put detection boxes on their own layer).
    """
    on_black = draw(np.zeros((height, width, 3), dtype=np.uint8))
    on_white = draw(np.full((height, width, 3), 255, dtype=np.uint8))
    painted = on_black[:, :, 0] == on_white[:, :, 0]
    return on_black, np.where(painted, 255, 0).astype(np.uint8)
