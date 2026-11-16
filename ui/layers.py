from __future__ import annotations

import numpy as np

class Layer:
    def __init__(self, name: str, rgb: np.ndarray, alpha: np.ndarray,
                 visible: bool = True, opacity: float = 1.0, original: np.ndarray | None = None):
        self.name = name
        self.rgb = rgb
        self.alpha = alpha
        self.visible = visible
        self.opacity = opacity
        # What the eraser brings back: the opened (or new) image for the Background,
        # so erasing removes only what was drawn on it. None: the eraser makes the
        # layer transparent (empty layers, pasted pieces, text...).
        self.original = original

    def copy(self) -> "Layer":
        return Layer(self.name, self.rgb, self.alpha, self.visible, self.opacity, self.original)

    def is_opaque(self) -> bool:
        """Return True if all pixels are fully opaque (alpha = 255)"""
        return bool(self.alpha.min() == 255)


class LayerStack:
    def __init__(self, layers: list[Layer], active: int = 0, next_number: int = 1):
        if not layers:
            raise ValueError("A layer stack needs at least one layer")
        self.layers = layers
        self.active = active
        self._next_number = next_number

        # Store the combined layers below and above the active layer for reuse
        self._below_cache = (None, None)
        self._above_cache = (None, None)
        self._full_cache = (None, None)

    # Creation
    @classmethod
    def from_image(cls, image: np.ndarray) -> "LayerStack":
        """Create a new document with one fully opaque "Background" layer containing the image."""
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return cls([Layer("Background", image, alpha, original=image)])

    def snapshot(self) -> "LayerStack":
        """Create a copy for the undo/redo history"""
        return LayerStack([layer.copy() for layer in self.layers], self.active, self._next_number)


    # Info
    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active]

    @property
    def size(self) -> tuple[int, int]:
        return self.layers[0].rgb.shape[:2]

    def __len__(self) -> int:
        return len(self.layers)


    # Add / delete
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

    # Composite
    def composite(self, override: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        layer = self.active_layer
        rgb, alpha = override if override is not None else (layer.rgb, layer.alpha)

        if (len(self.layers) == 1 and layer.visible and layer.opacity >= 1.0
                and (alpha is layer.alpha and layer.is_opaque() or alpha.min() == 255)):
            return rgb

        full = self._full()
        if override is None:
            return full

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

    # Edits
    def edit_active(self, operation, args=(), erase: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        return self._blend(region, rgb, alpha)

    def apply_to_all(self, operation, args=()) -> None:
        new_layers = []
        for layer in self.layers:
            rgb = operation(layer.rgb, *args)
            alpha3 = np.dstack([layer.alpha] * 3)       # core functions expect 3 channels
            alpha = np.ascontiguousarray(operation(alpha3, *args)[:, :, 0])
            new_layer = layer.copy()
            new_layer.rgb, new_layer.alpha = rgb, alpha
            if layer.original is not None:       # rotate / resize / crop it the same way
                new_layer.original = operation(layer.original, *args)
            new_layers.append(new_layer)
        self.layers = new_layers


def overlay_from_drawing(draw, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    on_black = draw(np.zeros((height, width, 3), dtype=np.uint8))
    on_white = draw(np.full((height, width, 3), 255, dtype=np.uint8))
    painted = on_black[:, :, 0] == on_white[:, :, 0]
    return on_black, np.where(painted, 255, 0).astype(np.uint8)
