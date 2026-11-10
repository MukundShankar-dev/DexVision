"""Appearance variation for the policy cameras (docs/PLAN.md, A; ranges in configs/appearance.yaml).

What varies per episode: the lights, the colour and texture of the table, wall, trays and floor,
and the colour, texture and shine of each object and the button. It is drawn once per episode
seed, from its own random streams, and applied to the rendered scene (``mjvScene``) just before
each frame is drawn. The MuJoCo model is not changed after it is built, and physics never reads
the scene, so an episode's physics is identical whatever it looks like.

The model carries a bank of procedural greyscale textures (``add_texture_bank``, at load) that
surfaces and objects may take, tinted by their drawn colour. The last variant of each texture
kind is held out of train seeds.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from dexvision import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "appearance.yaml"
FACTORS = ("lighting", "surfaces", "objects")
STREAM = 2  # episode random streams: task 1, demo noise 3, appearance 2 (one sub-stream per factor)
OBJECT_GEOMS = ("block_small", "block_large", "cylinder_short", "cylinder_tall", "puck_light",
                "puck_heavy", "button")
SURFACES = ("table", "trays", "floor")


@dataclass(frozen=True)
class AppearanceConfig:
    raw: Mapping

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> AppearanceConfig:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def bank(self) -> Mapping:
        return self.raw["texture_bank"]

    @property
    def bank_size(self) -> int:
        return len(self.bank["kinds"]) * int(self.bank["variants"])

    def held_out(self, texture: int) -> bool:
        """Bank texture ``texture`` is the last variant of its kind: never shown in train seeds."""
        variants = int(self.bank["variants"])
        return texture >= 0 and texture % variants == variants - 1


# ---------------------------------------------------------------------------- texture bank

def _value_noise(rng: np.random.Generator, size: int, cell: int) -> np.ndarray:
    """[size, size] in [0, 1]: a random grid every ``cell`` px, bilinearly interpolated, tiling."""
    n = size // cell
    grid = rng.random((n, n))
    x = (np.arange(size) + 0.5) / cell - 0.5
    i0 = np.floor(x).astype(int) % n
    i1 = (i0 + 1) % n
    t = x - np.floor(x)
    rows = grid[i0] * (1 - t)[:, None] + grid[i1] * t[:, None]
    return rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]


def _texture(kind: str, variant: int, rng: np.random.Generator, size: int) -> np.ndarray:
    """[size, size] pattern in [0, 1]. Variants differ in scale. Every pattern tiles."""
    scale = (8, 16, 32, 64)[variant % 4]
    if kind == "noise":
        return 0.65 * _value_noise(rng, size, scale) + 0.35 * _value_noise(rng, size, scale // 4)
    if kind == "grain":  # wood-like: stripes bent by smooth noise
        rows = np.arange(size)[:, None] / size
        bend = _value_noise(rng, size, size // 4)
        return 0.5 + 0.5 * np.sin(2 * np.pi * (size // scale * rows + 1.5 * bend))
    if kind == "checker":
        cells = (np.arange(size) // scale) % 2
        return 0.8 * (cells[:, None] ^ cells[None, :]) + 0.2 * _value_noise(rng, size, 8)
    if kind == "speckle":
        dots = np.ones((size, size))
        radius = max(1, scale // 8)
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disc = yy ** 2 + xx ** 2 <= radius ** 2
        for y, x in rng.integers(0, size, (int(6 * size * size / scale ** 2), 2)):
            dots[(y + yy[disc]) % size, (x + xx[disc]) % size] = 0.0
        return dots
    raise ValueError(f"unknown texture kind {kind!r}")


def texture_bank(config: AppearanceConfig) -> np.ndarray:
    """[K, size, size] uint8 greyscale. Texture k is kind k // variants, variant k % variants."""
    bank = config.bank
    rng = np.random.default_rng(int(bank["seed"]))
    size, (lo, hi) = int(bank["size_px"]), bank["luminance"]
    textures = [_texture(kind, v, rng, size)
                for kind in bank["kinds"] for v in range(int(bank["variants"]))]
    return np.round(255 * (lo + (hi - lo) * np.clip(textures, 0, 1))).astype(np.uint8)


def _bank_material(k: int, scale: str) -> str:
    return f"appearance_bank_{k}_{scale}"


def add_texture_bank(spec: mujoco.MjSpec, config: AppearanceConfig | None = None) -> None:
    """Add the bank's textures, each with a surface-scale and an object-scale material."""
    config = config or AppearanceConfig.load()
    bank = config.bank
    for k, pixels in enumerate(texture_bank(config)):
        size = pixels.shape[0]
        texture = spec.add_texture(name=f"appearance_bank_{k}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=size, height=size, nchannel=3)
        texture.data = np.repeat(pixels[..., None], 3, axis=2).tobytes()
        for scale in ("surface", "object"):
            material = spec.add_material(name=_bank_material(k, scale))
            material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
            material.texuniform = True
            repeat = float(bank[f"{scale}_repeat_per_m"])
            material.texrepeat = [repeat, repeat]


# ---------------------------------------------------------------------------- one episode's draw

@dataclass(frozen=True)
class Look:
    """How one surface or object looks."""

    rgb: tuple[float, float, float]
    texture: int  # bank index, or -1 for plain colour
    shininess: float | None = None  # objects only
    specular: float | None = None


@dataclass(frozen=True)
class Lighting:
    key_direction: tuple[float, float, float]
    key_elevation_deg: float
    key_brightness: float
    tint: tuple[float, float, float]  # per-channel scale shared by key and fill
    fill_brightness: float
    head_brightness: float
    ambient: float
    specular: float


@dataclass(frozen=True)
class Appearance:
    seed: int
    lighting: Lighting | None  # None: the model's lights
    looks: Mapping[str, Look]  # names in SURFACES and OBJECT_GEOMS; missing: the model's look


class AppearanceSampler:
    """Draws and applies the appearance of each episode for one model."""

    def __init__(self, model: mujoco.MjModel, factors: Sequence[str],
                 train_seeds: tuple[int, int], config: AppearanceConfig | None = None) -> None:
        unknown = set(factors) - set(FACTORS)
        if unknown:
            raise ValueError(f"unknown appearance factors {sorted(unknown)}; use {FACTORS}")
        self.factors = tuple(f for f in FACTORS if f in factors)
        self.train_seeds = tuple(train_seeds)
        self.config = config or AppearanceConfig.load()
        self._model = model

        def geom(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        groups = {
            "table": [geom("table_geom"), geom("fixture_wall_geom")],
            "trays": [g for g in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                      .startswith(("tray_left_", "tray_right_"))],
            "floor": [geom("floor")],
            **{name: [geom(f"{name}_geom")] for name in OBJECT_GEOMS},
        }
        if any(g < 0 for ids in groups.values() for g in ids) or not groups["trays"]:
            raise ValueError("the workcell model lacks a geom the appearance variation needs")
        self._group_of = {g: name for name, ids in groups.items() for g in ids}
        self.nominal_rgb = {name: _geom_rgb(model, ids[0]) for name, ids in groups.items()}
        self._bank_material = {
            (k, scale): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL,
                                          _bank_material(k, scale))
            for k in range(self.config.bank_size) for scale in ("surface", "object")}
        if min(self._bank_material.values()) < 0:
            raise ValueError("the workcell model lacks the texture bank (add_texture_bank)")
        self._key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key_light")
        self._fill = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "fill_light")

    def textures_for(self, seed: int) -> list[int]:
        """Bank textures this seed may show: held-out ones never in train seeds."""
        start, stop = self.train_seeds
        train = start <= seed < stop
        return [k for k in range(self.config.bank_size) if not (train and self.config.held_out(k))]

    def draw(self, seed: int) -> Appearance:
        allowed = self.textures_for(seed)
        raw = self.config.raw
        lighting, looks = None, {}
        if "lighting" in self.factors:
            lighting = _draw_lighting(np.random.default_rng([seed, STREAM, 0]), raw["lighting"])
        if "surfaces" in self.factors:
            rng = np.random.default_rng([seed, STREAM, 1])
            looks.update({name: _draw_look(rng, raw["surfaces"], allowed) for name in SURFACES})
        if "objects" in self.factors:
            rng = np.random.default_rng([seed, STREAM, 2])
            looks.update({name: _draw_look(rng, raw["objects"], allowed) for name in OBJECT_GEOMS})
        return Appearance(seed, lighting, looks)

    def apply(self, appearance: Appearance, scene: mujoco.MjvScene) -> None:
        """Restyle a scene made by ``mjv_updateScene`` for this model, before it is rendered."""
        for i in range(scene.ngeom):
            g = scene.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM or g.objid not in self._group_of:
                continue
            name = self._group_of[g.objid]
            look = appearance.looks.get(name)
            if look is None:
                continue
            g.rgba[:3] = look.rgb
            scale = "object" if name in OBJECT_GEOMS else "surface"
            g.matid = self._bank_material[look.texture, scale] if look.texture >= 0 else -1
            if look.shininess is not None:
                g.shininess, g.specular = look.shininess, look.specular
        light = appearance.lighting
        if light is None:
            return
        for i in range(scene.nlight):
            s = scene.lights[i]
            if s.headlight:
                s.diffuse[:] = light.head_brightness
                s.ambient[:] = light.ambient
            elif s.id == self._key:
                s.dir[:] = light.key_direction
                s.diffuse[:] = np.multiply(light.key_brightness, light.tint)
                s.specular[:] = light.specular
            elif s.id == self._fill:
                s.diffuse[:] = np.multiply(light.fill_brightness, light.tint)
                s.specular[:] = light.specular

    def describe(self, appearance: Appearance, target: str | None) -> dict:
        """Per-episode numbers for diagnostics: light, textures seen and unseen, contrast.
        ``target`` is the episode's object; None (press has no object) means the button, whose
        backdrop, the wall, shares the table's look."""
        target = target or "button"
        out: dict = {"factors": list(self.factors)}
        if appearance.lighting:
            out["key_brightness"] = round(appearance.lighting.key_brightness, 3)
            out["key_elevation_deg"] = round(appearance.lighting.key_elevation_deg, 1)
        for name in ("table", target):
            look = appearance.looks.get(name)
            if look:
                label = "table" if name == "table" else "object"
                out[f"{label}_texture"] = look.texture
                out[f"{label}_texture_held_out"] = self.config.held_out(look.texture)
        if target in self.nominal_rgb:
            def rgb(name):
                look = appearance.looks.get(name)
                return look.rgb if look else self.nominal_rgb[name]
            out["object_table_delta_e"] = round(delta_e(rgb(target), rgb("table")), 1)
        return out


def _draw_lighting(rng: np.random.Generator, cfg: Mapping) -> Lighting:
    azimuth = rng.uniform(0, 2 * math.pi)
    elevation = rng.uniform(*cfg["key_elevation_deg"])
    e = math.radians(elevation)
    direction = (-math.cos(e) * math.cos(azimuth), -math.cos(e) * math.sin(azimuth), -math.sin(e))
    u = rng.uniform(-1, 1)
    tint = (1 + cfg["tint"] * u, 1.0, 1 - cfg["tint"] * u)
    return Lighting(
        key_direction=direction, key_elevation_deg=elevation,
        key_brightness=rng.uniform(*cfg["key_brightness"]), tint=tint,
        fill_brightness=rng.uniform(*cfg["fill_brightness"]),
        head_brightness=rng.uniform(*cfg["head_brightness"]),
        ambient=rng.uniform(*cfg["ambient"]), specular=rng.uniform(*cfg["specular"]))


def _draw_look(rng: np.random.Generator, cfg: Mapping, allowed: Sequence[int]) -> Look:
    # every draw consumes the same random numbers, whichever branches it takes
    hue, sat, val = rng.uniform(), rng.uniform(*cfg["saturation"]), rng.uniform(*cfg["value"])
    textured = rng.uniform() < cfg["textured_fraction"]
    texture = int(allowed[rng.integers(len(allowed))])
    shine = (rng.uniform(*cfg["shininess"]), rng.uniform(*cfg["specular"])) \
        if "shininess" in cfg else (None, None)
    return Look(colorsys.hsv_to_rgb(hue, sat, val), texture if textured else -1, *shine)


def _geom_rgb(model: mujoco.MjModel, geom: int) -> tuple[float, float, float]:
    material = model.geom_matid[geom]
    rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
    return tuple(float(c) for c in rgba[:3])


def delta_e(a: Sequence[float], b: Sequence[float]) -> float:
    """CIE76 colour difference of two sRGB colours in [0, 1] (about 2: just noticeable)."""
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def _lab(rgb: Sequence[float]) -> np.ndarray:
    c = np.asarray(rgb, dtype=float)
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]]) @ linear / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
