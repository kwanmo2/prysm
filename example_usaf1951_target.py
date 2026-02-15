"""Build USAF-1951 target images from assets/USAF1951.svg.

This script parses rectangle primitives from the SVG and renders them to a raster
image without third-party image libraries. It then writes multiple noisy/degraded
variants to PNG for imaging and robustness tests.
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

from prysm.mathops import ndimage, np


def _parse_style(style: str | None) -> dict[str, str]:
    if not style:
        return {}

    out: dict[str, str] = {}
    for part in style.split(';'):
        if ':' not in part:
            continue
        key, value = part.split(':', 1)
        out[key.strip()] = value.strip()
    return out


def _parse_fill(element: ET.Element, inherited_fill: str | None) -> str | None:
    style_fill = _parse_style(element.get('style')).get('fill')
    fill = element.get('fill', style_fill if style_fill is not None else inherited_fill)
    if fill == 'none':
        return None
    return fill


def _transform_matrix(transform: str | None) -> np.ndarray:
    m = np.eye(3, dtype=float)
    if not transform:
        return m

    rest = transform.strip()
    while rest:
        left = rest.find('(')
        right = rest.find(')', left)
        if left == -1 or right == -1:
            break

        name = rest[:left].strip()
        raw_vals = rest[left + 1:right].replace(',', ' ').split()
        vals = [float(v) for v in raw_vals]

        t = np.eye(3, dtype=float)
        if name == 'translate':
            tx = vals[0]
            ty = vals[1] if len(vals) > 1 else 0.0
            t[0, 2] = tx
            t[1, 2] = ty
        elif name == 'scale':
            sx = vals[0]
            sy = vals[1] if len(vals) > 1 else sx
            t[0, 0] = sx
            t[1, 1] = sy
        elif name == 'matrix':
            a, b, c, d, e, f = vals
            t = np.array(
                [
                    [a, c, e],
                    [b, d, f],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        else:
            raise ValueError(f'Unsupported transform operation: {name}')

        m = m @ t
        rest = rest[right + 1:].strip()

    return m


def _draw_rect(mask: np.ndarray, transform: np.ndarray, x: float, y: float, w: float, h: float) -> None:
    # Rasterize by mapping output pixel centers through inverse transform.
    corners = np.array(
        [
            [x, y, 1.0],
            [x + w, y, 1.0],
            [x, y + h, 1.0],
            [x + w, y + h, 1.0],
        ],
        dtype=float,
    )
    mapped = (transform @ corners.T).T

    minx = max(0, int(math.floor(mapped[:, 0].min())))
    maxx = min(mask.shape[1] - 1, int(math.ceil(mapped[:, 0].max())))
    miny = max(0, int(math.floor(mapped[:, 1].min())))
    maxy = min(mask.shape[0] - 1, int(math.ceil(mapped[:, 1].max())))

    if minx > maxx or miny > maxy:
        return

    inv = np.linalg.inv(transform)

    xx = np.arange(minx, maxx + 1, dtype=float) + 0.5
    yy = np.arange(miny, maxy + 1, dtype=float) + 0.5
    gx, gy = np.meshgrid(xx, yy)

    ones = np.ones_like(gx)
    pts = np.stack([gx, gy, ones], axis=0).reshape(3, -1)
    local = inv @ pts

    inside = (
        (local[0] >= x)
        & (local[0] <= x + w)
        & (local[1] >= y)
        & (local[1] <= y + h)
    )
    inside = inside.reshape(gx.shape)

    region = mask[miny:maxy + 1, minx:maxx + 1]
    region[inside] = True


def render_usaf1951(svg_path: Path, samples: int = 1024) -> np.ndarray:
    """Render black rectangles from USAF1951.svg into a grayscale image in [0,1]."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width = float(root.get('width'))
    height = float(root.get('height'))
    scale = samples / max(width, height)

    canvas_h = int(round(height * scale))
    canvas_w = int(round(width * scale))
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)

    def walk(node: ET.Element, parent_matrix: np.ndarray, inherited_fill: str | None) -> None:
        fill = _parse_fill(node, inherited_fill)
        local = _transform_matrix(node.get('transform'))
        matrix = parent_matrix @ local

        tag = node.tag.split('}')[-1]
        if tag == 'rect' and fill in ('#000000', '#000', 'black'):
            x = float(node.get('x', '0'))
            y = float(node.get('y', '0'))
            w = float(node.get('width'))
            h = float(node.get('height'))

            draw_t = np.array(
                [
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ) @ matrix
            _draw_rect(mask, draw_t, x, y, w, h)

        for child in node:
            walk(child, matrix, fill)

    walk(root, np.eye(3, dtype=float), '#000000')

    image = np.ones(mask.shape, dtype=float)
    image[mask] = 0.0
    return image


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


def add_poisson_noise(image: np.ndarray, peak_photons: float, rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(np.clip(image, 0.0, 1.0) * peak_photons)
    return np.clip(counts / peak_photons, 0.0, 1.0)


def add_salt_pepper_noise(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    n = int(amount * out.size)
    if n == 0:
        return out

    idx = rng.choice(out.size, size=n, replace=False)
    half = n // 2
    flat = out.reshape(-1)
    flat[idx[:half]] = 0.0
    flat[idx[half:]] = 1.0
    return out


def write_png_grayscale(path: Path, image: np.ndarray) -> None:
    """Write uint8 grayscale image to PNG with stdlib only."""
    u8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    h, w = u8.shape

    raw = bytearray()
    for row in u8:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack('!I', len(data))
            + tag
            + data
            + struct.pack('!I', zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack('!IIBBBBB', w, h, 8, 0, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)

    payload = b''.join(
        [
            b'\x89PNG\r\n\x1a\n',
            chunk(b'IHDR', ihdr),
            chunk(b'IDAT', idat),
            chunk(b'IEND', b''),
        ]
    )
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate USAF1951 noisy/degraded target set.')
    parser.add_argument('--svg', type=Path, default=Path('assets/USAF1951.svg'), help='Path to USAF1951.svg')
    parser.add_argument('--out', type=Path, default=Path('assets/usaf1951_testset'), help='Output directory')
    parser.add_argument('--samples', type=int, default=1024, help='Longest image dimension in pixels')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base = render_usaf1951(args.svg, samples=args.samples)

    variants: dict[str, np.ndarray] = {'clean': base}

    for sigma in (0.6, 1.2, 2.0):
        variants[f'blur_sigma_{sigma:.1f}'] = ndimage.gaussian_filter(base, sigma=sigma)

    for std in (0.01, 0.03, 0.07):
        variants[f'gauss_noise_std_{std:.2f}'] = add_gaussian_noise(base, std, rng)

    for photons in (200.0, 50.0):
        variants[f'poisson_peak_{int(photons)}'] = add_poisson_noise(base, photons, rng)

    for amount in (0.005, 0.02):
        variants[f'sp_amount_{amount:.3f}'] = add_salt_pepper_noise(base, amount, rng)

    combo = ndimage.gaussian_filter(base, sigma=1.4)
    combo = add_poisson_noise(combo, peak_photons=80.0, rng=rng)
    combo = add_gaussian_noise(combo, sigma=0.02, rng=rng)
    variants['combo_blur_poisson_gauss'] = combo

    for name, image in variants.items():
        write_png_grayscale(args.out / f'{name}.png', image)

    print(f'Wrote {len(variants)} images to {args.out.resolve()}')


if __name__ == '__main__':
    main()
