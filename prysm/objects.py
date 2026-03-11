"""Objects for image simulation with."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .conf import config
from .mathops import np, jinc
from .coordinates import optimize_xy_separable


_MATRIX_RE = re.compile(r'matrix\(([^)]*)\)')
_TRANSLATE_RE = re.compile(r'translate\(([^)]*)\)')


def _parse_float_list(txt):
    return [float(x) for x in re.split(r'[,\s]+', txt.strip()) if x]


def _transform_matrix(transform_text):
    mat = np.eye(3, dtype=float)
    if not transform_text:
        return mat

    m = _MATRIX_RE.search(transform_text)
    if m is not None:
        vals = _parse_float_list(m.group(1))
        if len(vals) == 6:
            a, b, c, d, e, f = vals
            m2 = np.asarray([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)
            mat = m2 @ mat

    t = _TRANSLATE_RE.search(transform_text)
    if t is not None:
        vals = _parse_float_list(t.group(1))
        tx = vals[0] if len(vals) > 0 else 0.0
        ty = vals[1] if len(vals) > 1 else 0.0
        t2 = np.asarray([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
        mat = t2 @ mat

    return mat


def _svg_len_to_float(s):
    if s is None:
        return 0.0
    m = re.match(r'([-+]?\d*\.?\d+)', str(s))
    return float(m.group(1)) if m else 0.0


def _style_fill(style_txt):
    if not style_txt:
        return None
    for item in style_txt.split(';'):
        if ':' not in item:
            continue
        k, v = item.split(':', 1)
        if k.strip() == 'fill':
            return v.strip().lower()
    return None


def _is_black_fill(elem):
    fill = (elem.attrib.get('fill') or _style_fill(elem.attrib.get('style')) or '').lower()
    return fill in ('#000', '#000000', 'black', 'rgb(0,0,0)')


def _iter_svg_rects(node, parent_mat):
    node_mat = _transform_matrix(node.attrib.get('transform'))
    cur = parent_mat @ node_mat

    tag = node.tag.rsplit('}', 1)[-1].lower()
    if tag == 'rect' and _is_black_fill(node):
        x = _svg_len_to_float(node.attrib.get('x', 0))
        y = _svg_len_to_float(node.attrib.get('y', 0))
        w = _svg_len_to_float(node.attrib.get('width', 0))
        h = _svg_len_to_float(node.attrib.get('height', 0))
        if w > 0 and h > 0:
            corners = np.asarray(
                [[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]],
                dtype=float,
            )
            pts = (cur @ corners.T).T[:, :2]
            yield pts

    for child in list(node):
        yield from _iter_svg_rects(child, cur)


def usaf1951(samples, svg_path=None):
    """Rasterize the bundled USAF1951 SVG test target.

    Parameters
    ----------
    samples : int
        width/height of output square array
    svg_path : str or Path, optional
        path to an alternate USAF1951 SVG; defaults to prysm's bundled asset

    Returns
    -------
    ndarray
        2D float array in [0, 1], with bright bars on dark background

    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError('Pillow is required for objects.usaf1951') from exc

    if svg_path is None:
        svg_path = Path(__file__).resolve().parents[1] / 'assets' / 'USAF1951.svg'
    else:
        svg_path = Path(svg_path)

    root = ET.parse(svg_path).getroot()
    w = _svg_len_to_float(root.attrib.get('width', 256))
    h = _svg_len_to_float(root.attrib.get('height', 256))

    scale = min(samples / max(w, 1e-9), samples / max(h, 1e-9))
    offx = 0.5 * (samples - scale * w)
    offy = 0.5 * (samples - scale * h)

    img = Image.new('L', (samples, samples), color=255)
    draw = ImageDraw.Draw(img)

    for poly in _iter_svg_rects(root, np.eye(3, dtype=float)):
        pix = []
        for x, y in poly:
            xp = offx + x * scale
            yp = offy + y * scale
            pix.append((float(xp), float(yp)))
        draw.polygon(pix, fill=0)

    arr = np.asarray(img, dtype=float) / 255.0
    return 1.0 - arr


def slit(x, y, width_x, width_y=None):
    """Rasterize a slit or pair of crossed slits.

    Parameters
    ----------
    x : ndarray
        x coordinates, 1D or 2D
    y : ndarray
        y coordinates, 1D or 2D
    width_x : float
        the half-width of the slit in x, diameter will be 2x width_x.
        produces a line along the y axis, use None to not do so
    width_y : float
        the half-height of the slit in y, diameter will be 2x width_y.
        produces a line along the y axis, use None to not do so
    orientation : string, {'Horizontal', 'Vertical', 'Crossed', 'Both'}
        the orientation of the slit; Crossed and Both produce the same results

    Notes
    -----
    Default of 0 samples allows quick creation for convolutions without
    generating the image; use samples > 0 for an actual image.

    """
    x, y = optimize_xy_separable(x, y)
    mask = np.zeros((y.size, x.size), dtype=bool)
    if width_x is not None:
        wx = width_x / 2
        mask |= abs(x) <= wx
    if width_y is not None:
        wy = width_y / 2
        mask |= abs(y) <= wy

    return mask


def slit_ft(width_x, width_y, fx, fy):
    """Analytic fourier transform of a slit.

    Parameters
    ----------
    width_x : float
        x width of the slit, pass zero if the slit only has width in y
    width_y : float
        y width of the slit, pass zero if the slit only has width in x
    fx : ndarray
        sample points in x frequency axis
    fy : ndarray
        sample points in y frequency axis

    Returns
    -------
    ndarray
        2D array containing the analytic fourier transform

    """
    if width_x is not None and width_y is not None:
        return (np.sinc(fx * width_x) +
                np.sinc(fy * width_y)).astype(config.precision)
    elif width_x is not None and width_y is None:
        return np.sinc(fx * width_x).astype(config.precision)
    else:
        return np.sinc(fy * width_y).astype(config.precision)


def pinhole(radius, rho):
    """Rasterize a pinhole.

    Parameters
    ----------
    radius : float
        radius of the pinhole
    rho : ndarray
        radial coordinates

    Returns
    -------
    ndarray
        2D array containing the pinhole

    """
    return rho <= radius


def pinhole_ft(radius, fr):
    """Analytic fourier transform of a pinhole.

    Parameters
    ----------
    radius : float
        radius of the pinhole
    fr : ndarray
        radial spatial frequency

    Returns
    -------
    ndarray
        2D array containing the analytic fourier transform

    """
    fr2 = fr * (radius * 2 * np.pi)
    return jinc(fr2)


def siemensstar(r, t, spokes, oradius=0.9, iradius=0, background='black', contrast=0.9, sinusoidal=False):
    """Rasterize a Siemen's Star.

    Parameters
    ----------
    r : ndarray
        radial coordinates, 2D
    t : ndarray
        azimuthal coordinates, 2D
    spokes : int
        number of spokes in the star
    oradius : float
        outer radius of the star
    iradius : float
        inner radius of the star
    background : str, optional, {'black', 'white'}
        background color
    contrast : float, optional
        contrast of the star, 1 = perfect black/white
    sinusoidal : bool, optional
        if True, generates a sinusoidal Siemen' star, else, generates a bar/block siemen's star

    Returns
    -------
    ndarray
        2D array of the same shape as r, t which is in the range [0,1]

    """
    background = background.lower()
    delta = (1 - contrast)/2
    bottom = delta
    top = 1 - delta
    # generate the siemen's star as a (rho,phi) polynomial
    arr = contrast * np.cos(spokes / 2 * t)

    # scale to (0,1) and clip into a disk
    arr = (arr + 1) / 2
    mask = r > oradius
    mask |= r < iradius

    if background in ('b', 'black'):
        arr[mask] = 0
    elif background in ('w', 'white'):
        arr[mask] = 1
    else:
        raise ValueError('invalid background color')

    if not sinusoidal:  # make binary
        arr[arr < 0.5] = bottom
        arr[arr > 0.5] = top

    return arr


def tiltedsquare(x, y, angle=4, radius=0.5, contrast=0.9, background='white'):
    """Rasterize a tilted square.

    Parameters
    ----------
    x : ndarray
        x coordinates, 2D
    y : ndarray
        y coordinates, 2D
    angle : float
        counter-clockwise angle of the square from x, degrees
    radius : float
        radius of the square
    contrast : float
        contrast of the square
    background: str, optional, {'white', 'black'}
        whether to paint a white square on a black background or vice-versa

    Returns
    -------
    ndarray
        ndarray containing the rasterized square

    """
    background = background.lower()
    delta = (1 - contrast) / 2

    angle = np.radians(angle)
    xp = x * np.cos(angle) - y * np.sin(angle)
    yp = x * np.sin(angle) + y * np.cos(angle)
    mask = (abs(xp) <= radius) * (abs(yp) <= radius)

    arr = np.zeros_like(x)
    if background in ('b', 'white'):
        arr[~mask] = (1 - delta)
        arr[mask] = delta
    else:
        arr[~mask] = delta
        arr[mask] = (1 - delta)

    return arr


def slantededge(x, y, angle=4, contrast=0.9, crossed=False):
    """Rasterize a slanted edge.

    Parameters
    ----------
    x : ndarray
        x coordinates, 2D
    y : ndarray
        y coordinates, 2D
    angle : float
        angle of the edge to the cartesian y axis
    contrast : float
        contrast of the edge
    crossed : bool, optional
        if True, draw crossed edges instead of just one

    """

    diff = (1 - contrast) / 2
    arr = np.full(x.shape, 1 - diff)

    angle = np.radians(angle)
    xp = x * np.cos(angle) - y * np.sin(angle)
    mask = xp > 0  # single edge
    if crossed:
        mask = xp > 0  # set of 4 edges
        upperright = mask & np.rot90(mask)
        lowerleft = np.rot90(upperright, 2)
        mask = upperright | lowerleft

    arr[mask] = diff

    return arr
