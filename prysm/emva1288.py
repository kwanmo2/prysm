"""EMVA1288-oriented camera simulation and metric estimation tools.

Notes
-----
This module implements a practical subset of EMVA1288 Release 3.1a concepts.
The model operates in three domains with explicit units:

1. photons/pixel -> electrons via QE
2. electrons -> DN via conversion gain (e-/DN)
3. DN -> quantized/clipped ADC output

The estimation routines focus on key metrics used in EMVA workflows:
sensitivity/QE, SNR curve, linearity, dark current, temporal dark noise,
dynamic range, and fixed-pattern components (PRNU/DSNU).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .mathops import np
from .detector import apply_lut


@dataclass
class EMVA1288Params:
    """Physical and digital parameters for EMVA-like simulation."""

    qe: float
    dark_current_e_s: float
    read_noise_e: float
    conversion_gain_e_per_dn: float
    bit_depth: int
    full_well_e: float
    offset_e: float = 0.0
    prnu_std: float = 0.0
    dsnu_e: float = 0.0
    lut: Optional[np.ndarray] = None
    enable_temporal_noise: bool = True
    enable_spatial_noise: bool = True
    enable_quantization: bool = True


@dataclass
class EMVA1288Dataset:
    """Frames and acquisition conditions used for EMVA metric estimation."""

    photons: np.ndarray
    exposure_s: np.ndarray
    bright_frames_dn: np.ndarray
    dark_frames_dn: np.ndarray
    gain_e_per_dn: float
    bit_depth: int
    metadata: Optional[dict] = None


@dataclass
class EMVA1288Results:
    """Estimated EMVA-oriented metrics and intermediate curves."""

    photons: np.ndarray
    exposure_s: np.ndarray
    mean_bright_dn: np.ndarray
    mean_dark_dn: np.ndarray
    net_mean_dn: np.ndarray
    temporal_var_bright_dn: np.ndarray
    temporal_var_dark_dn: np.ndarray
    snr: np.ndarray
    sensitivity_e_per_photon: float
    qe_estimate: float
    linearity_fit_dn: np.ndarray
    linearity_deviation_pct: np.ndarray
    linearity_max_error_pct: float
    dark_current_e_s: float
    temporal_dark_noise_e: float
    read_noise_e: float
    dynamic_range_db: float
    prnu_percent: float
    dsnu_e: float
    prnu_map: np.ndarray
    dsnu_map_e: np.ndarray


def _adc_dtype(bit_depth):
    if bit_depth <= 8:
        return np.uint8
    if bit_depth <= 16:
        return np.uint16
    if bit_depth <= 32:
        return np.uint32
    raise ValueError('bit_depth > 32 is not supported')


def _as_conditions(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _broadcast_conditions(photons, exposure_s):
    photons = _as_conditions(photons)
    exposure_s = _as_conditions(exposure_s)
    if photons.size == 1 and exposure_s.size > 1:
        photons = np.full(exposure_s.shape, float(photons[0]))
    elif exposure_s.size == 1 and photons.size > 1:
        exposure_s = np.full(photons.shape, float(exposure_s[0]))
    elif photons.size != exposure_s.size:
        raise ValueError('photons and exposure_s must have equal lengths or be scalar')
    return photons, exposure_s


def _prepare_spatial_maps(shape, params):
    prnu_map = np.ones(shape, dtype=float)
    dsnu_map = np.zeros(shape, dtype=float)
    if params.enable_spatial_noise:
        if params.prnu_std > 0:
            prnu_map = prnu_map + np.random.normal(0, params.prnu_std, shape)
        if params.dsnu_e > 0:
            dsnu_map = np.random.normal(0, params.dsnu_e, shape)
    return prnu_map, dsnu_map


def _simulate_frame_stack(signal_e, dark_e, shape, frames, params, prnu_map, dsnu_map, use_prnu):
    expected = signal_e + dark_e
    expected = np.maximum(expected, 0)
    if use_prnu:
        expected = expected * prnu_map

    expected = expected.ravel()
    dsnu = np.asarray(dsnu_map, dtype=float).ravel()
    if params.enable_temporal_noise:
        shot = np.random.poisson(expected, (frames, expected.size))
        read = np.random.normal(0, params.read_noise_e, shot.shape)
        electrons = shot + read + dsnu + params.offset_e
    else:
        electrons = np.tile(expected, (frames, 1)) + dsnu + params.offset_e

    electrons = np.clip(electrons, 0, params.full_well_e)
    dn = electrons / params.conversion_gain_e_per_dn

    adc_max = 2 ** params.bit_depth - 1
    dn = np.clip(dn, 0, adc_max)
    if params.enable_quantization:
        dn = np.rint(dn)
        dn = dn.astype(_adc_dtype(params.bit_depth))

    dn = dn.reshape((frames, *shape))
    if params.lut is not None:
        if not np.issubdtype(dn.dtype, np.integer):
            raise ValueError('lut requires quantized integer DN values')
        dn = apply_lut(dn, params.lut)
    return dn


def simulate_emva_sequence(params, photons, exposure_s, frames, shape, seed=None):
    """Generate bright/dark image stacks for EMVA-style analysis.

    Parameters
    ----------
    params : EMVA1288Params
        camera parameters
    photons : float or ndarray
        incident photons per pixel, one value per condition
    exposure_s : float or ndarray
        exposure time in seconds, one value per condition
    frames : int
        number of frames per condition
    shape : tuple[int, int]
        image shape
    seed : int, optional
        random seed for reproducibility

    Returns
    -------
    EMVA1288Dataset
        bright and dark frame stacks

    """
    photons, exposure_s = _broadcast_conditions(photons, exposure_s)
    if seed is not None:
        np.random.seed(seed)

    ncond = photons.size
    bright = np.empty((ncond, frames, *shape), dtype=float)
    dark = np.empty((ncond, frames, *shape), dtype=float)

    prnu_map, dsnu_map = _prepare_spatial_maps(shape, params)
    for i in range(ncond):
        signal_e = photons[i] * params.qe
        dark_e = exposure_s[i] * params.dark_current_e_s
        bright_i = _simulate_frame_stack(
            signal_e,
            dark_e,
            shape,
            frames,
            params,
            prnu_map,
            dsnu_map,
            use_prnu=True,
        )
        dark_i = _simulate_frame_stack(
            0.0,
            dark_e,
            shape,
            frames,
            params,
            np.ones(shape, dtype=float),
            dsnu_map,
            use_prnu=False,
        )
        bright[i] = bright_i
        dark[i] = dark_i

    if params.enable_quantization:
        bright = bright.astype(_adc_dtype(params.bit_depth))
        dark = dark.astype(_adc_dtype(params.bit_depth))

    return EMVA1288Dataset(
        photons=photons,
        exposure_s=exposure_s,
        bright_frames_dn=bright,
        dark_frames_dn=dark,
        gain_e_per_dn=params.conversion_gain_e_per_dn,
        bit_depth=params.bit_depth,
        metadata={'seed': seed},
    )


def _temporal_var_pairwise(frames):
    frames = np.asarray(frames, dtype=float)
    n = frames.shape[0]
    if n < 2:
        return float(np.var(frames, ddof=1))

    npairs = n // 2
    if npairs == 0:
        return float(np.var(frames, ddof=1))
    a = frames[0:2 * npairs:2]
    b = frames[1:2 * npairs:2]
    diff = a - b
    return float(np.var(diff, ddof=1) / 2.0)


def _linefit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(x, x[0]):
        slope = 0.0
        intercept = float(np.mean(y))
        fit = slope * x + intercept
        return slope, intercept, fit
    coef = np.polyfit(x, y, 1)
    fit = coef[0] * x + coef[1]
    return coef[0], coef[1], fit


def _safe_sqrt(x):
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), 0))


def estimate_emva1288(dataset, params):
    """Estimate key EMVA-like metrics from a dataset.

    Parameters
    ----------
    dataset : EMVA1288Dataset
        bright/dark frame stacks by condition
    params : EMVA1288Params
        camera parameters

    Returns
    -------
    EMVA1288Results
        estimated metrics and curves

    """
    bright = np.asarray(dataset.bright_frames_dn, dtype=float)
    dark = np.asarray(dataset.dark_frames_dn, dtype=float)
    photons = np.asarray(dataset.photons, dtype=float)
    exposure_s = np.asarray(dataset.exposure_s, dtype=float)
    kg = float(dataset.gain_e_per_dn)

    ncond = bright.shape[0]
    mean_bright_dn = np.zeros(ncond, dtype=float)
    mean_dark_dn = np.zeros(ncond, dtype=float)
    var_bright_dn = np.zeros(ncond, dtype=float)
    var_dark_dn = np.zeros(ncond, dtype=float)

    for i in range(ncond):
        mean_bright_dn[i] = float(np.mean(bright[i]))
        mean_dark_dn[i] = float(np.mean(dark[i]))
        var_bright_dn[i] = _temporal_var_pairwise(bright[i])
        var_dark_dn[i] = _temporal_var_pairwise(dark[i])

    net_mean_dn = mean_bright_dn - mean_dark_dn
    net_mean_e = net_mean_dn * kg
    var_bright_e = var_bright_dn * kg ** 2
    var_dark_e = var_dark_dn * kg ** 2

    sens, _, _ = _linefit(photons, net_mean_e)
    qe_estimate = float(sens)

    snr = net_mean_e / np.maximum(_safe_sqrt(var_bright_e), 1e-12)

    adc_max = 2 ** dataset.bit_depth - 1
    mask = (mean_bright_dn > 0.05 * adc_max) & (mean_bright_dn < 0.95 * adc_max)
    if mask.sum() < 2:
        mask = net_mean_dn > 0

    lin_slope, lin_intercept, linearity_fit = _linefit(photons[mask], mean_bright_dn[mask])
    linearity_fit_full = lin_slope * photons + lin_intercept
    with np.errstate(divide='ignore', invalid='ignore'):
        linearity_deviation_pct = 100 * (mean_bright_dn - linearity_fit_full) / np.maximum(linearity_fit_full, 1e-12)
    linearity_max_error_pct = float(np.nanmax(np.abs(linearity_deviation_pct[mask])))

    dark_mean_e = mean_dark_dn * kg
    dc_slope, _, _ = _linefit(exposure_s, dark_mean_e)
    dark_current_e_s = float(dc_slope)

    dark_var_fit_slope, dark_var_fit_int, _ = _linefit(exposure_s, var_dark_e)
    temporal_dark_noise_e = float(_safe_sqrt(dark_var_fit_int))
    read_noise_e = temporal_dark_noise_e

    if temporal_dark_noise_e <= 0:
        dynamic_range_db = np.inf
    else:
        dynamic_range_db = float(20 * np.log10(params.full_well_e / temporal_dark_noise_e))

    usable = (net_mean_dn > 0) & (mean_bright_dn < 0.98 * adc_max)
    if usable.any():
        idx = int(np.where(usable)[0][-1])
    else:
        idx = int(np.argmax(net_mean_dn))

    mean_img_bright = np.mean(bright[idx], axis=0)
    mean_img_dark = np.mean(dark[idx], axis=0)
    signal_img = np.maximum(mean_img_bright - mean_img_dark, 0)
    prnu_map = signal_img - float(np.mean(signal_img))
    signal_mean = float(np.mean(signal_img))
    if signal_mean <= 0:
        prnu_percent = 0.0
    else:
        prnu_percent = float(100 * np.std(prnu_map) / signal_mean)

    dsnu_map_dn = mean_img_dark - float(np.mean(mean_img_dark))
    dsnu_map_e = dsnu_map_dn * kg
    dsnu_e = float(np.std(dsnu_map_e))

    return EMVA1288Results(
        photons=photons,
        exposure_s=exposure_s,
        mean_bright_dn=mean_bright_dn,
        mean_dark_dn=mean_dark_dn,
        net_mean_dn=net_mean_dn,
        temporal_var_bright_dn=var_bright_dn,
        temporal_var_dark_dn=var_dark_dn,
        snr=snr,
        sensitivity_e_per_photon=float(sens),
        qe_estimate=qe_estimate,
        linearity_fit_dn=linearity_fit_full,
        linearity_deviation_pct=linearity_deviation_pct,
        linearity_max_error_pct=linearity_max_error_pct,
        dark_current_e_s=dark_current_e_s,
        temporal_dark_noise_e=temporal_dark_noise_e,
        read_noise_e=read_noise_e,
        dynamic_range_db=dynamic_range_db,
        prnu_percent=prnu_percent,
        dsnu_e=dsnu_e,
        prnu_map=prnu_map,
        dsnu_map_e=dsnu_map_e,
    )


def _spectrogram_axes(map2d):
    m = np.asarray(map2d, dtype=float)
    hor = np.mean(np.abs(np.fft.rfft(m, axis=1)), axis=0)
    ver = np.mean(np.abs(np.fft.rfft(m, axis=0)), axis=1)
    return hor, ver


def _hist_and_accum(values, bins=128):
    hist, edges = np.histogram(values, bins=bins)
    accum = np.cumsum(hist) / np.maximum(np.sum(hist), 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist, accum


def make_emva1288_plots(results):
    """Return data series for EMVA template-style plots.

    Parameters
    ----------
    results : EMVA1288Results
        output of estimate_emva1288

    Returns
    -------
    dict
        dictionary of plot-ready arrays for sensitivity/SNR/linearity and
        PRNU/DSNU spectrum and histogram visualizations.

    """
    prnu_hor, prnu_ver = _spectrogram_axes(results.prnu_map)
    dsnu_hor, dsnu_ver = _spectrogram_axes(results.dsnu_map_e)
    prnu_c, prnu_h, prnu_acc = _hist_and_accum(results.prnu_map.ravel())
    dsnu_c, dsnu_h, dsnu_acc = _hist_and_accum(results.dsnu_map_e.ravel())

    return {
        'sensitivity': {
            'x_photons': results.photons,
            'y_net_mean_dn': results.net_mean_dn,
        },
        'snr': {
            'x_photons': results.photons,
            'y_snr': results.snr,
        },
        'linearity': {
            'x_photons': results.photons,
            'y_mean_dn': results.mean_bright_dn,
            'y_fit_dn': results.linearity_fit_dn,
            'y_dev_pct': results.linearity_deviation_pct,
        },
        'dark': {
            'x_exposure_s': results.exposure_s,
            'y_mean_dark_dn': results.mean_dark_dn,
            'y_var_dark_dn': results.temporal_var_dark_dn,
        },
        'prnu_spectrogram': {
            'horizontal': prnu_hor,
            'vertical': prnu_ver,
        },
        'dsnu_spectrogram': {
            'horizontal': dsnu_hor,
            'vertical': dsnu_ver,
        },
        'prnu_histogram': {
            'x': prnu_c,
            'y': prnu_h,
            'y_accum': prnu_acc,
        },
        'dsnu_histogram': {
            'x': dsnu_c,
            'y': dsnu_h,
            'y_accum': dsnu_acc,
        },
    }
