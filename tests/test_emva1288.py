"""Tests for EMVA1288 camera modeling helpers."""

import numpy as onp
import pytest

from prysm.emva1288 import (
    EMVA1288Params,
    estimate_emva1288,
    make_emva1288_plots,
    simulate_emva_sequence,
)


def test_deterministic_path_matches_theory():
    params = EMVA1288Params(
        qe=0.5,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=1,
        bit_depth=16,
        full_well_e=1e6,
        offset_e=0,
        enable_temporal_noise=False,
        enable_spatial_noise=False,
        enable_quantization=False,
    )
    photons = onp.array([0, 10, 20], dtype=float)
    ds = simulate_emva_sequence(params, photons=photons, exposure_s=1, frames=4, shape=(8, 8))
    expected = photons * params.qe
    got = ds.bright_frames_dn.mean(axis=(1, 2, 3))
    assert onp.allclose(got, expected)


def test_shot_noise_only_variance_matches_mean_in_electrons():
    params = EMVA1288Params(
        qe=1.0,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=1,
        bit_depth=16,
        full_well_e=1e8,
        enable_quantization=False,
        enable_spatial_noise=False,
    )
    ds = simulate_emva_sequence(params, photons=[1000], exposure_s=1, frames=120, shape=(64, 64), seed=123)
    vals = ds.bright_frames_dn[0].ravel()
    assert onp.var(vals) == pytest.approx(onp.mean(vals), rel=0.08)


def test_read_noise_floor_is_photon_independent():
    params = EMVA1288Params(
        qe=0,
        dark_current_e_s=0,
        read_noise_e=4,
        conversion_gain_e_per_dn=1,
        bit_depth=16,
        full_well_e=1e8,
        enable_quantization=False,
        enable_spatial_noise=False,
    )
    ds = simulate_emva_sequence(params, photons=[0, 1000, 5000], exposure_s=1, frames=80, shape=(32, 32), seed=7)
    vars_ = [onp.var(ds.bright_frames_dn[i].ravel()) for i in range(3)]
    assert max(vars_) - min(vars_) < 1.0


def test_conversion_gain_and_bitdepth_scale_dn_consistently():
    p1 = EMVA1288Params(
        qe=1.0,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=1.0,
        bit_depth=16,
        full_well_e=1e8,
        enable_temporal_noise=False,
        enable_spatial_noise=False,
    )
    p2 = EMVA1288Params(
        qe=1.0,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=0.5,
        bit_depth=16,
        full_well_e=1e8,
        enable_temporal_noise=False,
        enable_spatial_noise=False,
    )
    d1 = simulate_emva_sequence(p1, photons=[200], exposure_s=1, frames=2, shape=(8, 8))
    d2 = simulate_emva_sequence(p2, photons=[200], exposure_s=1, frames=2, shape=(8, 8))
    m1 = d1.bright_frames_dn.mean()
    m2 = d2.bright_frames_dn.mean()
    assert m2 == pytest.approx(2 * m1)


def test_saturation_creates_linearity_error():
    params = EMVA1288Params(
        qe=1.0,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=1.0,
        bit_depth=12,
        full_well_e=200,
        enable_temporal_noise=False,
        enable_spatial_noise=False,
    )
    photons = onp.linspace(10, 600, 20)
    ds = simulate_emva_sequence(params, photons=photons, exposure_s=1, frames=4, shape=(8, 8))
    result = estimate_emva1288(ds, params)
    assert result.linearity_max_error_pct > 5


def test_prnu_dsnu_estimates_track_inputs():
    params = EMVA1288Params(
        qe=0.8,
        dark_current_e_s=0.1,
        read_noise_e=1.0,
        conversion_gain_e_per_dn=1.0,
        bit_depth=16,
        full_well_e=1e8,
        offset_e=20,
        prnu_std=0.03,
        dsnu_e=5.0,
        enable_quantization=False,
    )
    photons = onp.linspace(100, 5000, 8)
    exposure = onp.linspace(0.01, 0.08, 8)
    ds = simulate_emva_sequence(params, photons=photons, exposure_s=exposure, frames=64, shape=(64, 64), seed=11)
    result = estimate_emva1288(ds, params)
    assert result.prnu_percent == pytest.approx(100 * params.prnu_std, abs=0.8)
    assert result.dsnu_e == pytest.approx(params.dsnu_e, abs=1.2)


def test_seed_reproducibility():
    params = EMVA1288Params(
        qe=0.7,
        dark_current_e_s=0.1,
        read_noise_e=2.0,
        conversion_gain_e_per_dn=0.8,
        bit_depth=14,
        full_well_e=50000,
        prnu_std=0.01,
        dsnu_e=1.5,
    )
    d1 = simulate_emva_sequence(params, photons=[100, 500], exposure_s=[0.01, 0.02], frames=10, shape=(16, 16), seed=99)
    d2 = simulate_emva_sequence(params, photons=[100, 500], exposure_s=[0.01, 0.02], frames=10, shape=(16, 16), seed=99)
    assert onp.array_equal(d1.bright_frames_dn, d2.bright_frames_dn)
    assert onp.array_equal(d1.dark_frames_dn, d2.dark_frames_dn)


def test_plot_payload_contains_core_keys():
    params = EMVA1288Params(
        qe=1.0,
        dark_current_e_s=0,
        read_noise_e=0,
        conversion_gain_e_per_dn=1,
        bit_depth=16,
        full_well_e=1e6,
        enable_temporal_noise=False,
        enable_spatial_noise=False,
    )
    ds = simulate_emva_sequence(params, photons=[10, 20, 30], exposure_s=[1, 1, 1], frames=4, shape=(8, 8))
    result = estimate_emva1288(ds, params)
    plots = make_emva1288_plots(result)
    for k in ('sensitivity', 'snr', 'linearity', 'dark', 'prnu_spectrogram', 'dsnu_spectrogram', 'prnu_histogram', 'dsnu_histogram'):
        assert k in plots
