"""Simple optical simulation + EMVA1288 noise image generation example."""

from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.polynomials import zernike_nm
from prysm.propagation import focus
from prysm.objects import siemensstar
from prysm.convolution import conv
from prysm.emva1288 import (
    EMVA1288Params,
    simulate_emva_sequence,
    estimate_emva1288,
)


def build_optical_image(samples=256, q=2):
    """Generate a blurred optical image from a simple aberrated pupil."""
    x, y = make_xy_grid(samples, diameter=2)
    r, t = cart_to_polar(x, y)

    aperture = circle(1, r)
    phase = (
        0.30 * zernike_nm(2, 0, r, t)
        + 0.20 * zernike_nm(3, 1, r, t)
        + 0.10 * zernike_nm(4, 0, r, t)
    )

    wf = aperture * np.exp(1j * phase)
    psf = abs(focus(wf, Q=q)) ** 2
    psf = psf / psf.sum()

    tx, ty = make_xy_grid(psf.shape[0], diameter=2)
    tr, tt = cart_to_polar(tx, ty)
    target = siemensstar(tr, tt, spokes=36, oradius=0.9, background='black', contrast=0.9)

    blurred = conv(target, psf)
    blurred = np.clip(blurred, 0, None)
    return target, psf, blurred


def main():
    target, psf, blurred = build_optical_image()

    # Scale optical image to incident photons/pixel for EMVA simulation.
    peak_photons = 15_000.0
    photons_map = blurred / blurred.max() * peak_photons

    params = EMVA1288Params(
        qe=0.62,
        dark_current_e_s=0.15,
        read_noise_e=2.2,
        conversion_gain_e_per_dn=0.85,
        bit_depth=12,
        full_well_e=30_000,
        offset_e=30,
        prnu_std=0.01,
        dsnu_e=1.8,
    )

    exposure_s = 0.01
    frames = 8
    noisy_stack = np.empty((frames, *photons_map.shape), dtype=np.uint16)
    dark_stack = np.empty_like(noisy_stack)

    # Generate one EMVA-noisy frame stack from the optical image.
    # We use one "condition" here by passing per-pixel photons as a scalar scale.
    for k in range(frames):
        ds = simulate_emva_sequence(
            params=params,
            photons=[peak_photons],
            exposure_s=[exposure_s],
            frames=1,
            shape=photons_map.shape,
            seed=100 + k,
        )
        bright = ds.bright_frames_dn[0, 0].astype(float)
        dark = ds.dark_frames_dn[0, 0].astype(float)

        # Modulate EMVA noise realization by optical intensity map.
        # shot component follows local photons via scaling.
        noisy = bright * (photons_map / peak_photons)
        noisy = np.clip(np.rint(noisy), 0, (2 ** params.bit_depth) - 1).astype(np.uint16)
        noisy_stack[k] = noisy
        dark_stack[k] = dark.astype(np.uint16)

    # Create a compact EMVA sweep (1D conditions) for metric estimation.
    photons_sweep = np.linspace(100, peak_photons, 12)
    exposure_sweep = np.full_like(photons_sweep, exposure_s)
    seq = simulate_emva_sequence(
        params=params,
        photons=photons_sweep,
        exposure_s=exposure_sweep,
        frames=24,
        shape=(64, 64),
        seed=123,
    )
    results = estimate_emva1288(seq, params)

    np.save('emva_target.npy', target)
    np.save('emva_psf.npy', psf)
    np.save('emva_optical_blurred.npy', blurred)
    np.save('emva_noisy_stack_dn.npy', noisy_stack)
    np.save('emva_dark_stack_dn.npy', dark_stack)

    print('Saved files:')
    print('  emva_target.npy')
    print('  emva_psf.npy')
    print('  emva_optical_blurred.npy')
    print('  emva_noisy_stack_dn.npy')
    print('  emva_dark_stack_dn.npy')
    print()
    print('Quick metrics (from EMVA sweep):')
    print(f'  qe_estimate      : {results.qe_estimate:.4f}')
    print(f'  read_noise_e     : {results.read_noise_e:.4f}')
    print(f'  dark_current_e_s : {results.dark_current_e_s:.4f}')
    print(f'  dynamic_range_db : {results.dynamic_range_db:.2f}')
    print(f'  prnu_percent     : {results.prnu_percent:.4f}')
    print(f'  dsnu_e           : {results.dsnu_e:.4f}')

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(target, cmap='gray')
        axes[0, 0].set_title('Target')
        axes[0, 1].imshow(psf, cmap='inferno')
        axes[0, 1].set_title('PSF')
        axes[0, 2].imshow(blurred, cmap='gray')
        axes[0, 2].set_title('Optical Blurred')
        axes[1, 0].imshow(noisy_stack[0], cmap='gray')
        axes[1, 0].set_title('Noisy Frame #0 (DN)')
        axes[1, 1].imshow(dark_stack[0], cmap='gray')
        axes[1, 1].set_title('Dark Frame #0 (DN)')
        axes[1, 2].plot(photons_sweep, results.snr)
        axes[1, 2].set_title('SNR Curve')
        axes[1, 2].set_xlabel('Photons/pixel')
        axes[1, 2].set_ylabel('SNR')

        for ax in axes.flat:
            if hasattr(ax, 'axis'):
                ax.axis('off')
        axes[1, 2].axis('on')

        plt.tight_layout()
        plt.savefig('example_imaging_emva1288_result.png', dpi=150)
        print('Saved figure: example_imaging_emva1288_result.png')
    except ImportError:
        print('matplotlib not installed; skipped figure rendering.')


if __name__ == '__main__':
    main()

