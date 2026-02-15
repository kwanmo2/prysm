"""
Target → Lens → Image Sensor 시뮬레이션 예제

시뮬레이션 흐름:
1. 타겟(Siemens' Star) 생성
2. 렌즈의 PSF 계산 (동공 + 수차 → FFT 전파)
3. PSF로 타겟을 컨볼루션하여 광학 이미지 생성
4. 디텍터 노이즈를 적용하여 최종 센서 출력 생성
"""
from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.polynomials import zernike_nm
from prysm.propagation import focus
from prysm.objects import siemensstar
from prysm.convolution import conv
from prysm.detector import Detector, bindown

# ============================================================
# 1. 렌즈 PSF 계산
# ============================================================
pupil_samples = 256
x, y = make_xy_grid(pupil_samples, diameter=2)
r, t = cart_to_polar(x, y)

# 원형 동공
aperture = circle(1, r)

# Zernike 수차 추가 (단위: radians of phase)
#   defocus (n=2, m=0) + coma (n=3, m=1) + spherical (n=4, m=0)
phase = (0.3 * zernike_nm(2, 0, r, t)
       + 0.2 * zernike_nm(3, 1, r, t)
       + 0.1 * zernike_nm(4, 0, r, t))

# 복소 wavefront → PSF
wf = aperture * np.exp(1j * phase)
Q = 2  # oversampling factor
psf_field = focus(wf, Q=Q)
psf = abs(psf_field) ** 2
psf = psf / psf.sum()  # 에너지 정규화

print(f"PSF shape: {psf.shape}")

# ============================================================
# 2. 타겟 생성 (Siemens' Star)
# ============================================================
# PSF와 같은 크기의 타겟 생성
target_samples = psf.shape[0]
tx, ty = make_xy_grid(target_samples, diameter=2)
tr, tt = cart_to_polar(tx, ty)

target = siemensstar(tr, tt, spokes=36, oradius=0.9, background='black', contrast=0.9)
print(f"Target shape: {target.shape}")

# ============================================================
# 3. 광학 이미지 생성 (컨볼루션)
# ============================================================
blurred = conv(target, psf)
# 음수 값 클리핑 (수치 오차)
blurred = np.clip(blurred, 0, None)
print(f"Blurred image shape: {blurred.shape}")

# ============================================================
# 4. 디텍터 시뮬레이션
# ============================================================
# aerial image를 e-/sec 단위로 스케일링
peak_signal = 50000  # e-/sec at brightest pixel
aerial_img = blurred / blurred.max() * peak_signal

detector = Detector(
    dark_current=5,         # e-/sec
    read_noise=10,          # e-
    bias=100,               # e-
    fwc=100_000,            # e- (full well capacity)
    conversion_gain=2.0,    # e-/DN
    bits=12,                # 12-bit ADC
    exposure_time=0.01,     # 10ms exposure
)

sensor_output = detector.expose(aerial_img, frames=1)
print(f"Sensor output shape: {sensor_output.shape}, dtype: {sensor_output.dtype}")
print(f"Sensor DN range: [{sensor_output.min()}, {sensor_output.max()}]")

# ============================================================
# 5. 2x2 binning (선택적)
# ============================================================
binned = bindown(sensor_output.astype(float), factor=2, mode='avg')
print(f"Binned shape: {binned.shape}")

# ============================================================
# 6. 시각화 (matplotlib 필요)
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(target, cmap='gray')
    axes[0, 0].set_title('Target (Siemens Star)')

    axes[0, 1].imshow(psf, cmap='inferno')
    axes[0, 1].set_title('PSF (with aberrations)')

    axes[1, 0].imshow(blurred, cmap='gray')
    axes[1, 0].set_title('Optical Image (convolved)')

    axes[1, 1].imshow(sensor_output, cmap='gray')
    axes[1, 1].set_title('Sensor Output (12-bit, noisy)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('example_imaging_result.png', dpi=150)
    print("\nResult saved to example_imaging_result.png")
    plt.show()

except ImportError:
    print("\nmatplotlib not installed — skipping visualization.")
    print("Install with: pip install matplotlib")
