# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prysm is a Python 3.10+ library for numerical optics. It covers physical optics propagation, polynomial basis functions, interferometry, detector modeling, thin lens/film optics, segmented mirror systems, and ray tracing. Only numpy is required; scipy is used throughout via the backend shim.

## Build and Test Commands

```bash
# Install (uses Poetry)
pip install -e .

# Run all tests with coverage
pytest --cov

# Run a single test file
pytest tests/test_propagation.py

# Run a specific test by name
pytest -k "test_name_pattern"
```

CI runs on Python 3.11 via CircleCI. No linter is configured in CI; pydocstyle rules are specified in `pyproject.toml`.

## Architecture

### Backend Abstraction (`prysm/mathops.py`)

The most important architectural pattern. `mathops.py` wraps numpy, scipy.fft, scipy.ndimage, scipy.special, and scipy.interpolate in `BackendShim` objects. All modules import from `prysm.mathops` instead of numpy/scipy directly:

```python
from prysm.mathops import np, fft, special, ndimage, interpolate
```

This enables runtime backend swapping (`set_backend_to_cupy()`, `set_backend_to_pytorch()`) for GPU acceleration without code changes. When writing new code, always import math operations from `prysm.mathops`, never directly from numpy/scipy.

### Global Configuration (`prysm/conf.py`)

`config` singleton controls precision (32/64-bit float/complex types), plot settings. Access via `from prysm.conf import config`.

### RichData Base Class (`prysm/_richdata.py`)

Base class for 2D data with spatial metadata (dx spacing, wavelength). Provides lazy-loaded coordinate properties (x, y, r, t), interpolation, and plotting. Many data-carrying objects inherit from this.

### Module Organization

- **`prysm/`** — Core library: propagation, interferometry, coordinates, geometry, detectors, PSF/OTF metrics, I/O, convolution, thin lens/film, segmented systems
- **`prysm/polynomials/`** — All polynomial types (Zernike, Chebyshev, Legendre, Jacobi, Hermite, Laguerre, Dickson, Q-polynomials, XY). Each module follows a consistent API: `func(n, x)`, `func_seq(ns, x)`, `func_der(n, x)`, `func_der_seq(ns, x)`. The `__init__.py` re-exports everything and provides `sum_of_2d_modes()`, `lstsq()`, and `hopkins()`.
- **`prysm/x/`** — Extended/optional modules: deformable mirrors, fibers, polarization, phase diversity, Shack-Hartmann, phase-shifting interferometry
- **`prysm/x/optym/`** — Optimization (optimizers, cost functions, activation functions, operators)
- **`prysm/x/raytracing/`** — Sequential ray tracing (surfaces, ray generation, Spencer-Murty algorithm)

### Test Coverage

Coverage is configured to include `prysm/*` but omit `prysm/x/*` and `tests/*`. The `prysm/x/` modules are experimental/extended and not covered by CI.
