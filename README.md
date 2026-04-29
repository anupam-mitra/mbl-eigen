# Eigenvector and eigenvalue analysis for systems with many-body localization

This repository provides command-line workflows and a reusable Python package for
analyzing quantum many-body scars (QMBS), many-body localized discrete time
crystals (MBL-DTC), random-field many-body localization (MBL), return-rate
dynamics, and operator overlap dynamics.

The root `main_*.py` files are compatibility shims. The reusable implementation
now lives in the `mbl_eigen/` package.

## Setup

The current dependency source of truth is `requirements.txt` and
`pyproject.toml`.

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `qutip`
- `qutip-qip`
- `jupyter`
- `h5py`

Install the environment with:

```bash
python3 -m pip install -r requirements.txt
```

The package metadata currently requires Python 3.10 or newer.

Optional eigenvalue backends are available as extras:

```bash
python3 -m pip install .[torch]
python3 -m pip install .[jax]
```

Run commands from the repository root so both the root shim scripts and
`import mbl_eigen` resolve correctly.

## Repository Layout

- `main_qmbs.py`: CLI shim for QMBS / PXP eigenvalue and eigenphase analysis.
- `main_mbldtc.py`: CLI shim for MBL-DTC Floquet analysis.
- `main_mbl.py`: CLI shim for MBL spectrum and eigenvector entropy analysis.
- `main_mbl_dynamics.py`: CLI shim for MBL return-rate dynamics.
- `main_mbl_propagator.py`: CLI shim for propagator / overlap analysis.
- `mbl_eigen/cli.py`: shared `argparse` builders for the CLI shims.
- `mbl_eigen/qmbs_app.py`: QMBS implementation.
- `mbl_eigen/mbldtc_app.py`: MBL-DTC implementation.
- `mbl_eigen/mbl_app.py`: MBL, MBL dynamics, and MBL propagator implementations.
- `mbl_eigen/mbl_model.py`: shared random-field MBL Hamiltonian builder.
- `mbl_eigen/eigensolver.py`: backend-selectable eigenvalue/eigenvector solver layer.
- `mbl_eigen/output_names.py`: centralized output filename formatting.
- `mbl_eigen/level_repulsion.py`: adjacent-level-spacing-ratio helper.
- `mbl_eigen/reflection.py`: reflection-symmetry utilities.
- `mbl_eigen/symmetry.py`: symmetry base classes.
- `level_repulsion.py`, `reflection.py`, `symmetry.py`: compatibility wrappers
  that re-export the package implementations.

## CLI Overview

The repository currently exposes five root-level commands:

| Script | Purpose | Output |
| --- | --- | --- |
| `main_qmbs.py` | QMBS and PXP eigenphase analysis | Two stable PDF files |
| `main_mbldtc.py` | Floquet eigenphase analysis for an MBL-DTC model | One UUID-suffixed PDF |
| `main_mbl.py` | MBL spectrum and half-chain eigenvector entropy analysis | One UUID-suffixed PDF |
| `main_mbl_dynamics.py` | MBL return-rate dynamics | One UUID-suffixed PDF plus `print(...)` output |
| `main_mbl_propagator.py` | MBL propagator and operator-overlap analysis | No plot; logs matrices and overlaps |

Common CLI behavior:

- All flags are long-form options such as `--systemsize=6`.
- Hermitian workflows expose `--eigenBackend` to select among `qobj`, `numpy`,
  `scipy`, `torch`, and `jax`.
- `main_mbldtc.py` currently supports only `--eigenBackend=qobj` because it
  diagonalizes a general unitary Floquet operator rather than a Hermitian matrix.
- The parsers do not mark flags as `required=True`. If a flag is omitted, the
  script usually fails later with `None`-driven runtime errors instead of a
  clean `argparse` usage error.
- Output files are written to the repository root, not to `Plots/`.
- Several workflows draw random samples with `np.random.rand` or
  `scipy.stats.*.rvs` and do not seed the RNG, so results and many filenames are
  intentionally nondeterministic.

## QMBS CLI

Run QMBS / PXP eigenphase analysis with:

```bash
python3 main_qmbs.py --systemsize=6 --tduration=1.0 --Delta=0.1
```

To run the same Hermitian diagonalization through a dense backend instead of
QuTiP, pass `--eigenBackend`, for example:

```bash
python3 main_qmbs.py --systemsize=6 --tduration=1.0 --Delta=0.1 --eigenBackend=scipy
```

### Arguments

| Flag | Type | Meaning |
| --- | --- | --- |
| `--systemsize` | `int` | Number of spin-1/2 sites |
| `--tduration` | `float` | Time used to convert energies into unitary eigenphases through `exp(-i E t)` |
| `--Delta` | `float` | Detuning in units of the Rabi frequency `Omega` |
| `--eigenBackend` | `str` | Hermitian eigensolver backend: `qobj`, `numpy`, `scipy`, `torch`, or `jax` |

### Outputs

`main_qmbs.py` always writes two PDF files:

- `qmbs_sfim_N=%02d_tduration=%g_Vrr=%g_Omega=%g_Delta=%g.pdf`
- `qmbs_pxp_N=%02d_tduration=%g_Omega=%g_Delta=%g.pdf`

These names are stable for identical inputs because they do not include a UUID.

### Implementation Notes

- The script builds two Hamiltonians:
  - an Ising-like Hamiltonian with transverse drive, detuning, and nearest-neighbor blockade interaction
  - a constrained PXP Hamiltonian
- The implementation uses `Omega = 1.0` and `Vrr = 100.0 * Omega`.
- Both Hamiltonians are diagonalized with QuTiP.
- The code converts the energy spectra into unitary eigenvalues with
  `np.exp(-1j * eigenvalues * tduration)`.
- The helper `mbl_eigen.level_repulsion.calc_mean_adjacent_level_spacing_ratio`
  is applied both to eigenphases and to energy spectra.
- Two complex-plane eigenphase plots are saved.

## MBL-DTC CLI

Run the Floquet analysis for the many-body localized discrete time crystal with:

```bash
python3 main_mbldtc.py --systemsize=8 --thetaXPi=0.76
```

### Arguments

| Flag | Type | Meaning |
| --- | --- | --- |
| `--systemsize` | `int` | Number of spin-1/2 sites |
| `--thetaXPi` | `float` | Global transverse rotation angle in units of `pi`; the implementation uses `theta_x = pi * thetaXPi` |
| `--eigenBackend` | `str` | General eigensolver backend; the current implementation supports only `qobj` |

### Outputs

`main_mbldtc.py` writes one PDF file named:

- `mbldtc_N=%02d_thetax=%gpi_%s.pdf`

The UUID suffix makes repeated runs intentionally produce different filenames.

### Implementation Notes

- The script samples random on-site `z` rotation angles `phi_z` and random
  nearest-neighbor `zz` interaction angles `phi_zz` uniformly from `[0, pi)`.
- It builds a Floquet operator from:
  - a global `x` rotation
  - site-local `z` rotations
  - nearest-neighbor `zz` gates
- Each gate is expanded with `qutip.qip.operations.expand_operator`.
- The Floquet operator is diagonalized and its eigenphases are analyzed.
- The adjacent-level-spacing ratio is computed from the Floquet eigenphases.
- One complex-plane eigenvalue plot is saved.

## MBL CLI

Run the random-field MBL spectrum and eigenvector entropy analysis with:

```bash
python3 main_mbl.py --systemsize=12 --tduration=1.0 --jIntMean=1.0 --bFieldMean=1.0 --jIntStd=1.0 --bFieldStd=1.0 --anglePolarPiMin=0.0 --anglePolarPiMax=1.0 --eigenBackend=qobj
```

### Arguments

| Flag | Type | Meaning |
| --- | --- | --- |
| `--systemsize` | `int` | Number of spin-1/2 sites |
| `--tduration` | `float` | Time used to convert Hamiltonian eigenvalues into unitary eigenphases |
| `--jIntMean` | `float` | Mean of the normal distribution for nearest-neighbor `zz` couplings |
| `--bFieldMean` | `float` | Mean of the normal distribution for local field magnitudes |
| `--jIntStd` | `float` | Standard deviation of the normal distribution for nearest-neighbor `zz` couplings |
| `--bFieldStd` | `float` | Standard deviation of the normal distribution for local field magnitudes |
| `--anglePolarPiMin` | `float` | Lower bound of the polar-angle sampling interval, in units of `pi` |
| `--anglePolarPiMax` | `float` | Upper bound of the polar-angle sampling interval, in units of `pi` |
| `--eigenBackend` | `str` | Hermitian eigensolver backend: `qobj`, `numpy`, `scipy`, `torch`, or `jax` |

### Outputs

`main_mbl.py` writes one PDF file named:

- `mbl_sfim_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf`

The UUID suffix makes repeated runs intentionally produce different filenames.

### Implementation Notes

- The Hamiltonian construction is shared with `main_mbl_dynamics.py` and
  `main_mbl_propagator.py` through `mbl_eigen.mbl_model.build_mbl_model(...)`.
- The code samples:
  - `jInt_samples` from a normal distribution of shape `(systemsize - 1,)`
  - `bField_samples` from a normal distribution of shape `(systemsize,)`
  - `theta_samples` uniformly from `[anglePolarPiMin * pi, anglePolarPiMax * pi)`
- The Hamiltonian contains:
  - transverse `x` terms scaled by `bField * sin(theta)`
  - longitudinal `z` terms scaled by `bField * cos(theta)`
  - nearest-neighbor `zz` couplings scaled by `jInt`
- The script diagonalizes the Hamiltonian with QuTiP.
- It computes half-chain von Neumann entanglement entropies with
  `qutip.ptrace(...)` and `qutip.entropy_vn(...)`.
- It computes spacing ratios for both the Hamiltonian spectrum and the derived
  unitary eigenphases.

## MBL Dynamics CLI

Run the return-rate dynamics analysis with:

```bash
python3 main_mbl_dynamics.py --systemsize=12 --tduration=1.0 --jIntMean=1.0 --bFieldMean=1.0 --jIntStd=1.0 --bFieldStd=1.0 --anglePolarPiMin=0.0 --anglePolarPiMax=1.0 --eigenBackend=qobj
```

### Arguments

`main_mbl_dynamics.py` accepts the same flags as `main_mbl.py`, including
`--eigenBackend`.

Important behavior note:

- `--tduration` is accepted for CLI compatibility but the current implementation
  does not use it. The time grid is hard-coded inside `mbl_eigen.mbl_app.run_mbl_dynamics(...)`.

### Outputs

`main_mbl_dynamics.py` writes one PDF file named:

- `mbl_sfim_dynamics_N=%02d_anglePolarPiMin=%g_anglePolarPiMax=%g_jIntMean=%g_jIntStd=%g_bFieldMean=%g_bFieldStd=%g_%s.pdf`

It also prints the following intermediate values to standard output:

- `bField_samples`
- `theta_samples`
- `jInt_samples`
- the initial ket `qutip.ket('1' * systemsize)`
- the eigenbasis overlap amplitudes
- the time-dependent return amplitudes

### Implementation Notes

- The Hamiltonian is generated through the shared `mbl_eigen.mbl_model`
  pipeline.
- The initial state is the product ket `|11...1>` represented by
  `qutip.ket('1' * systemsize)`.
- The script expands the initial state in the Hamiltonian eigenbasis with
  `v.overlap(ket_initial)`.
- The return amplitude is evaluated on the fixed grid
  `np.arange(0.0, 10.0625, 0.0625)`.
- The plotted quantity is `-log(|A(t)|^2) / systemsize`.

## MBL Propagator CLI

Run the propagator / operator-overlap analysis with:

```bash
python3 main_mbl_propagator.py --systemsize=12 --tduration=1.0 --jIntMean=1.0 --bFieldMean=1.0 --jIntStd=1.0 --bFieldStd=1.0 --anglePolarPiMin=0.0 --anglePolarPiMax=1.0 --eigenBackend=numpy
```

### Arguments

`main_mbl_propagator.py` accepts the same flags as `main_mbl.py`.

Default backend note:

- `main_mbl_propagator.py` defaults to `--eigenBackend=numpy` to preserve the
  previous dense-matrix diagonalization path.

Important behavior note:

- `--tduration` is accepted for CLI compatibility but the current implementation
  does not use it. The time grid is hard-coded inside
  `mbl_eigen.mbl_app.run_mbl_propagator(...)`.

### Outputs

- No plot is saved.
- The script emits extensive `logging.info(...)` output including disorder
  samples, basis-change matrices, operator arrays, time-evolved operators, and
  the final overlap matrix.

### Implementation Notes

- The shared MBL Hamiltonian is converted to a dense array with
  `hamiltonian.full()`.
- The Hamiltonian is diagonalized with `numpy.linalg.eigh(...)`.
- Local `sigma_x`, `sigma_y`, and `sigma_z` operators are expanded to every site,
  normalized by `sqrt(2**systemsize)`, and rotated into the Hamiltonian
  eigenbasis.
- The code evolves those operators on the fixed grid
  `np.arange(0.0, 1.0625, 0.0625)`.
- It then computes the overlap matrix between the initial and time-evolved
  operators for all `x`, `y`, and `z` channels.

## Python API

The reusable implementation lives in `mbl_eigen/`. The most relevant public
entrypoints are listed below.

### CLI Builders

- `mbl_eigen.cli.build_qmbs_parser()`
- `mbl_eigen.cli.build_mbldtc_parser()`
- `mbl_eigen.cli.build_mbl_parser()`

These functions return `argparse.ArgumentParser` instances matching the root
shim scripts.

### Application Runners

- `mbl_eigen.qmbs_app.run_qmbs(args)`
- `mbl_eigen.mbldtc_app.run_mbldtc(args)`
- `mbl_eigen.mbl_app.run_mbl(args)`
- `mbl_eigen.mbl_app.run_mbl_dynamics(args)`
- `mbl_eigen.mbl_app.run_mbl_propagator(args)`

Each runner expects an object with the same attributes produced by the matching
parser.

### MBL Model API

- `mbl_eigen.mbl_model.spin_operators()`
- `mbl_eigen.mbl_model.sample_mbl_disorder(...)`
- `mbl_eigen.mbl_model.build_mbl_hamiltonian(...)`
- `mbl_eigen.mbl_model.build_mbl_model(...)`
- `mbl_eigen.eigensolver.solve_hermitian_eigenproblem(...)`
- `mbl_eigen.eigensolver.solve_general_eigenproblem(...)`

`build_mbl_model(...)` returns an `MBLModel` dataclass with:

- `hamiltonian`
- `sigma0`
- `sigmax`
- `sigmay`
- `sigmaz`
- `jInt_samples`
- `bField_samples`
- `theta_samples`

Example:

```python
from mbl_eigen.mbl_model import build_mbl_model

model = build_mbl_model(
    systemsize=4,
    jIntMean=1.0,
    jIntStd=0.1,
    bFieldMean=1.0,
    bFieldStd=0.1,
    anglePolarPiMin=0.0,
    anglePolarPiMax=1.0,
)

eigenvalues, eigenvectors = model.hamiltonian.eigenstates()
print(model.jInt_samples)
print(eigenvalues)
```

### Utility API

- `mbl_eigen.level_repulsion.calc_mean_adjacent_level_spacing_ratio(...)`
- `mbl_eigen.reflection.ReflectionAboutCenter`
- `mbl_eigen.reflection.reflection_about_center(...)`
- `mbl_eigen.symmetry.Symmetry`
- `mbl_eigen.symmetry.Involution`

The root `level_repulsion.py`, `reflection.py`, and `symmetry.py` files remain
available as wrappers if older code imports them directly.

## Implementation Details

The current execution flow is:

1. A root shim script such as `main_mbl.py` builds a parser from
   `mbl_eigen.cli`.
2. The shim parses CLI arguments and forwards the resulting namespace to a
   package-level runner such as `mbl_eigen.mbl_app.run_mbl(...)`.
3. Shared physics/model construction is handled by `mbl_eigen.mbl_model`.
4. Output filenames are generated centrally in `mbl_eigen.output_names`.
5. Level-spacing statistics are computed by
   `mbl_eigen.level_repulsion.calc_mean_adjacent_level_spacing_ratio(...)`.

Important implementation-specific details:

- The repository uses QuTiP for operator algebra, tensor products,
  diagonalization, and partial traces.
- `qutip-qip` is required because the code expands local operators and gates
  with `qutip.qip.operations.expand_operator`.
- Hermitian eigenvalue calculations now pass through `mbl_eigen.eigensolver`,
  which supports `Qobj.eigenstates()`, `numpy.linalg.eigh`,
  `scipy.linalg.eigh`, `torch.linalg.eigh`, and `jax.numpy.linalg.eigh`.
- `torch` and `jax` are optional extras and are imported lazily only when their
  backends are selected.
- The MBL workflows share one Hamiltonian builder instead of duplicating the
  random-field model setup in each script.
- The output naming logic is centralized so the shim scripts preserve the same
  filename conventions as before the refactor.
- `main_mbl_dynamics.py` and `main_mbl_propagator.py` still expose the same CLI
  surface as `main_mbl.py` even though `--tduration` is not currently consumed
  by those implementations.

## Verification

There is no automated test suite in this repository.

A fast syntax check after edits is:

```bash
python3 -m compileall mbl_eigen main_*.py level_repulsion.py reflection.py symmetry.py
```

When dependencies are installed, the most useful runtime verification is a
small-`systemsize` run of the specific script you changed.
