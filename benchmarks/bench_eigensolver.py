#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qutip


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbl_eigen.eigensolver import EIGENSOLVER_DEVICE_CHOICES
from mbl_eigen.eigensolver import HERMITIAN_EIGEN_BACKENDS
from mbl_eigen.eigensolver import solve_hermitian_eigenproblem
from mbl_eigen.mbl_model import build_mbl_hamiltonian
from mbl_eigen.mbl_model import spin_operators


SUITE_PRESETS = {
    "smoke": {
        "systemsizes": [6],
        "dims": [64],
    },
    "default": {
        "systemsizes": [6, 8, 10],
        "dims": [64, 256, 1024],
    },
    "extended": {
        "systemsizes": [6, 8, 10, 11],
        "dims": [64, 256, 1024, 2048],
    },
}


@dataclass
class BenchmarkCase:
    workload: str
    label: str
    operator: qutip.Qobj
    dimension: int
    systemsize: int | None


@dataclass
class BenchmarkRecord:
    workload: str
    label: str
    backend: str
    requested_device: str
    actual_device: str | None
    dimension: int
    systemsize: int | None
    return_eigenvectors: bool
    warmup: int
    repeat: int
    median_ms: float | None
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    stdev_ms: float | None
    times_ms: list[float]
    max_abs_eigenvalue_diff: float | None
    orthogonality_residual: float | None
    reconstruction_residual: float | None
    skipped: bool
    skip_reason: str | None


def parse_args():
    parser = argparse.ArgumentParser(
        prog="benchmarks/bench_eigensolver.py",
        description="Benchmark Hermitian eigensolver backends on realistic MBL and synthetic matrices.",
    )
    parser.add_argument("--suite", choices=tuple(SUITE_PRESETS), default="default")
    parser.add_argument("--workload", choices=("mbl", "synthetic", "both"), default="both")
    parser.add_argument("--backends", default=",".join(HERMITIAN_EIGEN_BACKENDS))
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--systemsizes", default="")
    parser.add_argument("--dims", default="")
    parser.add_argument("--return-eigenvectors", choices=("false", "true", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    backends = _parse_backends(args.backends)
    devices = _parse_devices(args.devices)
    return_eigenvectors_modes = _parse_return_eigenvectors_modes(args.return_eigenvectors)
    systemsizes = _parse_optional_int_list(args.systemsizes) or SUITE_PRESETS[args.suite]["systemsizes"]
    dims = _parse_optional_int_list(args.dims) or SUITE_PRESETS[args.suite]["dims"]
    cases = _build_cases(
        workload=args.workload,
        systemsizes=systemsizes,
        dims=dims,
        seed=args.seed,
    )

    records = []
    for case in cases:
        for return_eigenvectors in return_eigenvectors_modes:
            baseline = _compute_baseline(case, return_eigenvectors)
            for backend in backends:
                for device in devices:
                    record = _benchmark_case(
                        case=case,
                        backend=backend,
                        device=device,
                        return_eigenvectors=return_eigenvectors,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        baseline=baseline,
                    )
                    records.append(record)
                    _print_record(record)

    payload = {
        "environment": _collect_environment_metadata(),
        "arguments": {
            "suite": args.suite,
            "workload": args.workload,
            "backends": backends,
            "devices": devices,
            "systemsizes": systemsizes,
            "dims": dims,
            "return_eigenvectors": return_eigenvectors_modes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
        },
        "results": [asdict(record) for record in records],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_backends(raw_value):
    backends = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [backend for backend in backends if backend not in HERMITIAN_EIGEN_BACKENDS]
    if invalid:
        raise ValueError("unsupported backends: %s" % ", ".join(invalid))
    return backends


def _parse_devices(raw_value):
    devices = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [device for device in devices if device not in EIGENSOLVER_DEVICE_CHOICES]
    if invalid:
        raise ValueError("unsupported devices: %s" % ", ".join(invalid))
    return devices


def _parse_optional_int_list(raw_value):
    if not raw_value:
        return []
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def _parse_return_eigenvectors_modes(raw_value):
    if raw_value == "both":
        return [False, True]
    return [raw_value == "true"]


def _build_cases(workload, systemsizes, dims, seed):
    cases = []

    if workload in ("mbl", "both"):
        for systemsize in systemsizes:
            operator = _build_mbl_operator(systemsize=systemsize, seed=seed + systemsize)
            cases.append(BenchmarkCase(
                workload="mbl",
                label="mbl_N=%d" % systemsize,
                operator=operator,
                dimension=operator.shape[0],
                systemsize=systemsize,
            ))

    if workload in ("synthetic", "both"):
        for dimension in dims:
            operator = _build_synthetic_operator(dimension=dimension, seed=seed + dimension)
            cases.append(BenchmarkCase(
                workload="synthetic",
                label="synthetic_dim=%d" % dimension,
                operator=operator,
                dimension=dimension,
                systemsize=None,
            ))

    return cases


def _build_mbl_operator(systemsize, seed):
    rng = np.random.default_rng(seed)
    sigma0, sigmax, sigmay, sigmaz = spin_operators()
    del sigma0, sigmay

    jInt_samples = rng.normal(loc=1.0, scale=0.1, size=systemsize - 1)
    bField_samples = rng.normal(loc=1.0, scale=0.1, size=systemsize)
    theta_samples = rng.uniform(low=0.0, high=np.pi, size=systemsize)

    return build_mbl_hamiltonian(
        systemsize=systemsize,
        jInt_samples=jInt_samples,
        bField_samples=bField_samples,
        theta_samples=theta_samples,
        sigma0=qutip.qeye(2),
        sigmax=sigmax,
        sigmaz=sigmaz,
    )


def _build_synthetic_operator(dimension, seed):
    rng = np.random.default_rng(seed)
    real_part = rng.normal(size=(dimension, dimension))
    imag_part = rng.normal(size=(dimension, dimension))
    dense = real_part + 1j * imag_part
    hermitian = (dense + dense.conj().T) / (2.0 * np.sqrt(dimension))
    return qutip.Qobj(hermitian)


def _compute_baseline(case, return_eigenvectors):
    for backend in ("scipy", "numpy", "qobj"):
        try:
            return solve_hermitian_eigenproblem(
                case.operator,
                backend=backend,
                device="cpu",
                return_eigenvectors=return_eigenvectors,
            )
        except Exception:
            continue
    raise RuntimeError("failed to compute a baseline eigensolution for %s" % case.label)


def _benchmark_case(case, backend, device, return_eigenvectors, warmup, repeat, baseline):
    try:
        for _ in range(warmup):
            solve_hermitian_eigenproblem(
                case.operator,
                backend=backend,
                device=device,
                return_eigenvectors=return_eigenvectors,
            )

        times_ms = []
        last_result = None
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            last_result = solve_hermitian_eigenproblem(
                case.operator,
                backend=backend,
                device=device,
                return_eigenvectors=return_eigenvectors,
            )
            end_ns = time.perf_counter_ns()
            times_ms.append((end_ns - start_ns) / 1e6)

        return BenchmarkRecord(
            workload=case.workload,
            label=case.label,
            backend=backend,
            requested_device=device,
            actual_device=last_result.device,
            dimension=case.dimension,
            systemsize=case.systemsize,
            return_eigenvectors=return_eigenvectors,
            warmup=warmup,
            repeat=repeat,
            median_ms=statistics.median(times_ms),
            mean_ms=statistics.fmean(times_ms),
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            times_ms=times_ms,
            max_abs_eigenvalue_diff=_max_abs_eigenvalue_diff(baseline.eigenvalues, last_result.eigenvalues),
            orthogonality_residual=_orthogonality_residual(last_result.eigenvectors_array),
            reconstruction_residual=_reconstruction_residual(
                case.operator,
                last_result.eigenvalues,
                last_result.eigenvectors_array,
            ),
            skipped=False,
            skip_reason=None,
        )
    except Exception as exc:
        return BenchmarkRecord(
            workload=case.workload,
            label=case.label,
            backend=backend,
            requested_device=device,
            actual_device=None,
            dimension=case.dimension,
            systemsize=case.systemsize,
            return_eigenvectors=return_eigenvectors,
            warmup=warmup,
            repeat=repeat,
            median_ms=None,
            mean_ms=None,
            min_ms=None,
            max_ms=None,
            stdev_ms=None,
            times_ms=[],
            max_abs_eigenvalue_diff=None,
            orthogonality_residual=None,
            reconstruction_residual=None,
            skipped=True,
            skip_reason="%s: %s" % (type(exc).__name__, exc),
        )


def _max_abs_eigenvalue_diff(reference, candidate):
    return float(np.max(np.abs(reference - candidate)))


def _orthogonality_residual(eigenvectors_array):
    if eigenvectors_array is None:
        return None

    identity = np.eye(eigenvectors_array.shape[1], dtype=np.complex128)
    gram = eigenvectors_array.conj().T @ eigenvectors_array
    return float(np.linalg.norm(gram - identity, ord=np.inf))


def _reconstruction_residual(operator, eigenvalues, eigenvectors_array):
    if eigenvectors_array is None:
        return None

    dense = np.asarray(operator.full(), dtype=np.complex128)
    reconstructed = eigenvectors_array @ np.diag(eigenvalues) @ eigenvectors_array.conj().T
    denominator = max(float(np.linalg.norm(dense, ord=np.inf)), 1.0)
    return float(np.linalg.norm(dense - reconstructed, ord=np.inf) / denominator)


def _print_record(record):
    if record.skipped:
        print(
            "SKIP workload=%s label=%s backend=%s requested_device=%s reason=%s"
            % (
                record.workload,
                record.label,
                record.backend,
                record.requested_device,
                record.skip_reason,
            )
        )
        return

    print(
        "OK workload=%s label=%s backend=%s requested_device=%s actual_device=%s return_eigenvectors=%s median_ms=%.3f max_abs_eigenvalue_diff=%.3e"
        % (
            record.workload,
            record.label,
            record.backend,
            record.requested_device,
            record.actual_device,
            record.return_eigenvectors,
            record.median_ms,
            record.max_abs_eigenvalue_diff,
        )
    )


def _collect_environment_metadata():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": _run_command("sysctl -n machdep.cpu.brand_string"),
        "physical_cpu": _run_command("sysctl -n hw.physicalcpu"),
        "logical_cpu": _run_command("sysctl -n hw.logicalcpu"),
        "gpu": _gpu_summary(),
        "package_versions": _package_versions(),
    }


def _run_command(command):
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _gpu_summary():
    summary = _run_command("system_profiler SPDisplaysDataType")
    if summary is None:
        return None
    return summary


def _package_versions():
    packages = ["numpy", "scipy", "qutip", "torch", "jax", "jaxlib"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == "__main__":
    main()
