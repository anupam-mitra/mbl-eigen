# Benchmarking Eigensolver Backends

This directory contains a lightweight benchmark harness for the Hermitian
eigensolver backends exposed by `mbl_eigen.eigensolver`.

The benchmark currently focuses on Hermitian workloads because the general
eigendecomposition path only supports `qobj` today.

## What It Benchmarks

- Realistic random-field MBL Hamiltonians built from the same operator
  construction used by the main analysis scripts.
- Synthetic dense complex Hermitian matrices for backend-to-backend kernel
  comparisons.

For each case, the harness measures:

- `qobj`
- `numpy`
- `scipy`
- `torch` when installed
- `jax` when installed

It reports:

- timing statistics
- the requested and actual execution device
- eigenvalue differences relative to a baseline solver
- eigenvector orthogonality residuals
- reconstruction residuals when eigenvectors are requested
- skip reasons for unavailable backends or unsupported device requests

## CPU Smoke Run

```bash
".venv/bin/python" benchmarks/bench_eigensolver.py --suite smoke --workload both --backends qobj,numpy,scipy --devices cpu --return-eigenvectors both --warmup 1 --repeat 3
```

## CPU And GPU Requests

When optional backends are installed, request CPU and GPU configurations with:

```bash
".venv/bin/python" benchmarks/bench_eigensolver.py --suite default --workload both --backends qobj,numpy,scipy,torch,jax --devices cpu,gpu --return-eigenvectors both
```

More explicit device selections are also supported:

- `cpu`
- `gpu`
- `cuda`
- `mps`
- `auto`

Notes:

- `qobj`, `numpy`, and `scipy` are CPU-only. GPU requests for those backends are
  reported as skips.
- `torch` may run on CPU, CUDA, or Apple MPS depending on installation and
  hardware.
- `jax` uses the best matching available device for `gpu`, `cuda`, or `mps`
  requests and reports the actual JAX device used.

Apple M1 note from the current verified setup:

- `torch` CPU benchmarking works.
- `torch` GPU requests on MPS currently skip because `torch.linalg.eigh` is not
  implemented on MPS, and the solver currently uses `float64` / `complex128`
  dtypes that MPS does not accept.
- `jax` CPU benchmarking works when launched with `JAX_PLATFORMS=cpu`.
- `jax-metal` initializes a Metal device, but the current GPU benchmark path
  still fails on this machine with `UNIMPLEMENTED: default_memory_space is not supported`.

## Writing Results To JSON

```bash
".venv/bin/python" benchmarks/bench_eigensolver.py --suite default --workload both --backends qobj,numpy,scipy --devices cpu --output benchmarks/results/default_cpu.json
```

The JSON output includes environment metadata such as CPU model, GPU summary,
and installed package versions.

## Suggested Interpretation

- Use MBL workloads to compare the backends in the same shape and data flow that
  the production analysis scripts use.
- Use synthetic workloads to compare eigensolver kernels more directly.
- Compare `return_eigenvectors=false` and `return_eigenvectors=true` separately;
  those paths can have different performance characteristics.
