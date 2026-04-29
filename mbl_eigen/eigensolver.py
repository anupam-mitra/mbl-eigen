from dataclasses import dataclass

import numpy as np
import qutip
import scipy.linalg


HERMITIAN_EIGEN_BACKENDS = ("qobj", "numpy", "scipy", "torch", "jax")
GENERAL_EIGEN_BACKENDS = ("qobj",)
EIGENSOLVER_DEVICE_CHOICES = ("auto", "cpu", "gpu", "cuda", "mps")


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors_array: np.ndarray | None
    backend: str
    dims: object | None = None
    device: str = "cpu"

    def as_qobj_kets(self):
        if self.eigenvectors_array is None:
            raise ValueError("eigenvectors were not requested")

        if self.dims is None:
            raise ValueError("operator dims are required to rebuild QuTiP kets")

        ket_dims = [self.dims[0], [1]]
        return [
            qutip.Qobj(self.eigenvectors_array[:, ix], dims=ket_dims)
            for ix in range(self.eigenvectors_array.shape[1])
        ]

    def as_basis_qobj(self):
        if self.eigenvectors_array is None:
            raise ValueError("eigenvectors were not requested")

        if self.dims is None:
            raise ValueError("operator dims are required to rebuild a basis-change Qobj")

        return qutip.Qobj(self.eigenvectors_array, dims=self.dims)


def solve_hermitian_eigenproblem(
        operator,
        *,
        backend="qobj",
        device="auto",
        return_eigenvectors=True):
    if backend not in HERMITIAN_EIGEN_BACKENDS:
        raise ValueError(
            "unsupported Hermitian eigen backend %r; expected one of %s"
            % (backend, HERMITIAN_EIGEN_BACKENDS)
        )

    if device not in EIGENSOLVER_DEVICE_CHOICES:
        raise ValueError(
            "unsupported eigensolver device %r; expected one of %s"
            % (device, EIGENSOLVER_DEVICE_CHOICES)
        )

    operator_qobj = _as_qobj_operator(operator)
    dims = operator_qobj.dims

    if backend == "qobj":
        actual_device = _resolve_cpu_only_device(device, backend)
        eigenvalues, eigenvectors = operator_qobj.eigenstates()
        eigenvectors_array = None

        if return_eigenvectors:
            eigenvectors_array = np.column_stack([
                np.asarray(v.full(), dtype=np.complex128).reshape(-1)
                for v in eigenvectors
            ])

        return _build_hermitian_result(
            eigenvalues=eigenvalues,
            eigenvectors_array=eigenvectors_array,
            backend=backend,
            dims=dims,
            device=actual_device,
        )

    operator_ndarray = np.asarray(operator_qobj.full(), dtype=np.complex128)

    if backend == "numpy":
        actual_device = _resolve_cpu_only_device(device, backend)
        eigenvalues, eigenvectors_array = np.linalg.eigh(operator_ndarray)
    elif backend == "scipy":
        actual_device = _resolve_cpu_only_device(device, backend)
        eigenvalues, eigenvectors_array = scipy.linalg.eigh(operator_ndarray)
    elif backend == "torch":
        eigenvalues, eigenvectors_array, actual_device = _torch_eigh(
            operator_ndarray,
            device=device,
        )
    else:
        eigenvalues, eigenvectors_array, actual_device = _jax_eigh(
            operator_ndarray,
            device=device,
        )

    if not return_eigenvectors:
        eigenvectors_array = None

    return _build_hermitian_result(
        eigenvalues=eigenvalues,
        eigenvectors_array=eigenvectors_array,
        backend=backend,
        dims=dims,
        device=actual_device,
    )


def solve_general_eigenproblem(
        operator,
        *,
        backend="qobj",
        device="auto",
        return_eigenvectors=True):
    if backend not in GENERAL_EIGEN_BACKENDS:
        raise ValueError(
            "unsupported general eigen backend %r; expected one of %s"
            % (backend, GENERAL_EIGEN_BACKENDS)
        )

    if device not in EIGENSOLVER_DEVICE_CHOICES:
        raise ValueError(
            "unsupported eigensolver device %r; expected one of %s"
            % (device, EIGENSOLVER_DEVICE_CHOICES)
        )

    operator_qobj = _as_qobj_operator(operator)
    actual_device = _resolve_cpu_only_device(device, backend)
    eigenvalues, eigenvectors = operator_qobj.eigenstates()
    eigenvectors_array = None

    if return_eigenvectors:
        eigenvectors_array = np.column_stack([
            np.asarray(v.full(), dtype=np.complex128).reshape(-1)
            for v in eigenvectors
        ])

    return EigenResult(
        eigenvalues=np.asarray(eigenvalues, dtype=np.complex128),
        eigenvectors_array=eigenvectors_array,
        backend=backend,
        dims=operator_qobj.dims,
        device=actual_device,
    )


def _as_qobj_operator(operator):
    if isinstance(operator, qutip.Qobj):
        return operator

    return qutip.Qobj(np.asarray(operator, dtype=np.complex128))


def _build_hermitian_result(eigenvalues, eigenvectors_array, backend, dims, device):
    eigenvalues = np.real_if_close(np.asarray(eigenvalues, dtype=np.complex128))
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]

    if eigenvectors_array is not None:
        eigenvectors_array = np.asarray(eigenvectors_array, dtype=np.complex128)[:, order]

    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors_array=eigenvectors_array,
        backend=backend,
        dims=dims,
        device=device,
    )


def _torch_eigh(operator_ndarray, device):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch backend requires installing the optional 'torch' extra"
        ) from exc

    actual_device = _resolve_torch_device(torch, device)
    operator_tensor = torch.tensor(
        operator_ndarray,
        dtype=torch.complex128,
        device=torch.device(actual_device),
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(operator_tensor)
    _torch_synchronize(torch, actual_device)

    return np.asarray(eigenvalues.cpu(), dtype=np.float64), np.asarray(
        eigenvectors.cpu(), dtype=np.complex128), actual_device


def _jax_eigh(operator_ndarray, device):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "jax backend requires installing the optional 'jax' extra"
        ) from exc

    jax.config.update("jax_enable_x64", True)
    actual_device = _resolve_jax_device(jax, device)
    operator_array = jax.device_put(
        jnp.asarray(operator_ndarray, dtype=jnp.complex128),
        device=actual_device,
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(operator_array)
    eigenvalues.block_until_ready()
    eigenvectors.block_until_ready()

    return np.asarray(eigenvalues, dtype=np.float64), np.asarray(
        eigenvectors, dtype=np.complex128), _describe_jax_device(actual_device)


def _resolve_cpu_only_device(device, backend):
    if device in ("auto", "cpu"):
        return "cpu"

    raise ValueError(
        "%s backend supports only CPU execution, got device=%r"
        % (backend, device)
    )


def _resolve_torch_device(torch, device):
    has_cuda = torch.cuda.is_available()
    has_mps = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if device == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if device == "gpu":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        raise ValueError("torch backend requested a GPU device but no CUDA or MPS device is available")

    if device == "cuda":
        if has_cuda:
            return "cuda"
        raise ValueError("torch backend requested CUDA but no CUDA device is available")

    if device == "mps":
        if has_mps:
            return "mps"
        raise ValueError("torch backend requested MPS but no MPS device is available")

    return "cpu"


def _torch_synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _resolve_jax_device(jax, device):
    all_devices = list(jax.devices())
    cpu_devices = [d for d in all_devices if d.platform.lower() == "cpu"]
    accelerator_devices = [d for d in all_devices if d.platform.lower() != "cpu"]

    if device == "auto":
        if accelerator_devices:
            return accelerator_devices[0]
        if cpu_devices:
            return cpu_devices[0]
        raise ValueError("jax backend did not report any devices")

    if device == "cpu":
        if cpu_devices:
            return cpu_devices[0]
        raise ValueError("jax backend requested CPU but no CPU device is available")

    if device == "gpu":
        if accelerator_devices:
            return accelerator_devices[0]
        raise ValueError("jax backend requested a GPU device but no accelerator device is available")

    requested_platforms = {
        "cuda": {"gpu", "cuda", "rocm"},
        "mps": {"metal", "mps", "gpu"},
    }
    matching_devices = [
        d for d in accelerator_devices
        if d.platform.lower() in requested_platforms[device]
    ]

    if matching_devices:
        return matching_devices[0]

    raise ValueError(
        "jax backend requested %s but no matching accelerator device is available"
        % device
    )


def _describe_jax_device(device):
    platform = getattr(device, "platform", str(device))
    device_id = getattr(device, "id", None)
    if device_id is None:
        return str(platform)
    return "%s:%s" % (platform, device_id)


__all__ = [
    "EigenResult",
    "EIGENSOLVER_DEVICE_CHOICES",
    "GENERAL_EIGEN_BACKENDS",
    "HERMITIAN_EIGEN_BACKENDS",
    "solve_general_eigenproblem",
    "solve_hermitian_eigenproblem",
]
