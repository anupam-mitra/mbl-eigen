import logging
from dataclasses import dataclass

import numpy as np
import qutip
import scipy.linalg


_logger = logging.getLogger(__name__)

_JAX_X64_INITIALIZED = False

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
    _validate_backend_and_device(backend, device, HERMITIAN_EIGEN_BACKENDS, "Hermitian")

    operator_qobj = _as_qobj_operator(operator)
    dims = operator_qobj.dims

    operator_ndarray = np.asarray(operator_qobj.full(), dtype=np.complex128)
    if not np.allclose(operator_ndarray, operator_ndarray.conj().T):
        _logger.warning(
            "non-Hermitian operator symmetrized for %s eigen backend", backend
        )
        operator_qobj = (operator_qobj + operator_qobj.dag()) / 2
        operator_ndarray = np.asarray(operator_qobj.full(), dtype=np.complex128)

    if backend == "qobj":
        actual_device = _resolve_cpu_only_device(device, backend)
        if return_eigenvectors:
            eigenvalues, eigenvectors = operator_qobj.eigenstates()
            eigenvectors_array = np.column_stack([
                np.asarray(v.full(), dtype=np.complex128).reshape(-1)
                for v in eigenvectors
            ])
        else:
            eigenvalues = operator_qobj.eigenenergies()
            eigenvectors_array = None

        return _build_hermitian_result(
            eigenvalues=eigenvalues,
            eigenvectors_array=eigenvectors_array,
            backend=backend,
            dims=dims,
            device=actual_device,
        )

    if backend == "numpy":
        actual_device = _resolve_cpu_only_device(device, backend)
        if return_eigenvectors:
            eigenvalues, eigenvectors_array = _eigh_with_context(
                np.linalg.eigh, operator_ndarray, backend
            )
        else:
            eigenvalues = _eigh_with_context(
                np.linalg.eigvalsh, operator_ndarray, backend
            )
            eigenvectors_array = None
    elif backend == "scipy":
        actual_device = _resolve_cpu_only_device(device, backend)
        if return_eigenvectors:
            eigenvalues, eigenvectors_array = _eigh_with_context(
                scipy.linalg.eigh, operator_ndarray, backend
            )
        else:
            eigenvalues = _eigh_with_context(
                scipy.linalg.eigvalsh, operator_ndarray, backend
            )
            eigenvectors_array = None
    elif backend == "torch":
        eigenvalues, eigenvectors_array, actual_device = _torch_eigh(
            operator_ndarray,
            device=device,
            return_eigenvectors=return_eigenvectors,
        )
    else:
        eigenvalues, eigenvectors_array, actual_device = _jax_eigh(
            operator_ndarray,
            device=device,
            return_eigenvectors=return_eigenvectors,
        )

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
    _validate_backend_and_device(backend, device, GENERAL_EIGEN_BACKENDS, "general")

    operator_qobj = _as_qobj_operator(operator)
    actual_device = _resolve_cpu_only_device(device, backend)
    if return_eigenvectors:
        eigenvalues, eigenvectors = operator_qobj.eigenstates()
        eigenvectors_array = np.column_stack([
            np.asarray(v.full(), dtype=np.complex128).reshape(-1)
            for v in eigenvectors
        ])
    else:
        eigenvalues = operator_qobj.eigenenergies()
        eigenvectors_array = None

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


def _validate_backend_and_device(backend, device, allowed_backends, label):
    if backend not in allowed_backends:
        raise ValueError(
            "unsupported %s eigen backend %r; expected one of %s"
            % (label, backend, allowed_backends)
        )

    if device not in EIGENSOLVER_DEVICE_CHOICES:
        raise ValueError(
            "unsupported eigensolver device %r; expected one of %s"
            % (device, EIGENSOLVER_DEVICE_CHOICES)
        )


def _eigh_with_context(solver, operator, backend):
    try:
        return solver(operator)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        raise np.linalg.LinAlgError(
            "eigendecomposition failed: backend=%s dtype=%s dimension=%s (%s)"
            % (backend, operator.dtype, operator.shape, exc)
        ) from exc


def _build_hermitian_result(eigenvalues, eigenvectors_array, backend, dims, device):
    eigenvalues = np.real_if_close(np.asarray(eigenvalues, dtype=np.complex128))
    if np.iscomplexobj(eigenvalues) and np.abs(eigenvalues.imag).max() > 0:
        raise ValueError(
            "eigenvalues have non-negligible imaginary parts (max |imag|=%r) "
            "for backend=%s; refusing to discard them"
            % (np.abs(eigenvalues.imag).max(), backend)
        )
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


def _torch_eigh(operator_ndarray, device, return_eigenvectors):
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
    if return_eigenvectors:
        eigenvalues, eigenvectors = _eigh_with_context(
            torch.linalg.eigh, operator_tensor, "torch"
        )
    else:
        eigenvalues = _eigh_with_context(
            torch.linalg.eigvalsh, operator_tensor, "torch"
        )
        eigenvectors = None
    _torch_synchronize(torch, actual_device)

    eigenvectors_array = None
    if eigenvectors is not None:
        eigenvectors_array = np.asarray(eigenvectors.cpu(), dtype=np.complex128)

    return np.asarray(eigenvalues.cpu(), dtype=np.float64), eigenvectors_array, actual_device


def _jax_eigh(operator_ndarray, device, return_eigenvectors):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "jax backend requires installing the optional 'jax' extra"
        ) from exc

    global _JAX_X64_INITIALIZED
    if not _JAX_X64_INITIALIZED:
        jax.config.update("jax_enable_x64", True)
        _JAX_X64_INITIALIZED = True
    actual_device = _resolve_jax_device(jax, device)
    operator_array = jax.device_put(
        jnp.asarray(operator_ndarray, dtype=jnp.complex128),
        device=actual_device,
    )
    if return_eigenvectors:
        eigenvalues, eigenvectors = _eigh_with_context(
            jnp.linalg.eigh, operator_array, "jax"
        )
    else:
        eigenvalues = _eigh_with_context(
            jnp.linalg.eigvalsh, operator_array, "jax"
        )
        eigenvectors = None
    eigenvalues.block_until_ready()
    if eigenvectors is not None:
        eigenvectors.block_until_ready()

    eigenvectors_array = None
    if eigenvectors is not None:
        eigenvectors_array = np.asarray(eigenvectors, dtype=np.complex128)

    return np.asarray(eigenvalues, dtype=np.float64), eigenvectors_array, _describe_jax_device(actual_device)


def _resolve_cpu_only_device(device, backend):
    if device in ("auto", "cpu"):
        return "cpu"

    raise ValueError(
        "%s backend supports only CPU execution, got device=%r"
        % (backend, device)
    )


def _resolve_torch_device(torch, device):
    has_cuda = torch.cuda.is_available()

    if device == "auto":
        if has_cuda:
            return "cuda"
        return "cpu"

    if device == "gpu":
        if has_cuda:
            return "cuda"
        raise ValueError("torch backend supports only CUDA GPU execution")

    if device == "cuda":
        if has_cuda:
            return "cuda"
        raise ValueError("torch backend requested CUDA but no CUDA device is available")

    if device == "mps":
        raise ValueError("torch backend does not support MPS eigendecomposition")

    return "cpu"


def _torch_synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


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
        "mps": {"metal", "mps"},
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
