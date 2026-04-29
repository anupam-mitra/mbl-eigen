from dataclasses import dataclass

import numpy as np
import qutip
import scipy.linalg


HERMITIAN_EIGEN_BACKENDS = ("qobj", "numpy", "scipy", "torch", "jax")
GENERAL_EIGEN_BACKENDS = ("qobj",)


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors_array: np.ndarray | None
    backend: str
    dims: object | None = None

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
        return_eigenvectors=True):
    if backend not in HERMITIAN_EIGEN_BACKENDS:
        raise ValueError(
            "unsupported Hermitian eigen backend %r; expected one of %s"
            % (backend, HERMITIAN_EIGEN_BACKENDS)
        )

    operator_qobj = _as_qobj_operator(operator)
    dims = operator_qobj.dims

    if backend == "qobj":
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
        )

    operator_ndarray = np.asarray(operator_qobj.full(), dtype=np.complex128)

    if backend == "numpy":
        eigenvalues, eigenvectors_array = np.linalg.eigh(operator_ndarray)
    elif backend == "scipy":
        eigenvalues, eigenvectors_array = scipy.linalg.eigh(operator_ndarray)
    elif backend == "torch":
        eigenvalues, eigenvectors_array = _torch_eigh(operator_ndarray)
    else:
        eigenvalues, eigenvectors_array = _jax_eigh(operator_ndarray)

    if not return_eigenvectors:
        eigenvectors_array = None

    return _build_hermitian_result(
        eigenvalues=eigenvalues,
        eigenvectors_array=eigenvectors_array,
        backend=backend,
        dims=dims,
    )


def solve_general_eigenproblem(
        operator,
        *,
        backend="qobj",
        return_eigenvectors=True):
    if backend not in GENERAL_EIGEN_BACKENDS:
        raise ValueError(
            "unsupported general eigen backend %r; expected one of %s"
            % (backend, GENERAL_EIGEN_BACKENDS)
        )

    operator_qobj = _as_qobj_operator(operator)
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
    )


def _as_qobj_operator(operator):
    if isinstance(operator, qutip.Qobj):
        return operator

    return qutip.Qobj(np.asarray(operator, dtype=np.complex128))


def _build_hermitian_result(eigenvalues, eigenvectors_array, backend, dims):
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
    )


def _torch_eigh(operator_ndarray):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch backend requires installing the optional 'torch' extra"
        ) from exc

    operator_tensor = torch.as_tensor(operator_ndarray, dtype=torch.complex128)
    eigenvalues, eigenvectors = torch.linalg.eigh(operator_tensor)

    return np.asarray(eigenvalues.cpu(), dtype=np.float64), np.asarray(
        eigenvectors.cpu(), dtype=np.complex128)


def _jax_eigh(operator_ndarray):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "jax backend requires installing the optional 'jax' extra"
        ) from exc

    jax.config.update("jax_enable_x64", True)
    operator_array = jnp.asarray(operator_ndarray, dtype=jnp.complex128)
    eigenvalues, eigenvectors = jnp.linalg.eigh(operator_array)

    return np.asarray(eigenvalues, dtype=np.float64), np.asarray(
        eigenvectors, dtype=np.complex128)


__all__ = [
    "EigenResult",
    "GENERAL_EIGEN_BACKENDS",
    "HERMITIAN_EIGEN_BACKENDS",
    "solve_general_eigenproblem",
    "solve_hermitian_eigenproblem",
]
