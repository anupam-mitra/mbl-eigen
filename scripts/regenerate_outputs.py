"""Regenerate QMBS / PXP and MBL-DTC plots from existing output filenames.

Unique parameter combinations are parsed from ``Plots/*.pdf`` filenames and
the corresponding workflows are re-run via direct function calls. QMBS plots
are exactly reproducible; MBL-DTC plots depend on an unknown original seed,
so regenerated MBL-DTC plots will differ from the originals.
"""

import argparse
import logging
import math
import os
import re
from re import Match
from types import SimpleNamespace

from mbl_eigen import GENERAL_EIGEN_BACKENDS
from mbl_eigen.mbldtc_app import run_mbldtc
from mbl_eigen.qmbs_app import _OMEGA, _VRR, run_qmbs


logger = logging.getLogger(__name__)


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

_QMBS_SFIM_RE = re.compile(
    rf"qmbs_sfim_N=(?P<systemsize>\d+)"
    rf"_tduration=(?P<tduration>{_NUMBER})"
    rf"_Vrr=(?P<Vrr>{_NUMBER})"
    rf"_Omega=(?P<Omega>{_NUMBER})"
    rf"_Delta=(?P<Delta>{_NUMBER})"
    rf"(?:_{_UUID})?\.pdf"
)
_QMBS_PXP_RE = re.compile(
    rf"qmbs_pxp_N=(?P<systemsize>\d+)"
    rf"_tduration=(?P<tduration>{_NUMBER})"
    rf"_Omega=(?P<Omega>{_NUMBER})"
    rf"_Delta=(?P<Delta>{_NUMBER})"
    rf"(?:_{_UUID})?\.pdf"
)
_MBLDTC_RE = re.compile(
    rf"mbldtc_N=(?P<systemsize>\d+)"
    rf"_thetax=(?P<thetaXPi>{_NUMBER})pi"
    rf"(?:_{_UUID})?\.pdf"
)


def _collect_qmbs_params(plots_dir: str) -> set[tuple[int, float, float]]:
    """Collect unique QMBS parameter triples from existing plot filenames.

    One ``run_qmbs`` call regenerates both the SFIM and PXP plots, so the
    triples parsed from both filename patterns are merged into one set.
    Filenames whose Vrr or Omega values are not reproducible with the
    current code are skipped with a warning.

    Parameters
    ----------
    plots_dir : str
        Directory containing existing PDF outputs.

    Returns
    -------
    set[tuple[int, float, float]]
        Unique ``(systemsize, tduration, Delta)`` triples.
    """
    params: set[tuple[int, float, float]] = set()
    for filename in sorted(os.listdir(plots_dir)):
        sfim = _QMBS_SFIM_RE.fullmatch(filename)
        if sfim is not None:
            if not (
                math.isclose(float(sfim.group("Vrr")), _VRR)
                and math.isclose(float(sfim.group("Omega")), _OMEGA)
            ):
                logger.warning(
                    "skipping %s: Vrr/Omega not reproducible with current code",
                    filename,
                )
                continue
            params.add(_qmbs_params_from_match(sfim))
            continue
        pxp = _QMBS_PXP_RE.fullmatch(filename)
        if pxp is not None:
            if not math.isclose(float(pxp.group("Omega")), _OMEGA):
                logger.warning(
                    "skipping %s: Omega not reproducible with current code",
                    filename,
                )
                continue
            params.add(_qmbs_params_from_match(pxp))
    return params


def _collect_mbldtc_params(plots_dir: str) -> set[tuple[int, float]]:
    """Collect unique MBL-DTC parameter pairs from existing plot filenames.

    Parameters
    ----------
    plots_dir : str
        Directory containing existing PDF outputs.

    Returns
    -------
    set[tuple[int, float]]
        Unique ``(systemsize, thetaXPi)`` pairs.
    """
    params: set[tuple[int, float]] = set()
    for filename in sorted(os.listdir(plots_dir)):
        match = _MBLDTC_RE.fullmatch(filename)
        if match is not None:
            params.add(
                (
                    int(match.group("systemsize")),
                    float(match.group("thetaXPi")),
                )
            )
    return params


def _qmbs_params_from_match(match: Match) -> tuple[int, float, float]:
    """Extract the ``(systemsize, tduration, Delta)`` triple from a match.

    Parameters
    ----------
    match : Match
        Regex match of a QMBS SFIM or PXP plot filename.

    Returns
    -------
    tuple[int, float, float]
        Parsed ``(systemsize, tduration, Delta)`` values.
    """
    return (
        int(match.group("systemsize")),
        float(match.group("tduration")),
        float(match.group("Delta")),
    )


def _general_backend(backend: str) -> str:
    """Return a backend valid for general (non-Hermitian) eigenproblems.

    Parameters
    ----------
    backend : str
        Requested eigenproblem backend.

    Returns
    -------
    str
        ``backend`` if supported for general eigenproblems, otherwise the
        first supported general backend.
    """
    if backend in GENERAL_EIGEN_BACKENDS:
        return backend
    fallback = GENERAL_EIGEN_BACKENDS[0]
    logger.warning(
        "backend %s unsupported for general eigenproblems; using %s",
        backend,
        fallback,
    )
    return fallback


def _build_qmbs_args(
    systemsize: int,
    tduration: float,
    Delta: float,
    backend: str,
    device: str,
) -> SimpleNamespace:
    """Build the argument namespace consumed by ``run_qmbs``.

    Parameters
    ----------
    systemsize : int
        Number of lattice sites.
    tduration : float
        Evolution duration.
    Delta : float
        Detuning in units of Omega.
    backend : str
        Eigenproblem backend.
    device : str
        Eigenproblem device.

    Returns
    -------
    SimpleNamespace
        Namespace with the exact fields read by ``run_qmbs``.
    """
    return SimpleNamespace(
        systemsize=systemsize,
        tduration=tduration,
        Delta=Delta,
        eigenBackend=backend,
        eigenDevice=device,
    )


def _build_mbldtc_args(
    systemsize: int,
    thetaXPi: float,
    backend: str,
    device: str,
    seed: int,
) -> SimpleNamespace:
    """Build the argument namespace consumed by ``run_mbldtc``.

    Parameters
    ----------
    systemsize : int
        Number of lattice sites.
    thetaXPi : float
        X rotation angle in units of pi.
    backend : str
        Eigenproblem backend.
    device : str
        Eigenproblem device.
    seed : int
        Seed for the random disorder angle sampler.

    Returns
    -------
    SimpleNamespace
        Namespace with the exact fields read by ``run_mbldtc``.
    """
    return SimpleNamespace(
        systemsize=systemsize,
        thetaXPi=thetaXPi,
        eigenBackend=backend,
        eigenDevice=device,
        seed=seed,
    )


def main(
    plots_dir: str = "Plots",
    seed: int = 0,
    backend: str = "numpy",
    device: str = "auto",
) -> None:
    """Regenerate all reproducible plots found in ``plots_dir``.

    Parameters
    ----------
    plots_dir : str
        Directory scanned for existing PDF filenames; new PDFs are written
        here by the plotting module.
    seed : int
        Seed used to rebuild a deterministic rng for MBL-DTC runs.
    backend : str
        Eigenproblem backend passed to the workflows.
    device : str
        Eigenproblem device passed to the workflows.

    Notes
    -----
    The plotting module writes relative to the current working directory,
    so ``main`` chdirs into ``plots_dir`` (restored afterwards) before
    running the workflows.

    Exact reproduction holds only if the original ``tduration``/``Delta``
    values had no more than 6 significant digits — ``%g`` truncation in
    the filename formatters may have already lost precision.

    Admitted combos always regenerate both plots with the current code
    (``Omega=1``, ``Vrr=100``); filenames carrying other values are
    skipped with a warning and never reproduced.
    """
    resolved_plots_dir = os.path.abspath(plots_dir)
    if not os.path.isdir(resolved_plots_dir):
        raise SystemExit(
            "plots directory does not exist: %s" % resolved_plots_dir
        )
    original_cwd = os.getcwd()
    os.chdir(resolved_plots_dir)
    try:
        qmbs_params = _collect_qmbs_params(resolved_plots_dir)
        mbldtc_params = _collect_mbldtc_params(resolved_plots_dir)
        total = len(qmbs_params) + len(mbldtc_params)
        succeeded = 0
        failed = 0
        for systemsize, tduration, Delta in sorted(qmbs_params):
            args = _build_qmbs_args(
                systemsize, tduration, Delta, backend, device
            )
            try:
                run_qmbs(args)
                succeeded += 1
            except Exception as exc:
                logger.error(
                    "run_qmbs failed: systemsize=%d tduration=%g Delta=%g (%s)",
                    systemsize,
                    tduration,
                    Delta,
                    exc,
                )
                failed += 1
        mbldtc_backend = _general_backend(backend)
        for systemsize, thetaXPi in sorted(mbldtc_params):
            args = _build_mbldtc_args(
                systemsize, thetaXPi, mbldtc_backend, device, seed
            )
            try:
                run_mbldtc(args)
                succeeded += 1
            except Exception as exc:
                logger.error(
                    "run_mbldtc failed: systemsize=%d thetaXPi=%g (%s)",
                    systemsize,
                    thetaXPi,
                    exc,
                )
                failed += 1
        logger.info(
            "%d runs attempted, %d succeeded, %d failed",
            total,
            succeeded,
            failed,
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate QMBS / PXP and MBL-DTC plots from existing "
            "Plots/*.pdf filenames."
        )
    )
    parser.add_argument("--plots-dir", type=str, default="Plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default="numpy")
    parser.add_argument("--device", type=str, default="auto")
    cli_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(
        plots_dir=cli_args.plots_dir,
        seed=cli_args.seed,
        backend=cli_args.backend,
        device=cli_args.device,
    )
