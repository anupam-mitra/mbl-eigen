import logging

import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip.qip.operations import expand_operator

from . import eigensolver
from . import level_repulsion
from . import output_names
from .mbl_model import build_mbl_model


"""
Generating the PXP Hamiltonian from Rydberg blockade
"""


def _build_model_from_args(args):
    return build_mbl_model(
        systemsize=args.systemsize,
        jIntMean=args.jIntMean,
        jIntStd=args.jIntStd,
        bFieldMean=args.bFieldMean,
        bFieldStd=args.bFieldStd,
        anglePolarPiMin=args.anglePolarPiMin,
        anglePolarPiMax=args.anglePolarPiMax,
    )


def run_mbl(args):
    systemsize = args.systemsize
    tduration = args.tduration
    jIntMean = args.jIntMean
    jIntStd = args.jIntStd
    bFieldMean = args.bFieldMean
    bFieldStd = args.bFieldStd
    anglePolarPiMin = args.anglePolarPiMin
    anglePolarPiMax = args.anglePolarPiMax

    model = _build_model_from_args(args)

    logging.info("bField_samples = %s" % model.bField_samples)
    logging.info("theta_samples = %s" % model.theta_samples)
    logging.info("jInt_Samples = %s" % model.jInt_samples)

    ## Diagonalizing Hamiltonian
    diagonalization = eigensolver.solve_hermitian_eigenproblem(
        model.hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    eigenvalues = diagonalization.eigenvalues
    eigenvectors = diagonalization.as_qobj_kets()

    eigenvector_entropies = np.array([
            qutip.entropy_vn(qutip.ptrace(v, [ix for ix in range(systemsize >> 1)]))
                 for v in eigenvectors])

    logging.info(eigenvector_entropies.shape)

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
            eigenvalues, fraction_cutoff=0.0, use_spacing=True)

    logging.info("ratio = %g" % (ratio,))

    ## Eigenphases
    eigenvalues_unitary = np.exp(-1j * eigenvalues * tduration)

    eigenphases = np.sort([(np.angle(v) % 2.0 * np.pi) for v in eigenvalues_unitary])

    ratio = level_repulsion.calc_mean_adjacent_level_spacing_ratio(
            eigenphases, fraction_cutoff=0.0, use_spacing=True)

    logging.info("ratio = %g" % (ratio))

    ## Plotting
    ## Eigenvector entropy plot for the Ising Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(18.0 / 2.54, 12.0 / 2.54))

    ax.set_xlabel(r'Energy')
    ax.set_ylabel(r'$\mathcal{S}_1$')

    ax.plot(eigenvalues,
            eigenvector_entropies,
            marker='o', markeredgecolor='k', markerfacecolor='blue',
            ls='',
    )

    ax.axhline(np.log(2) * (systemsize >> 1), ls='dotted', color='k')

    plotfilename = output_names.mbl_entropy_plot_name(
        systemsize=systemsize,
        anglePolarPiMin=anglePolarPiMin,
        anglePolarPiMax=anglePolarPiMax,
        jIntMean=jIntMean,
        jIntStd=jIntStd,
        bFieldMean=bFieldMean,
        bFieldStd=bFieldStd,
    )

    fig.savefig(plotfilename)
    plt.close()


def run_mbl_dynamics(args):
    systemsize = args.systemsize
    jIntMean = args.jIntMean
    jIntStd = args.jIntStd
    bFieldMean = args.bFieldMean
    bFieldStd = args.bFieldStd
    anglePolarPiMin = args.anglePolarPiMin
    anglePolarPiMax = args.anglePolarPiMax

    model = _build_model_from_args(args)

    print("bField_samples = %s" % model.bField_samples)
    print("theta_samples = %s" % model.theta_samples)
    print("jInt_samples = %s" % model.jInt_samples)

    ## Diagonalizing Hamiltonian
    diagonalization = eigensolver.solve_hermitian_eigenproblem(
        model.hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    eigenvalues = diagonalization.eigenvalues
    eigenvectors = diagonalization.as_qobj_kets()

    ## Time evolution
    ket_initial = qutip.ket('1' * systemsize)
    print(ket_initial)

    amplitude_eigenvectors = [v.overlap(ket_initial) for v in eigenvectors]
    print(amplitude_eigenvectors)

    times_array = np.arange(0.0, 10.0625, 0.0625)

    amplitude_return_array = np.empty_like(times_array, dtype=complex)

    for ix_time, t in enumerate(times_array):
        a = np.sum([
            np.exp(-1j * t * eigenvalues[k]) * \
             np.abs(amplitude_eigenvectors[k])**2
            for k in range(len(eigenvalues))])

        amplitude_return_array[ix_time] = a

    print(amplitude_return_array)

    ## Eigenvector entropy plot for the Ising Hamiltonian
    fig, ax = plt.subplots(1, 1, figsize=(18.0 / 2.54, 12.0 / 2.54))

    ax.plot(times_array, -np.log(np.abs(amplitude_return_array)**2) / systemsize)
    ax.set_ylabel(r"$\lambda(t)$")
    ax.set_xlabel(r"$B_{\mathrm{mean}} t$")

    plotfilename = output_names.mbl_dynamics_plot_name(
        systemsize=systemsize,
        anglePolarPiMin=anglePolarPiMin,
        anglePolarPiMax=anglePolarPiMax,
        jIntMean=jIntMean,
        jIntStd=jIntStd,
        bFieldMean=bFieldMean,
        bFieldStd=bFieldStd,
    )

    fig.savefig(plotfilename)
    plt.close()


def run_mbl_propagator(args):
    systemsize = args.systemsize
    model = _build_model_from_args(args)

    logging.info("bField_samples = %s" % model.bField_samples)
    logging.info("theta_samples = %s" % model.theta_samples)
    logging.info("jInt_samples = %s" % model.jInt_samples)

    hamiltonian = model.hamiltonian
    sigmax = model.sigmax
    sigmay = model.sigmay
    sigmaz = model.sigmaz

    ## Diagonalizing Hamiltonian
    diagonalization = eigensolver.solve_hermitian_eigenproblem(
        hamiltonian,
        backend=args.eigenBackend,
        device=args.eigenDevice,
    )
    energies = diagonalization.eigenvalues
    basis_changer_qobj = diagonalization.as_basis_qobj()

    logging.info("basis_changer_obj = %s" % (basis_changer_qobj,))

    ## Operators of interest
    sigmax_array = np.asarray(
        [expand_operator(
            sigmax / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    sigmay_array = np.asarray(
        [expand_operator(
            sigmay / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    sigmaz_array = np.asarray(
        [expand_operator(
            sigmaz / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    logging.info("sigmax: \n %s \n" % (sigmax_array,))
    logging.info("sigmay: \n %s \n" % (sigmay_array,))
    logging.info("sigmaz: \n %s \n" % (sigmaz_array,))

    sigmax_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.Qobj(x, dims=hamiltonian.dims) * basis_changer_qobj
            for x in sigmax_array]

    sigmay_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.Qobj(y, dims=hamiltonian.dims) * basis_changer_qobj
            for y in sigmay_array]

    sigmaz_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.Qobj(z, dims=hamiltonian.dims) * basis_changer_qobj
            for z in sigmaz_array]

    logging.info("sigmax: \n %s \n" % (sigmax_hamiltonian_eigenbasis_array,))
    logging.info("sigmay: \n %s \n" % (sigmay_hamiltonian_eigenbasis_array,))
    logging.info("sigmaz: \n %s \n" % (sigmaz_hamiltonian_eigenbasis_array,))

    ## Time evolution
    times_array: np.ndarray = np.arange(0.0, 1.0625, 0.0625)

    eigenphases: np.ndarray = np.empty((len(times_array), len(energies)), dtype=complex)

    for ix_time, t in enumerate(times_array):
        eigenphases[ix_time, :] = np.exp(-1j * t * energies)

    sigmax_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)
    sigmay_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)
    sigmaz_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)

    for ix_time, t in enumerate(times_array):
        propagator = \
                basis_changer_qobj * \
                qutip.Qobj(
                    np.diag(eigenphases[ix_time, :]),
                    dims=basis_changer_qobj.dims)

        logging.info("propagator type = %s" % (type(propagator),))

        for ix_site in range(systemsize):
            x = sigmax_hamiltonian_eigenbasis_array[ix_site]
            y = sigmay_hamiltonian_eigenbasis_array[ix_site]
            z = sigmaz_hamiltonian_eigenbasis_array[ix_site]

            logging.info("x type = %s" % type(x))
            logging.info("y type = %s" % type(y))
            logging.info("z type = %s" % type(z))

            sigmax_time_evolved_array[ix_site, ix_time] = \
                    propagator * x * propagator.dag()

            sigmay_time_evolved_array[ix_site, ix_time] = \
                    propagator * y * propagator.dag()

            sigmaz_time_evolved_array[ix_site, ix_time] = \
                    propagator * z * propagator.dag()

    logging.info("sigmax time evolved: \n %s \n" % (sigmax_time_evolved_array[0, -1],))
    logging.info("sigmay time evolved: \n %s \n" % (sigmay_time_evolved_array[0, -1],))
    logging.info("sigmaz time evolved: \n %s \n" % (sigmaz_time_evolved_array[0, -1],))

    overlap_matrix = np.empty((3 * systemsize, 3 * systemsize, len(times_array)), dtype=complex)

    for ix_time, t in enumerate(times_array):
        for ix_site_initial in range(systemsize):
            for ix_site_final in range(systemsize):

                op_initial = qutip.Qobj(sigmax_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.Qobj(sigmax_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                logging.info("op_initial type = %s" % type(op_initial))
                logging.info("op_final type = %s" % type(op_final))

                overlap = (op_initial * op_final).tr()
                overlap_matrix[ix_site_initial, ix_site_final, ix_time] = overlap

                op_initial = qutip.Qobj(sigmay_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.Qobj(sigmay_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                overlap = (op_initial * op_final).tr()
                overlap_matrix[systemsize + ix_site_initial, systemsize + ix_site_final, ix_time] = overlap

                op_initial = qutip.Qobj(sigmaz_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.Qobj(sigmaz_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                overlap = (op_initial * op_final).tr()
                overlap_matrix[2 * systemsize + ix_site_initial, 2 * systemsize + ix_site_final, ix_time] = overlap

    logging.info("overlap_matrix = \n%s" % (np.round(overlap_matrix[:, :, -1], 4),))
