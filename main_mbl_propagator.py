import numpy as np
import scipy
import scipy.stats
import itertools
import qutip
import uuid
import argparse
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

"""
Generating the propagator for a many-body localized system
"""
if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(
        prog="main_mbl_propagator.py",
        description="",
        epilog=""
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--tduration", type=float)
    argument_parser.add_argument("--jIntMean", type=float)
    argument_parser.add_argument("--bFieldMean", type=float)
    argument_parser.add_argument("--jIntStd", type=float)
    argument_parser.add_argument("--bFieldStd", type=float)
    argument_parser.add_argument("--anglePolarPiMin", type=float)
    argument_parser.add_argument("--anglePolarPiMax", type=float)

    args = argument_parser.parse_args()

    systemsize = args.systemsize
    tduration = args.tduration
    jIntMean = args.jIntMean
    jIntStd = args.jIntStd
    bFieldMean = args.bFieldMean
    bFieldStd = args.bFieldStd
    anglePolarPiMin = args.anglePolarPiMin
    anglePolarPiMax = args.anglePolarPiMax

    sigma0 = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmay = qutip.sigmay()
    sigmaz = qutip.sigmaz()

    sigmaz_sigmaz = qutip.tensor(sigmaz, sigmaz)
    
    projector_g = (sigma0 + sigmaz) * 0.5
    projector_r = (sigma0 - sigmaz) * 0.5

    projector_rr = qutip.tensor(projector_r, projector_r)

    jInt_samples = scipy.stats.norm.rvs(
        size=(systemsize-1), loc=jIntMean, scale=jIntStd)
    
    bField_samples = scipy.stats.norm.rvs(
        size=systemsize, loc=bFieldMean, scale=bFieldStd)

    theta_samples = scipy.stats.uniform.rvs(
        size=systemsize,
        loc=anglePolarPiMin * np.pi,
        scale=(anglePolarPiMax - anglePolarPiMin)* np.pi,)

    ## Generating Hamiltonian 
    bperp_terms = [
         bField_samples[ix_site] * np.sin(theta_samples[ix_site]) * sigmax \
            for ix_site in range(systemsize)]

    bparallel_terms = [
        bField_samples[ix_site] * np.cos(theta_samples[ix_site]) * sigmaz \
            for ix_site in range(systemsize)]

    interaction_zz_terms = [
        jInt_samples[ix_site] * sigmaz_sigmaz \
            for ix_site in range(systemsize-1)]

    hamiltonian = 0.0 * qutip.qip.operations.expand_operator(
                        sigma0, N=systemsize, targets=(0,))

    logging.info("bField_samples = %s" % bField_samples)
    logging.info("theta_samples = %s" % theta_samples)
    logging.info("jInt_samples = %s" % jInt_samples)
    
    for ix_site in range(systemsize):
        h = bperp_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

        h = bparallel_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site,))

    for ix_site in range(systemsize-1):
        h = interaction_zz_terms[ix_site]
        hamiltonian = hamiltonian + \
                qutip.qip.operations.expand_operator(
                        h, N=systemsize, targets=(ix_site, ix_site+1))

    ## Diagonalizing Hamiltonian
    hamiltonian_ndarray = hamiltonian.full()
    energies, basis_changer = np.linalg.eigh(hamiltonian_ndarray)

    basis_changer_qobj = qutip.qobj.Qobj(basis_changer, dims=hamiltonian.dims)

    logging.info("basis_changer_obj = %s" % (basis_changer_qobj,))

    ## Operators of interest
    sigmax_array = np.asarray(
        [qutip.qip.operations.expand_operator(
            sigmax / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    sigmay_array = np.asarray(
        [qutip.qip.operations.expand_operator(
            sigmay / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    sigmaz_array = np.asarray(
        [qutip.qip.operations.expand_operator(
            sigmaz / np.sqrt(1 << systemsize), N=systemsize, targets=(ix_site,))
            for ix_site in range(systemsize)], dtype=object)

    logging.info("sigmax: \n %s \n" % (sigmax_array,))
    logging.info("sigmay: \n %s \n" % (sigmay_array,))
    logging.info("sigmaz: \n %s \n" % (sigmaz_array,))

    sigmax_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.qobj.Qobj(x, dims=hamiltonian.dims) * basis_changer_qobj
            for x in sigmax_array]

    sigmay_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.qobj.Qobj(y, dims=hamiltonian.dims) * basis_changer_qobj
            for y in sigmay_array]

    sigmaz_hamiltonian_eigenbasis_array = \
        [basis_changer_qobj.dag() * qutip.qobj.Qobj(z, dims=hamiltonian.dims) * basis_changer_qobj
            for z in sigmaz_array]

    logging.info("sigmax: \n %s \n" % (sigmax_hamiltonian_eigenbasis_array,))
    logging.info("sigmay: \n %s \n" % (sigmay_hamiltonian_eigenbasis_array,))
    logging.info("sigmaz: \n %s \n" % (sigmaz_hamiltonian_eigenbasis_array,))

    ## Time evolution
    times_array:np.ndarray = np.arange(0.0, 1.0625, 0.0625)

    eigenphases:np.ndarray = np.empty((len(times_array), len(energies)), dtype=complex)

    for ix_time, t in enumerate(times_array):
        eigenphases[ix_time, :] = np.exp(-1j * t * energies)

    sigmax_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)
    sigmay_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)
    sigmaz_time_evolved_array = np.empty((systemsize, len(times_array)), dtype=object)

    for ix_time, t in enumerate(times_array):
        propagator = \
                basis_changer_qobj * \
                qutip.qobj.Qobj(
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

    overlap_matrix = np.empty((3*systemsize, 3*systemsize, len(times_array)), dtype=complex)

    for ix_time, t in enumerate(times_array):
        for ix_site_initial in range(systemsize):
            for ix_site_final in range(systemsize):

                op_initial = qutip.qobj.Qobj(sigmax_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.qobj.Qobj(sigmax_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                logging.info("op_initial type = %s" % type(op_initial))
                logging.info("op_final type = %s" % type(op_final))

                overlap = (op_initial * op_final).tr()
                overlap_matrix[ix_site_initial, ix_site_final, ix_time] = overlap

                op_initial = qutip.qobj.Qobj(sigmay_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.qobj.Qobj(sigmay_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                overlap = (op_initial * op_final).tr()
                overlap_matrix[systemsize + ix_site_initial, systemsize + ix_site_final, ix_time] = overlap

                op_initial = qutip.qobj.Qobj(sigmaz_array[ix_site_initial], dims=hamiltonian.dims)
                op_final = qutip.qobj.Qobj(sigmaz_time_evolved_array[ix_site_final, ix_time], dims=hamiltonian.dims)

                overlap = (op_initial * op_final).tr()
                overlap_matrix[2 * systemsize + ix_site_initial, 2 * systemsize + ix_site_final, ix_time] = overlap

    logging.info("overlap_matrix = \n%s" % (np.round(overlap_matrix[:, :, -1], 4),))

