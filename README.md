# Eigenvector and eigenvalue analysis for systems with many-body localization

## Requirements

- `qutip`
- `numpy`
- `scipy`
- `matplotlib`

## Examples

### Quantum many-body scars (QMBS)

To calculate and plot eigenvalues of the time evolution operator at
time `tduration=1.0`, for a system size of `systemsize=6` with
detuning `Delta=0.1` in units of Rabi frequency `Omega`, run the 
following.
```
python main_qmbs.py --systemsize=6 --tduration=1.0 --Delta=0.1
```

### Many-body localized discrete time crystal (MBL-DTC)

To calculate and plot eigenvalues of the Floquet operator for a 
system size of `systemsize=8` and a transverse rotation angle
`thetaXPi=0.6` in units of $\pi$, run the following.
```
python main_mbldtc.py --systemsize=8 --thetaXPi=0.76
```

### Many-body localization

To calculate and plot eigenvalues and entanglement entropies of eigenvectors of
the Hamiltonian system size of `systemsize=12` and a polar angle between
`anglePolarPiMin=0.0` and `anglePolarPiMax=1.0`, interaction `jInt` sampled from
a normal distrbution with mean `jIntMean` and standard deviation `jIntStd`, 
magnetic field `bField` sampled from a normal distribution with mean `bFieldMean`
and standard deviation `bFieldStd, run the following.
```
python main_mbl.py --jIntMean=1.0 --jIntStd=1.0 --bFieldMean=1.0 --bFieldStd=1.0 --anglePolarPiMin=0.0 --anglePolarPiMax=1.0 --systemsize=12
```
