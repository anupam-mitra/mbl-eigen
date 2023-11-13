# Eigenvector and eigenvalue analysis for systems with many-body localization

## Requirements

- `qutip`
- `numpy`
- `scipy`
- `matplotlib`

## Examples

Quantum many-body scars (QMBS)

To calculate and plot eigenvalues of the time evolution operator at
time `tduration=1.0`, for a system size of `systemsize=6` with
detuning `Delta=0.1` in units of Rabi frequency `Omega`, run the 
following.
```
python main_qmbs.py --systemsize=6 --tduration=1.0 --Delta=0.1
```

Many-body localized discrete time crystal (MBL-DTC)

To calculate and plot eigenvalues of the Floquet operator for a 
system size of `systemsize=8` and a transverse rotation angle
`thetaXPi=0.6` in units of $\pi$, run the following.
```
python main_mbldtc.py --systemsize=8 --thetaXPi=0.76
```
