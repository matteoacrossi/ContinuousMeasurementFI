# ContinuousMeasurementFI
![arxiv:1803.05891](https://img.shields.io/badge/arXiv-1803.05891-brightgreen.svg?link=https://arxiv.org/abs/1803.05891)

Fisher information for magnetometry with continuously monitored spin systems.



## Noise at an arbitrary angle
This code covers the case of independent noise (i.e. noise applied to each qubit) at an arbitrary angle θ.

This should implement the master equation

$$ d\rho = -i [H,\rho] + \sum_{j=1}^n \mathcal{D}[L_j] \rho dt + \sum_{j=1}^n \eta_j \mathcal H [L_j] \rho dW$$

where the noise operators are

$$ L_j = \cos \theta \sigma_y + \sin \theta \sigma_z $$

so that when $\theta = 0$ we recover the parallel noise case and when $\theta = \pi/2$ we recover the transverse noise case.

## TODO
Write the most general evolution routine, and pass noise as an argument.

## Dependencies
* [`QuantumOptics`](https://github.com/qojulia/QuantumOptics.jl) for constructing states and operators
* [`ZChop`](https://github.com/jlapeyre/ZChop.jl) for rounding off small imaginary parts in ρ
