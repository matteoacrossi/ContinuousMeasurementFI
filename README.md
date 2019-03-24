# ContinuousMeasurementFI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1456660.svg)](https://doi.org/10.5281/zenodo.1456660)

Fisher information for magnetometry and frequency estimation with continuously monitored spin systems, with independent Markovian noise acting on each spin. 

Companion code for
[F. Albarelli, M. A. C. Rossi, D. Tamascelli1, and M, G. Genoni, Quantum 2, 110 (2018)](https://doi.org/10.22331/q-2018-12-03-110).

The algorithm is described in Sec. V of the paper.


## Installation

> This version is compatible with Julia v0.7 onwards

From the Julia `pkg` REPL (press `]`)
```julia
  pkg> add https://github.com/matteoacrossi/ContinuousMeasurementFI
```

## Usage

```julia
    using ContinuousMeasurementFI
    (t, FI, QFI) = Eff_QFI(kwargs...)
```

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous monitoring of each half-spin
particle affected by noise at an angle θ, with efficiency η using SME
(stochastic master equation) or SSE (stochastic Schrödinger equation).

The function returns a tuple `(t, FI, QFI)` containing the time vector and the
vectors containing the FI and average QFI

### Arguments

* `Nj`: number of spins
* `Ntraj`: number of trajectories for the SSE
* `Tfinal`: final time of evolution
* `measurement = :pd` measurement (either `:pd` or `:hd`)
* `dt`: timestep of the evolution
* `κ = 1`: the noise coupling
* `θ = 0`: noise angle (0 parallel, π/2 transverse)
* `ω = 0`: local value of the frequency
* `η = 1`: measurement efficiency

### Example
```julia
using Plots
include("Eff_QFI.jl")

(t, fi, qfi) = Eff_QFI(Nj=5, Ntraj=10000, Tfinal=5., dt=.1; measurement=:pd, θ = pi/2, ω = 1)
plot(t, (fi + qfi)./t, xlabel="t", ylabel="Q/t")
```

![](readme.png)


### Distributed computing
`ContinuousMeasurementFI` can parallelize the Montecarlo evaluation
of trajectories using the builtin distributed computing system of Julia

```julia
using Distributed

addprocs(#_of_processes)

@everywhere using ContinuousMeasurementFI
(t, FI, QFI) = Eff_QFI(kwargs...)
```

## Dependencies
* [`ZChop`](https://github.com/jlapeyre/ZChop.jl) for rounding off small imaginary parts in ρ

## Citing
If you found the code useful for your research, please cite the paper:

[F. Albarelli, M. A. C. Rossi, D. Tamascelli1, and M, G. Genoni, Quantum 2, 110 (2018)](https://doi.org/10.22331/q-2018-12-03-110).

## License
[MIT License](LICENSE)
