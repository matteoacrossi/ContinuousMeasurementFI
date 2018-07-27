include("NoiseOperators.jl")
include("Eff_QFI_PD_pure.jl")
include("States.jl")
using Plots

κ_ind = 1.
θ = pi/2
ω = 0.
Nj = 2

@time t, fi, qfi = Eff_QFI_HD_pure(Nj,          # Number of spins
    100,                       # Number of trajectories
    5.,                    # Final time
    .001,                        # Time step
    κ = 1.,                             # Noise coupling
    θ = pi/2,                             # Noise angle
    ω = 1.)                             # Frequency


plot(t, fi)
plot!(t, qfi)
