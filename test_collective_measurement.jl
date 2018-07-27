include("NoiseOperators.jl")
include("Eff_QFI_PD.jl")
include("States.jl")

using Plots

κ_ind = 1.
κ_coll = 0.1
θ = 0.
ω = 0.
Nj = 1

# ω is the parameter that we want to estimate
H = ω * σ(:z, Nj) / 2       # Hamiltonian of the spin system

dH = σ(:z, Nj) / 2          # Derivative of H wrt the parameter ω

#non_monitored_noise_op = []
non_monitored_noise_op = sqrt(κ_ind/2) *
    [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:x, j, Nj)) for j = 1:Nj]

monitored_noise_op = [sqrt(κ_coll/2) * σ(:x, Nj)]

@time t, fi, qfi = Eff_QFI_PD(Nj,          # Number of spins
    100,                               # Number of trajectories
    5.,                                 # Final time
    .001,                                # Time step
    H, dH,                              # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,             # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = plus_state)          # Initial state


plot(t, fi ./ t)
plot!(t, qfi ./ t)
