include("src/Eff_QFI_HD_sup.jl")
include("src/Eff_QFI_HD.jl")

include("src/NoiseOperators.jl")
include("src/States.jl")
include("src/Fisher.jl")

using Plots
using Test
using Random

Ntraj = 40
Tfinal = 5.
dt = 0.01
κ_ind = 1.
κ_coll = 1.
θ = pi/4 # Angle of the independent noise (θ = 0 : parallel)
ω = 1.
Nj = 3


# ω is the parameter that we want to estimate
H = ω * σ(:z, Nj) / 2       # Hamiltonian of the spin system

dH = σ(:z, Nj) / 2          # Derivative of H wrt the parameter ω

#non_monitored_noise_op = []
non_monitored_noise_op = (sqrt(κ_ind/2) *
    [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:x, j, Nj)) for j = 1:Nj])

monitored_noise_op = [sqrt(κ_coll/2) * σ(:x, Nj)]

seed = 2
Random.seed!(seed)
@time t1, fi, qfi = Eff_QFI_HD(Ntraj,# Number of trajectories
    Tfinal,                                  # Final time
    dt,                               # Time step
    H, dH,                          # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,           # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = random_state
    )          # Initial state

Random.seed!(seed)
@time t2, fi2, qfi2 = Eff_QFI_HD_sup(Ntraj,# Number of trajectories
        Tfinal,                                  # Final time
        dt,                               # Time step
        H, dH,                          # Hamiltonian and its derivative wrt ω
        non_monitored_noise_op,           # Non monitored noise operators
        monitored_noise_op;                 # Monitored noise operators
        initial_state = random_state
        )          # Initial state


# plot(t1, fi, label= "FI")
# plot!(t1, (fi + qfi), label= "Eff QFI")

# plot!(t2, fi2, linestyle=:dash, label="FI 2")
# plot!(t2, (fi2 + qfi2), label= "Eff QFI 2")

@test fi ≈ fi2
@test qfi ≈ qfi2
