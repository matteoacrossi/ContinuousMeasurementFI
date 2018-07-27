include("Eff_QFI_PD_sup.jl")
include("Eff_QFI_PD.jl")
using Plots
using Base.Test

κ_ind = 1.
κ_coll = 1.
θ = 0.
ω = 1.
Nj = 4

# ω is the parameter that we want to estimate
H = ω * σ(:z, Nj) / 2       # Hamiltonian of the spin system

dH = σ(:z, Nj) / 2          # Derivative of H wrt the parameter ω

#non_monitored_noise_op = []
non_monitored_noise_op = (sqrt(κ_ind/2) *
    [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:x, j, Nj)) for j = 1:Nj])

monitored_noise_op = [sqrt(κ_coll/2) * σ(:x, Nj)]

srand(100)
@time t1, fi, qfi = Eff_QFI_PD(40,# Number of trajectories
    5,                                  # Final time
    .01,                               # Time step
    H, dH,                          # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,           # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = ghz_state
    )          # Initial state

srand(100)
@time t2, fi2, qfi2 = Eff_QFI_PD_sup(40,# Number of trajectories
        5,                                  # Final time
        .01,                               # Time step
        H, dH,                          # Hamiltonian and its derivative wrt ω
        non_monitored_noise_op,           # Non monitored noise operators
        monitored_noise_op;                 # Monitored noise operators
        initial_state = ghz_state
        )          # Initial state


@test fi ≈ fi2
@test qfi ≈ qfi2
# plot(t1, fi ./ t1, label= "FI")
# plot!(t1, (fi + qfi) ./ t1, label= "Eff QFI")
#
# plot!(t2, fi ./ t1, linestyle=:dash, label="FI 2")
# plot!(t2, (fi2 + qfi2) ./ t1, label= "Eff QFI 2")
