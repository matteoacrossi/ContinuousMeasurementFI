include("NoiseOperators.jl")
include("Eff_QFI_HD_pure.jl")
include("Eff_QFI_PD_pure.jl")
include("States.jl")
include("Unconditional_QFI.jl")
include("Uncond_qfi_transverse.jl")
include("Uncond_qfi_parallel.jl")

using Plots

κ_ind = 1.
κ_coll = 1.
θ = 0.
ω = 1.
Nj = 2

# ω is the parameter that we want to estimate
H = ω * σ(:z, Nj) / 2       # Hamiltonian of the spin system

dH = σ(:z, Nj) / 2          # Derivative of H wrt the parameter ω

#non_monitored_noise_op = []
non_monitored_noise_op = (sqrt(κ_ind/2) *
    [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:x, j, Nj)) for j = 1:Nj])

monitored_noise_op = [sqrt(κ_coll/2) * σ(:x, Nj)]

@time t1, fi, qfi = Eff_QFI_PD(100,# Number of trajectories
    5,                                  # Final time
    .01,                               # Time step
    H, dH,                          # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,           # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = ghz_state
    )          # Initial state

@time t2, u_qfi = Unconditional_QFI(
    5,                                 # Final time
    .001,                                # Time step
    H, dH,                              # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,             # Non monitored noise operators
    initial_state = ghz_state
    )



plot(t1, fi ./ t1, label= "FI")
plot!(t1, (fi + qfi) ./ t1, label= "Eff QFI")
plot!(t2, u_qfi ./ t2, color="black", label="Unc.")
plot!(t2, Uncond_qfi_parallel(t2, Nj, κ_ind, ω) ./ t2,color="black" , linestyle=:dash, label="Analytic Unc.")