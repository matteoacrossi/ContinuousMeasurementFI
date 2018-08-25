using ZChop # For chopping small imaginary parts in ρ
using Distributed

include("NoiseOperators.jl")
include("States.jl")
include("Fisher.jl")

"""
    (t, FI, QFI) = Eff_QFI_PD_pure(Nj, Ntraj, Tfinal, dt; kwargs... )

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous photo-detection monitoring
of each half-spin particle affected by noise at an angle θ,
with efficiency η = 1 using SSE (Stochastic Schrödinger equation).

The function returns a tuple `(t, FI, QFI)` containing the time vector and the
vectors containing the FI and average QFI

# Arguments

* `Nj`: number of spins
* `Ntraj`: number of trajectories for the SSE
* `Tfinal`: final time of evolution
* `dt`: timestep of the evolution
* `κ = 1`: the noise coupling
* `θ = 0`: noise angle (0 parallel, π/2 transverse)
* `ω = 0`: local value of the frequency
"""
function Eff_QFI_PD_pure(Nj::Int64,     # Number of spins
    Ntraj::Int64,                       # Number of trajectories
    Tfinal::Float64,                    # Final time
    dt::Float64;                        # Time step
    κ = 1.,                             # Noise coupling
    θ = 0.,                             # Noise angle
    ω = 0)                              # Frequency

    Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

    dimJ = Int(2^Nj)   # Dimension of the corresponding Hilbert space

    # Define an array of noise channels (they are all sparse matrices)
    cj = sqrt(κ/2) *
        [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:x, j, Nj))
            for j = 1:Nj]

    # Hamiltonian of the Nj-atom system
    H = ω * σ(:z, Nj) / 2
    dH = σ(:z, Nj) / 2        # Derivative of H wrt the parameter ω

    dW() = sqrt(dt) * randn(Nj) # Function that returns a Wiener increment vector

    # Kraus-like operator, trajectory-independent part
    # see Eqs. (50-51)
    M0 = sparse(I - 1im * H * dt - 0.5 * dt * sum([c'*c for c in cj]) )
    M1 = sqrt(dt) * cj

    # Derivative of the Kraus-like operator wrt to ω
    dM0 = - 1im * dH * dt
    dM1 = 0

    # Probability of detecting a single photon (the probability of
    # detecting more than one in dt is negligible)
    # p_PD = η * sum(tr(ρ cj'*cj)) *dt,
    # but each noise operator gives 1/2 * κ * id when squared
    pPD = 0.5 * Nj * κ * dt;

    # Initial state of the dynamics
    ψ0 = ghz_state(Nj)

    t = (1 : Ntime) * dt

    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    # Uses Julia @distributed macro for simple parallelization
    result = @distributed (+) for ktraj = 1 : Ntraj

        ψ = copy(ψ0) # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dψ = zero(ψ)
        ϕ = zero(ψ)

        # Vectors for the FI and QFI for the specific trajectory
        FisherT = zero(t)
        QFisherT = zero(t)

        for jt=1:Ntime
            # Has the photon been detected?
            if (rand() < real(pPD)) # Detected
                # Choose randomly which channel detected the photon
                # (with equal probabiltiy)
                ch = rand(1:Nj)

                new_ψ = M1[ch] * ψ

                norm_ψ = norm(new_ψ)

                ϕ = (dM1 * ψ + M1[ch] * ϕ) / norm_ψ
            else # Not detected
                new_ψ = M0 * ψ

                norm_ψ = norm(new_ψ)

                ϕ = (dM0 * ψ + M0 * ϕ) / norm_ψ
            end

            # Now we can renormalize ψ and evaluate its derivative wrt ω
            ψ = new_ψ / norm_ψ;
            dψ = ϕ - 0.5 * (ϕ' * ψ + ψ' * ϕ) * ψ;

            # We evaluate the classical FI for the continuous measurement
            FisherT[jt] = real((ψ' * ϕ + ϕ' * ψ)^2);

            # We evaluate the QFI for a final strong measurement done at time t
            QFisherT[jt] = 4 * real( dψ' * dψ + (dψ' * ψ)^2);
        end

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT)
    end

    return (t, result[:,1] / Ntraj, result[:,2] / Ntraj)
end
