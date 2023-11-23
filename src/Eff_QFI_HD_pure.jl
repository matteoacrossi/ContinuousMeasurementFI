using ZChop # For chopping small imaginary parts in ρ
using Distributed

"""
    (t, FI, QFI) = Eff_QFI_HD_pure(Nj, Ntraj, Tfinal, dt; kwargs... )

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous homodyne monitoring of each half-spin
particle affected by noise at an angle θ, with efficiency η = 1 using SSE
(Stochastic Schrödinger equation).

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
function Eff_QFI_HD_pure(Nj::Int64,     # Number of spins
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

    # Store the product of operators for speed
    cjProd = [cj[i]*cj[j] for i in eachindex(cj), j in eachindex(cj)]

    # Hamiltonian of the Nj-atom system
    H = ω * σ(:z, Nj) / 2
    dH = σ(:z, Nj) / 2        # Derivative of H wrt the parameter ω

    dW() = sqrt(dt) * randn(Nj) # Function that returns a Wiener increment vector

    # Kraus-like operator, trajectory-independent part
    M0 = I - 1im * H * dt -
                0.5 * dt * sum([c' * c for c in cj])

    # Initialize the Kraus-like operator
    M = similar(M0)

    # Derivative of the Kraus-like operator wrt to ω (also not dependent on
    # the trajectory)
    dM = -1im * dH * dt

    # Initial state of the dynamics
    ψ0 = ghz_state(Nj)

    # Output time vector
    t = (1 : Ntime) * dt

    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    # Uses Julia @distributed macro for simple parallelization
    result = @distributed (+) for ktraj = 1 : Ntraj

        ψ = copy(ψ0)
        # Derivative of ψ wrt the parameter
        # (Initial state does not depend on the paramter)
        dψ = zero(ψ0)

        ϕ = zero(ψ0)

        # Vectors for the FI and QFI for the specific trajectory
        FisherT = similar(t)
        QFisherT = similar(t)

        for jt=1:Ntime
            # Homodyne current (Eq. 35) (! c† = c)
            dy = [ψ' * ( c + c' ) * ψ * dt for c in cj] + dW()

            # Kraus operator (Eq. 36)
            M = M0
            for i = 1 : Nj
                M += cj[i] * dy[i]
                M += 0.5 * cjProd[i,i] * (dy[i]^2 - dt)
            end

            # We only have to evaluate half of the products
            # because ci . cj == cj . ci (they act on different spins)
            for i = 1 : Nj
                for j = i + 1 : Nj
                    M += cjProd[i,j] * (dy[i] * dy[j])
                end
            end

            # Evolve the density operator
            new_ψ = M * ψ
            norm_ψ = norm(new_ψ)

            # Evolution of the unnormalized derivative wrt ω (Eq. 45)
            ϕ = (dM * ψ + M * ϕ) / norm_ψ;

            # Now we can renormalize ψ and evaluate its derivative wrt ω
            ψ = new_ψ / norm_ψ;
            dψ = ϕ - 0.5 * (ϕ' * ψ + ψ' * ϕ) * ψ;  # Eq.(46)

            # We evaluate the classical FI for the continuous measurement
            FisherT[jt] = real((ψ' * ϕ + ϕ' * ψ)^2); # Eqs. (43) and (30)

            # We evaluate the QFI for a final strong measurement done at time t
            QFisherT[jt] = 4 * real( dψ' * dψ + (dψ' * ψ)^2); # Eq. (47)
        end

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT)
    end

    return (t, result[:,1] / Ntraj, result[:,2] / Ntraj)
end
