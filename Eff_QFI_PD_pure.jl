using ZChop # For chopping small imaginary parts in ρ

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
function Eff_QFI_PD_pure(Ntraj::Int64,  # Number of trajectories
    Tfinal::Number,                     # Final time
    dt::Number,                         # Time step
    H, dH,                              # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,             # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = ghz_state)          # Initial state

    Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

    dimJ = size(H, 1)       # Dimension of the corresponding Hilbert space
    Nj = Int(log2(dimJ))    # Number of spins


    # Non-monitored noise operators
    # cj = [] if all the noise is monitored
    cj = non_monitored_noise_op
    Nnm = length(cj)

    # Monitored noise operators
    Cj = monitored_noise_op
    Nm = length(Cj)

    # Kraus-like operator, trajectory-independent part
    # see Eqs. (50-51)
    M0 = I - 1im * H * dt
            - 0.5 * dt * sum([c'*c for c in cj])
            - 0.5 * dt * sum([C'*C for C in Cj])

    M1 = sqrt(dt) * Cj

    # Derivative of the Kraus-like operator wrt to ω
    dM0 = - 1im * dH * dt
    dM1 = 0

    # Probability of detecting a single photon (the probability of
    # detecting more than one in dt is negligible)
    # p_PD = η * sum(tr(ρ cj'*cj)) *dt,
    # but each noise operator gives 1/2 * κ * id when squared

    # Initial state of the dynamics
    ψ0 = initial_state(Nj)

    pPD = sum([ψ0' * C' * C * ψ0 for C in Cj]) * dt

    t = (1 : Ntime) * dt

    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    # Uses Julia @parallel macro for simple parallelization
    result = @parallel (+) for ktraj = 1 : Ntraj

        ψ = copy(ψ0) # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dψ = zeros(ψ)
        ϕ = zeros(ψ)

        # Vectors for the FI and QFI for the specific trajectory
        FisherT = zeros(t)
        QFisherT = zeros(t)

        for jt=1:Ntime
            # Has the photon been detected?
            if (rand() < real(pPD)) # Detected
                # Choose randomly which channel detected the photon
                # (with equal probabiltiy)
                ch = rand(1:Nm)

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

        # Use the reduction feature of @parallel for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT)
    end

    return (t, result[:,1] / Ntraj, result[:,2] / Ntraj)
end
