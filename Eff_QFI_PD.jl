using ZChop # For chopping small imaginary parts in ρ

include("NoiseOperators.jl")
include("States.jl")
include("Fisher.jl")

"""
    (t, FI, QFI) = Eff_QFI_PD(Nj, Ntraj, Tfinal, dt; kwargs... )

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous photo-detection monitoring
of each half-spin particle affected by noise at an angle θ,
with efficiency η using SME (stochastic master equation).

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
* `η = 1`: measurement efficiency
"""
function Eff_QFI_PD(Ntraj::Int64,       # Number of trajectories
    Tfinal::Number,                     # Final time
    dt::Number,                         # Time step
    H, dH,                              # Hamiltonian and its derivative wrt ω
    non_monitored_noise_op,             # Non monitored noise operators
    monitored_noise_op;                 # Monitored noise operators
    initial_state = ghz_state,          # Initial state
    η = 1.)                             # Measurement efficiency

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
    M0 = I - 1im * H * dt
                - 0.5 * dt * sum([c'*c for c in cj])
                - 0.5 * dt * sum([C'*C for C in Cj])

    M1 = sqrt(η * dt) * Cj

    # Derivative of the Kraus-like operator wrt to ω
    dM0 = - 1im * dH * dt
    dM1 = 0

    # Probability of detecting a single photon (the probability of
    # detecting more than one in dt is negligible)
    # p_PD = η * sum(tr(ρ cj'*cj)) * dt,
    # but each noise operator gives 1/2 * κ * id when squared
    #pPD = 0.5 * Nm * η * κ * dt;

    # Initial state of the system
    ψ0 = initial_state(Nj)
    ρ0 = ψ0 * ψ0'

    # C' * C is actually unitary up to some constant factor, that is what we are
    # interested in
    pPD = η * sum([trace(ρ0 * C' * C) for C in Cj]) * dt

    t = (1 : Ntime) * dt

    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    result = @parallel (+) for ktraj = 1 : Ntraj
        ρ = ρ0 # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dρ = zeros(ρ)
        τ = dρ

        FisherT  = zeros(t)
        QFisherT = zeros(t)

        for jt=1:Ntime
            # Has the photon been detected?
            if (rand() < real(pPD)) # Detected
                # Choose randomly which channel detected the photon
                # (with equal probabiltiy)
                ch = rand(1:Nm)

                new_ρ = M1[ch] * ρ * M1[ch]' ;

                zchop!(new_ρ) # Round off elements smaller than 1e-14
                tr_ρ = real(trace(new_ρ));

                τ = (M1[ch] * (τ * M1[ch]' + ρ * dM1') + dM1 * ρ * M1[ch]') / tr_ρ

            else # Not detected
                new_ρ = M0 * ρ * M0' + (1 - η) * dt * sum([c * ρ * c' for c in Cj])
                zchop!(new_ρ)

                tr_ρ = real(trace(new_ρ));

                τ = (M0 * (τ * M0' + ρ * dM0') + dM0 * ρ * M0' +
                       (1 - η)* dt * sum([c* τ * c' for c in Cj]))/ tr_ρ;
            end

            zchop!(τ) # Round off elements smaller than 1e-14

            tr_τ = trace(τ)

            # Now we can renormalize ρ and its derivative wrt ω
            ρ = new_ρ / tr_ρ
            dρ = τ - tr_τ * ρ

            # We evaluate the classical FI for the continuous measurement
            FisherT[jt] = real(tr_τ^2)

            # We evaluate the QFI for a final strong measurement done at time t
            QFisherT[jt] = QFI(ρ, dρ)
        end

        # Use the reduction feature of @parallel for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT)
    end

    return (t, result[:,1] / Ntraj, result[:,2] / Ntraj)
end
