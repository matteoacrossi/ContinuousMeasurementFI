using ZChop # For chopping small imaginary parts in ρ
using Distributed
using TimerOutputs
using ProgressMeter


function squeezing_param(N, ΔJ1, J2m, J3m)
    """
        ξ2 = squeezing_param(N, ΔJ1, J2m, J3m)

    Returns the squeezing parameter defined, e.g.,
    in Phys. Rev. A 65, 061801 (2002), Eq. (1).
    """
    return (J2m .^2 + J3m .^2) ./ (N * ΔJ1)
end

function density(s::SparseMatrixCSC)
    return length(s.nzval) / (s.n * s.m)
end

function Unconditional_QFI_Dicke(Nj::Int64, Tfinal::Real, dt::Real;
    κ::Real = 1.,                    # Independent noise strength
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0                   # Frequency of the Hamiltonian
    )
    return Eff_QFI_HD_Dicke(Nj, 1, Tfinal, dt; κ=κ, κcoll = κcoll, ω=ω, η=0.0)
end

"""
    (t, FI, QFI) = Eff_QFI_HD_Dicke(Nj, Ntraj, Tfinal, dt; kwargs... )

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous homodyne monitoring of
collective transverse noise, with each half-spin particle affected by
parallel noise, with efficiency η using SME
(stochastic master equation).

The function returns a tuple `(t, FI, QFI)` containing the time vector and the vectors containing the FI and average QFI

# Arguments

* `Nj`: number of spins
* `Ntraj`: number of trajectories for the SSE
* `Tfinal`: final time of evolution
* `dt`: timestep of the evolution
* `κ = 1`: the independent noise coupling
* `κcoll = 1`: the collective noise coupling
* `ω = 0`: local value of the frequency
* `η = 1`: measurement efficiency
* `outpoints = 200`: save a total of outpoints timesteps
"""
function Eff_QFI_HD_Dicke(Nj::Int64, # Number of spins
    Ntraj::Int64,                    # Number of trajectories
    Tfinal::Real,                    # Final time
    dt::Real;                        # Time step
    κ::Real = 1.,                    # Independent noise strength
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0,                   # Frequency of the Hamiltonian
    η::Real = 1.,                    # Measurement efficiency
    outpoints = 0,                 # Number of output points
    to = TimerOutput())

    @info "Eff_QFI_HD_Dicke starting"
    @info "Parameters" Nj Ntraj Tfinal dt κ κcoll ω η outpoints

    outsteps = 1

    if outpoints > 0
        try
            outsteps = Int(round(Tfinal / dt / outpoints, digits=3))
        catch InexactError
            @warn "The requested $outpoints output points does not divide
            the total time steps. Using the full time output."
        end
    end

    @info "Output every $outsteps steps"

    dW() = sqrt(dt) * randn() # Define the Wiener increment

    @timeit_debug to "Preparation" begin
        Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

        @timeit_debug to "PIQS" begin
            # Spin operators
            (Jx, Jy, Jz) = map(x-> blockdiagonal(x, dense=true), jspin(Nj))

            sys = piqs.Dicke(Nj)
            sys.dephasing = 4.

            liouvillian = tosparse(sys.liouvillian())
            indprepost = liouvillian + Nj*I

            # Initial state of the system
            # is a spin coherent state |++...++>
            ρ0 = blockdiagonal(css(Nj), dense=true)
        end

        Jx2 = Jx^2
        Jy2 = Jy^2
        Jz2 = Jz^2

        @info "Type of ρ: $(typeof(ρ0))"
        @info "Size of ρ: $(size(ρ0))"
        @info "Density of noise superoperator: $(density(indprepost))"

        @timeit_debug to "op_creation" begin
            H = ω * Jz
            dH = Jz

            # Kraus-like operator, trajectory-independent part
            M0 = blockdiagonal(I - 1im * H * dt -
                        0.25 * dt * κ * Nj * I - # The Id comes from the squares of sigmaz_j
                        (κcoll/2) * Jy2 * dt, dense=true)


            # Derivative of the Kraus-like operator wrt to ω
            dM = -1im * dH * dt
            dMt = copy(dM')

            # TODO: Find better name
            second_term = ((1 - η) * dt * κcoll * sup_pre_post(sparse(Jy)) +
                  dt * (κ/2) * indprepost)
            dropzeros!(second_term)

            t = (1 : Ntime) * dt
            t = t[outsteps:outsteps:end]

            second_term = SuperOperator(second_term)
            #indices = get_superop_indices(second_term, ρ0)
        end
    end

    println(typeof(second_term))
    # traj_count = 0
    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    @timeit_debug to "trajectories" begin
    result = @showprogress 1 "Computing..." @distributed (+) for ktraj = 1 : Ntraj
        ρ = copy(ρ0) # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dρ = zero(ρ)
        τ = zero(ρ)

        # Temporarily store the new density operator
        new_ρ = similar(ρ0)

        # Allocate auxiliary variables
        tmp1 = similar(ρ0)
        tmp2 = similar(ρ0)

        # Allocate the Kraus operator
        M = similar(M0)
        Mt = similar(M)

        # Output variables
        jx = similar(t)
        jy = similar(t)
        jz = similar(t)

        jx2 = similar(t)
        jy2 = similar(t)
        jz2 = similar(t)

        FisherT = similar(t)
        QFisherT = similar(t)

        jto = 1 # Counter for the output
        for jt = 1 : Ntime
            @timeit_debug to "current" begin
                # Homodyne current (Eq. 35)
                mul!(tmp1, Jy, ρ)
                dy = 2 * sqrt(κcoll * η) * tr(tmp1) * dt + dW()
            end

            # Kraus operator Eq. (36)
            @timeit_debug to "op_creation" begin
                for i in 1:length(M.blocks)
                    M.blocks[i] .= (M0.blocks[i] .+ sqrt(η * κcoll) .* Jy.blocks[i] .* dy .+
                                   η .* (κcoll/2) .* Jy2.blocks[i] .* (dy^2 - dt))
                end

                copy!(Mt, M')
            end

            #@info "Eigvals" eigvals(Hermitian(Matrix(reshape(ρ, size(Jx)))))[1]
            @timeit_debug to "dynamics" begin
                # Evolve the density operator
                # Non-allocating code for
                # new_ρ = Mpre * Mpost * ρ + second_term * ρ
                @timeit_debug to "rho" begin
                    mul!(tmp1, ρ, Mt)
                    mul!(new_ρ, M, tmp1)
                    @timeit_debug to "superop" apply_superop!(tmp1, second_term, ρ)
                end


                # TODO: Replace with broadcasting once implemented
                for (i, b) in enumerate(blocks(new_ρ))
                    b .+= tmp1.blocks[i]
                end

                zchop!(new_ρ) # Round off elements smaller than 1e-14
                tr_ρ = tr(new_ρ)

                # Evolve the unnormalized derivative wrt ω

                # Non-allocating code for
                # τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
                #      second_term * τ )/ tr_ρ;
                @timeit_debug to "tau" begin
                mul!(tmp1, ρ, dMt)
                mul!(tmp1, τ, Mt, 1., 1.)
                @timeit_debug to "superop" apply_superop!(tmp2, second_term, τ)
                mul!(tmp2, M, tmp1, 1., 1.)
                mul!(tmp1, ρ, Mt)
                mul!(tmp2, dM, tmp1, 1., 1.)
                end
                τ .= tmp2

                # TODO: Use broadcasting when it is implemented
                for b in blocks(τ)
                    b ./= tr_ρ
                end


                zchop!(τ) # Round off elements smaller than 1e-14

                tr_τ = tr(τ)

                # Now we can renormalize ρ and its derivative wrt ω
                @timeit_debug to "normalization" begin
                # TODO: Use broadcasting when it is implemented
                for i = 1:length(ρ.blocks)
                    ρ.blocks[i] .= new_ρ.blocks[i] ./ tr_ρ
                end
                for i = 1:length(dρ.blocks)
                    dρ.blocks[i] .= τ.blocks[i] .- tr_τ .* ρ.blocks[i]
                end
                end
            end

            if jt % outsteps == 0
                @timeit_debug to "Output" begin
                    jx[jto] = real(tr(mul!(tmp1, Jx, ρ)))
                    jy[jto] = real(tr(mul!(tmp1, Jy, ρ)))
                    jz[jto] = real(tr(mul!(tmp1, Jz, ρ)))

                    jx2[jto] = real(tr(mul!(tmp1, Jx2, ρ)))
                    jy2[jto] = real(tr(mul!(tmp1, Jy2, ρ)))
                    jz2[jto] = real(tr(mul!(tmp1, Jz2, ρ)))

                    # We evaluate the classical FI for the continuous measurement
                    FisherT[jto] = real(tr_τ^2)

                    # We evaluate the QFI for a final strong measurement done at time t
                    @timeit_debug to "QFI" QFisherT[jto] = QFI(ρ, dρ)

                    jto += 1
                end
            end

        end

        Δjx2 = jx2 - jx.^2
        Δjy2 = jy2 - jy.^2
        Δjz2 = jz2 - jz.^2

        xi2x = squeezing_param(Nj, Δjx2, jy, jz)
        xi2y = squeezing_param(Nj, Δjy2, jx, jz)
        xi2z = squeezing_param(Nj, Δjz2, jx, jy)

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT, jx, jy, jz, Δjx2, Δjy2, Δjz2, xi2x, xi2y, xi2z)
    end
    end

    # Evaluate averages
    jx = result[:, 3] / Ntraj
    jy = result[:, 4] / Ntraj
    jz = result[:, 5] / Ntraj

    Δjx2 = result[:, 6] / Ntraj
    Δjy2 = result[:, 7] / Ntraj
    Δjz2 = result[:, 8] / Ntraj

    xi2x = result[:, 9] / Ntraj
    xi2y = result[:, 10] / Ntraj
    xi2z = result[:, 11] / Ntraj

    @info "Time details \n$to"
    return (t=t,
            FI=result[:,1] / Ntraj,
            QFI=result[:,2] / Ntraj,
            jx=jx, jy=jy, jz=jz,
            Δjx=Δjx2, Δjy=Δjy2, Δjz=Δjz2,
            xi2x=xi2x, xi2y=xi2y, xi2z=xi2z)
end



"""
    (t, FI, QFI) = Eff_QFI_HD_Dicke(Nj, Ntraj, Tfinal, dt; kwargs... )

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous homodyne monitoring of
collective transverse noise, with each half-spin particle affected by
parallel noise, with efficiency η using SME
(stochastic master equation).

The function returns a tuple `(t, FI, QFI)` containing the time vector and the vectors containing the FI and average QFI

# Arguments

* `Nj`: number of spins
* `Ntraj`: number of trajectories for the SSE
* `Tfinal`: final time of evolution
* `dt`: timestep of the evolution
* `κ = 1`: the independent noise coupling
* `κcoll = 1`: the collective noise coupling
* `ω = 0`: local value of the frequency
* `η = 1`: measurement efficiency
* `outpoints = 200`: save a total of outpoints timesteps
"""
function Eff_QFI_HD_Dicke_0(Nj::Int64, # Number of spins
    Ntraj::Int64,                    # Number of trajectories
    Tfinal::Real,                    # Final time
    dt::Real;                        # Time step
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0,                   # Frequency of the Hamiltonian
    η::Real = 1.,                    # Measurement efficiency
    outpoints = 0,                 # Number of output points
    to = TimerOutput())

    @info "Eff_QFI_HD_Dicke starting"

    κ = 0.0
    @info "Parameters" Nj Ntraj Tfinal dt κ κcoll ω η outpoints

    outsteps = 1

    if outpoints > 0
        try
            outsteps = Int(round(Tfinal / dt / outpoints, digits=3))
        catch InexactError
            @warn "The requested $outpoints output points does not divide
            the total time steps. Using the full time output."
        end
    end

    @info "Output every $outsteps steps"

    dW() = sqrt(dt) * randn() # Define the Wiener increment

    @timeit_debug to "Preparation" begin
        Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

        @timeit_debug to "PIQS" begin
            # Spin operators
            (Jx, Jy, Jz) = map(blockdiagonal, jspin(Nj))

            sys = piqs.Dicke(Nj)
            sys.dephasing = 4.

            #liouvillian = tosparse(sys.liouvillian())
            #indprepost = liouvillian + Nj*I

            # Initial state of the system
            # is a spin coherent state |++...++>
            ρ0 = blockdiagonal(css(Nj), dense=true)
        end

        ρ0 = ρ0.blocks[1]

        Jx = Jx.blocks[1]
        Jy = Jy.blocks[1]
        Jz = Jz.blocks[1]

        Jx2 = Jx^2
        Jy2 = Jy^2
        Jz2 = Jz^2


        @info "Type of ρ: $(typeof(ρ0))"
        @info "Size of ρ: $(size(ρ0))"
        #@info "Density of noise superoperator: $(density(indprepost))"

        @timeit_debug to "op_creation" begin
            H = ω * Jz
            dH = Jz

            # Kraus-like operator, trajectory-independent part
            M0 = (I - 1im * H * dt -
                        0.25 * dt * κ * Nj * I - # The Id comes from the squares of sigmaz_j
                        (κcoll/2) * Jy2 * dt)


            # Derivative of the Kraus-like operator wrt to ω
            dM = -1im * dH * dt

            # TODO: Find better name
            # second_term = (1 - η) * dt * κcoll * sup_pre_post(sparse(Jy)) +
                 # dt * (κ/2) * indprepost)
           # dropzeros!(second_term)

            t = (1 : Ntime) * dt
            t = t[outsteps:outsteps:end]

            # second_term = SuperOperator(second_term)
            #indices = get_superop_indices(second_term, ρ0)
        end
    end

    # println(typeof(second_term))
    # traj_count = 0
    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    @timeit_debug to "trajectories" begin
    result = @showprogress 1 "Computing..." @distributed (+) for ktraj = 1 : Ntraj
        ρ = copy(ρ0) # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dρ = zero(ρ)
        τ = zero(ρ)

        # Temporarily store the new density operator
        new_ρ = similar(ρ0)

        # Allocate auxiliary variables
        tmp1 = similar(ρ0)
        tmp2 = similar(ρ0)
        tmp3 = similar(ρ0)

        # Temporary operator in order to allocate Mpre
        # and Mpost
        M = (M0 + sqrt(η * κcoll) * Jy * 1. +
                    η * (κcoll/2) * Jy2 * (1. ^2 - dt))

        # Output variables
        jx = similar(t)
        jy = similar(t)
        jz = similar(t)

        jx2 = similar(t)
        jy2 = similar(t)
        jz2 = similar(t)

        FisherT = similar(t)
        QFisherT = similar(t)

        jto = 1 # Counter for the output
        for jt = 1 : Ntime
            @timeit_debug to "current" begin
                # Homodyne current (Eq. 35)
                mul!(tmp1, Jy, ρ)
                dy = 2 * sqrt(κcoll * η) * tr(tmp1) * dt + dW()
            end

            # Kraus operator Eq. (36)
            @timeit_debug to "op_creation" begin
                M = (M0 + sqrt(η * κcoll) * Jy * dy +
                    η * (κcoll/2) * Jy2 * (dy^2 - dt))
            end

            #@info "Eigvals" eigvals(Hermitian(Matrix(reshape(ρ, size(Jx)))))[1]
            @timeit_debug to "dynamics" begin
                # Evolve the density operator
                # Non-allocating code for
                # new_ρ = Mpre * Mpost * ρ + second_term * ρ
                mul!(tmp1, ρ, M')
                mul!(new_ρ, M, tmp1)

                mul!(tmp1, (1 - η) * dt * κcoll * Jy, ρ)
                mul!(new_ρ, tmp1, Jy', 1., 1.)

                zchop!(new_ρ) # Round off elements smaller than 1e-14
                tr_ρ = tr(new_ρ)

                # Evolve the unnormalized derivative wrt ω

                # Non-allocating code for
                # τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
                #      second_term * τ )/ tr_ρ;
                mul!(tmp1, ρ, dM')
                mul!(tmp1, τ, M', 1., 1.)

                mul!(tmp3, (1 - η) * dt * κcoll * Jy, ρ)
                mul!(tmp2, tmp3, Jy')

                #apply_superop!(tmp2, second_term, τ)
                mul!(tmp2, M, tmp1, 1., 1.)
                mul!(tmp1, ρ, M')
                mul!(tmp2, dM, tmp1, 1., 1.)
                τ .= tmp2
                τ ./= tr_ρ



                zchop!(τ) # Round off elements smaller than 1e-14

                tr_τ = tr(τ)

                # Now we can renormalize ρ and its derivative wrt ω
                ρ .= new_ρ ./ tr_ρ
                dρ .= τ .- tr_τ .* ρ
            end

            if jt % outsteps == 0
                @timeit_debug to "Output" begin
                    jx[jto] = real(tr(mul!(tmp1, Jx, ρ)))
                    jy[jto] = real(tr(mul!(tmp1, Jy, ρ)))
                    jz[jto] = real(tr(mul!(tmp1, Jz, ρ)))

                    jx2[jto] = real(tr(mul!(tmp1, Jx2, ρ)))
                    jy2[jto] = real(tr(mul!(tmp1, Jy2, ρ)))
                    jz2[jto] = real(tr(mul!(tmp1, Jz2, ρ)))

                    # We evaluate the classical FI for the continuous measurement
                    FisherT[jto] = real(tr_τ^2)

                    # We evaluate the QFI for a final strong measurement done at time t
                    @timeit_debug to "QFI" QFisherT[jto] = QFI(ρ, dρ)

                    jto += 1
                end
            end

        end

        Δjx2 = jx2 - jx.^2
        Δjy2 = jy2 - jy.^2
        Δjz2 = jz2 - jz.^2

        xi2x = squeezing_param(Nj, Δjx2, jy, jz)
        xi2y = squeezing_param(Nj, Δjy2, jx, jz)
        xi2z = squeezing_param(Nj, Δjz2, jx, jy)

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT, jx, jy, jz, Δjx2, Δjy2, Δjz2, xi2x, xi2y, xi2z)
    end
    end

    # Evaluate averages
    jx = result[:, 3] / Ntraj
    jy = result[:, 4] / Ntraj
    jz = result[:, 5] / Ntraj

    Δjx2 = result[:, 6] / Ntraj
    Δjy2 = result[:, 7] / Ntraj
    Δjz2 = result[:, 8] / Ntraj

    xi2x = result[:, 9] / Ntraj
    xi2y = result[:, 10] / Ntraj
    xi2z = result[:, 11] / Ntraj

    @info "Time details \n$to"
    return (t=t,
            FI=result[:,1] / Ntraj,
            QFI=result[:,2] / Ntraj,
            jx=jx, jy=jy, jz=jz,
            Δjx=Δjx2, Δjy=Δjy2, Δjz=Δjz2,
            xi2x=xi2x, xi2y=xi2y, xi2z=xi2z)
end
