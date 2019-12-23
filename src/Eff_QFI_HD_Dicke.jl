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

function density(s)
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
* `outsteps = 1`: save output every outsteps (QFI is expensive!)
"""
function Eff_QFI_HD_Dicke(Nj::Int64, # Number of spins
    Ntraj::Int64,                    # Number of trajectories
    Tfinal::Real,                    # Final time
    dt::Real;                        # Time step
    κ::Real = 1.,                    # Independent noise strength
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0,                   # Frequency of the Hamiltonian
    η::Real = 1.,                    # Measurement efficiency
    outsteps = 1,
    to = TimerOutput())

    @info "Eff_QFI_HD_Dicke starting"
    @info "Parameters" Nj Ntraj Tfinal dt κ κcoll ω η

    dW() = sqrt(dt) * randn() # Define the Wiener increment

    @timeit_debug to "Preparation" begin
        Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

        @timeit_debug to "PIQS" begin
            # Spin operators
            (Jx, Jy, Jz) = tosparse.( piqs.jspin(Nj))

            sys = piqs.Dicke(Nj)
            sys.dephasing = 4.

            liouvillian = tosparse(sys.liouvillian())
            indprepost = liouvillian + Nj*I


            # Initial state of the system
            # is a spin coherent state |++...++>
            ρ0 = Matrix(tosparse(piqs.css(Nj)))[:]
        end

        Jx2 = Jx^2
        Jy2 = Jy^2
        Jz2 = Jz^2

        @info "Size of ρ: $(length(ρ0))"
        @info "Density of noise superoperator: $(density(indprepost))"

        @timeit_debug to "op creation" begin
            Jyprepost = sup_pre_post(Jy)

            Jxpre = sup_pre(Jx)
            Jypre = sup_pre(Jy)
            Jzpre = sup_pre(Jz)

            Jx2pre = sup_pre(Jx2)
            Jy2pre = sup_pre(Jy2)
            Jz2pre = sup_pre(Jz2)

            H = ω * Jz
            dH = Jz

            # Kraus-like operator, trajectory-independent part
            M0 = sparse(I - 1im * H * dt -
                        0.25 * dt * κ * Nj * I - # The Id comes from the squares of sigmaz_j
                        (κcoll/2) * Jy2 * dt)

            @info "Density of M0: $(density(M0))"

            # Derivative of the Kraus-like operator wrt to ω
            dM = -1im * dH * dt

            dMpre = sup_pre(dM)
            dMpost = sup_post(dM)

            # TODO: Find better name
            second_term = ((1 - η) * dt * κcoll * Jyprepost +
                  dt * (κ/2) * indprepost)

            t = (1 : Ntime) * dt
            t = t[outsteps:outsteps:end]
        end
    end

    # traj_count = 0
    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    @timeit_debug to "trajectories" begin
    result = @showprogress 1 "Computing..." @distributed (+) for ktraj = 1 : Ntraj
        ρ = ρ0 # Assign initial state to each trajectory

        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dρ = zero(ρ)
        τ = zero(ρ)

        # Temporarily store the new density operator
        new_ρ = similar(ρ0)

        # Allocate auxiliary variables
        tmp1 = similar(ρ0)
        tmp2 = similar(ρ0)

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
            mul!(tmp1, Jypre, ρ)
            dy = 2 * sqrt(κcoll * η) * trace(tmp1) * dt + dW()
            end
            # Kraus operator Eq. (36)
            @timeit_debug to "op creation" begin
                M = (M0 + sqrt(η * κcoll) * Jy * dy +
                    η * (κcoll/2) * Jy2 * (dy^2 - dt))
            end

            @timeit_debug to "sup creation" begin
                Mpre = sup_pre(M)
                Mpost = sup_post(M)
            end

            #@info "Eigvals" eigvals(Hermitian(Matrix(reshape(ρ, size(Jx)))))[1]
            @timeit_debug to "dynamics" begin
                # Evolve the density operator
                # Non-allocating code for
                # new_ρ = Mpre * Mpost * ρ + second_term * ρ
                mul!(tmp1, Mpost, ρ)
                mul!(new_ρ, Mpre, tmp1)
                mul!(new_ρ, second_term, ρ, 1., 1.)

                zchop!(new_ρ) # Round off elements smaller than 1e-14

                tr_ρ = trace(new_ρ)

                # Evolve the unnormalized derivative wrt ω

                # Non-allocating code for
                # τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
                #      tmp * τ )/ tr_ρ;
                mul!(tmp1, Mpost, τ)
                mul!(tmp2, Mpre, tmp1)
                mul!(tmp2, second_term, τ, 1., 1.)

                mul!(tmp1, dMpost, ρ)
                mul!(τ, Mpre, tmp1)
                τ .+= tmp2

                mul!(tmp1, Mpost, ρ)
                mul!(tmp2, dMpre, tmp1)
                τ .+= tmp2
                τ ./= tr_ρ

                zchop!(τ) # Round off elements smaller than 1e-14

                tr_τ = trace(τ)

                # Now we can renormalize ρ and its derivative wrt ω
                ρ .= new_ρ ./ tr_ρ
                dρ .= τ .- tr_τ .* ρ
            end

            if jt % outsteps == 0
                @timeit_debug to "Output" begin
                    jx[jto] = real(trace(Jxpre * ρ))
                    jy[jto] = real(trace(Jypre * ρ))
                    jz[jto] = real(trace(Jzpre * ρ))

                    jx2[jto] = real(trace(Jx2pre * ρ))
                    jy2[jto] = real(trace(Jy2pre * ρ))
                    jz2[jto] = real(trace(Jz2pre * ρ))

                    # We evaluate the classical FI for the continuous measurement
                    FisherT[jto] = real(tr_τ^2)
                    # We evaluate the QFI for a final strong measurement done at time t
                    @timeit_debug to "QFI" QFisherT[jto] = QFI(reshape(ρ, size(Jy)), reshape(dρ, size(Jy)))

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

        # traj_count += 1
        # if traj_count % 10 == 0
        #     @info "$(traj_count) trajectories done"
        # end

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT, jx, jy, jz, Δjx2, Δjy2, Δjz2, xi2x, xi2y, xi2z)
    end
    end

    jx = result[:,3] / Ntraj
    jy = result[:,4] / Ntraj
    jz = result[:,5] / Ntraj

    Δjx2 = result[:,6] / Ntraj
    Δjy2 = result[:,7] / Ntraj
    Δjz2 = result[:,8] / Ntraj

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
