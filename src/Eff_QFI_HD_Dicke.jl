using ZChop # For chopping small imaginary parts in ρ
using Distributed
using TimerOutputs

 
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
"""
function Eff_QFI_HD_Dicke(Nj::Int64, # Number of spins
    Ntraj::Int64,                    # Number of trajectories
    Tfinal::Real,                    # Final time
    dt::Real;                        # Time step
    κ::Real = 1.,                    # Independent noise strength
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0,                   # Frequency of the Hamiltonian
    η::Real = 1.)                    # Measurement efficiency

    to = TimerOutput()
    @timeit to "Preparation" begin    
    Ntime = Int(floor(Tfinal/dt)) # Number of timesteps
    
    @timeit to "PIQS" begin
        # Spin operators
        (Jx, Jy, Jz) = tosparse.( piqs.jspin(Nj))

        sys = piqs.Dicke(Nj)
        sys.dephasing = 4.
        
        liouvillian = tosparse(sys.liouvillian())
        indprepost = liouvillian + Nj*I

        ρ0 = Matrix(tosparse(piqs.css(Nj)))[:]
    end

    Jx2 = Jx^2
    Jy2 = Jy^2
    Jz2 = Jz^2
    @timeit to "op creation" begin
    Jyprepost = sup_pre_post(Jy)
    Jypre = sup_pre(Jy)

    dW() = sqrt(dt) * randn() # Define the Wiener increment

    H = ω * Jz
    dH = Jz

    # Kraus-like operator, trajectory-independent part
    M0 = sparse(I - 1im * H * dt -
                0.25 * dt * κ * Nj * I - # The Id comes from the squares of sigmaz_j
                (κcoll/2) * Jy2 * dt)

    # Derivative of the Kraus-like operator wrt to ω
    dM = -1im * dH * dt

    dMpre = sup_pre(dM)
    dMpost = sup_post(dM')

    # Initial state of the system
    # is a spin coherent state |++...++>
    
    t = (1 : Ntime) * dt
    end
    end
    # Run evolution for each trajectory, and build up the average
    # for FI and final strong measurement QFI
    @timeit to "trajectories" begin
    result = @distributed (+) for ktraj = 1 : Ntraj
        ρ = ρ0 # Assign initial state to each trajectory
        
        # Derivative of ρ wrt the parameter
        # Initial state does not depend on the paramter
        dρ = zero(ρ)
        τ = dρ

        jx = similar(t)
        jy = similar(jx)
        jz = similar(jx)

        jx2 = similar(jx)
        jy2 = similar(jy)
        jz2 = similar(jz)

        # Vectors for the FI and QFI for each trajectory
        FisherT = zero(t)
        QFisherT = zero(t)

        for jt = 1 : Ntime

            # Homodyne current (Eq. 35)
            @timeit to "current" dy = 2 * sqrt(κcoll * η) * trace(Jxpre*ρ) * dt + dW()
            
            # Kraus operator Eq. (36)
            @timeit to "op creation" begin
                M = (M0 + sqrt(η * κcoll) * Jy * dy +
                    η * (κcoll/2) * Jy2 * (dy^2 - dt))
            end
            @timeit to "sup creation" Mpre = sup_pre(M)
            @timeit to "sup creation" Mpost = sup_post(M')

            @timeit to "Exp values" begin
                jx[jt] = real(trace(sup_pre(Jx) * ρ))
                jy[jt] = real(trace(sup_pre(Jy) * ρ))
                jz[jt] = real(trace(sup_pre(Jz) * ρ))

                jx2[jt] = real(trace(sup_pre(Jx2) * ρ))
                jy2[jt] = real(trace(sup_pre(Jy2) * ρ))
                jz2[jt] = real(trace(sup_pre(Jz2) * ρ))
            end
            #@info "Eigvals" eigvals(Hermitian(Matrix(reshape(ρ, size(Jx)))))[1]
            @timeit to "dynamics" begin
                # Evolve the density operator
                new_ρ = (Mpre * Mpost * ρ +
                        (1 - η) * dt * κcoll * Jyprepost * ρ +
                        dt * (κ/2) * indprepost * ρ)
                
                zchop!(new_ρ) # Round off elements smaller than 1e-14

                tr_ρ = trace(new_ρ)
                #@info "tr_rho" tr_ρ
                # Evolve the unnormalized derivative wrt ω            
                τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
                    (1 - η) * dt * κcoll * Jyprepost * τ +
                    dt * (κ/2) * indprepost * τ )/ tr_ρ;

                zchop!(τ) # Round off elements smaller than 1e-14

                tr_τ = trace(τ)

                # Now we can renormalize ρ and its derivative wrt ω
                ρ = new_ρ / tr_ρ
                dρ = τ - tr_τ * ρ
            end
            # We evaluate the classical FI for the continuous measurement
            FisherT[jt] = real(tr_τ^2)

            # We evaluate the QFI for a final strong measurement done at time t
            @timeit to "QFI" QFisherT[jt] = QFI(reshape(ρ, size(Jy)), reshape(dρ, size(Jy)))
        end

        # Use the reduction feature of @distributed for
        # (at the end of each cicle, sum the result to result)
        hcat(FisherT, QFisherT, jx, jy, jz, jx2, jy2, jz2)
    end
    end
    jx=result[:,3] / Ntraj 
    jy=result[:,4] / Ntraj 
    jz=result[:, 5] / Ntraj
    Δjx=result[:,6] / Ntraj - jx.^2
    Δjy=result[:,7] / Ntraj - jy.^2
    Δjz=result[:,8] / Ntraj - jz.^2

    return (t=t, 
            FI=result[:,1] / Ntraj, 
            QFI=result[:,2] / Ntraj, timer=to,
            jx=jx, jy=jy, jz=jz,
            Δjx=Δjx, Δjy=Δjy, Δjz=Δjz)
end
