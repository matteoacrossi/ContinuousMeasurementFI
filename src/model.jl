using BlockDiagonalMatrices
using SparseArrays
using LinearAlgebra
using TimerOutputs

struct ModelParameters
    Nj::Integer
    kind::Real
    kcoll::Real
    omega::Real
    eta::Real
    dt::Real
    Tfinal::Real
    Ntime::Integer
    outpoints::Integer
    _outsteps::Integer

    function ModelParameters(; Nj::Integer=1,
                               kind::Real=1.0,
                               kcoll::Real=1.0,
                               omega::Real=1.0,
                               eta::Real=1.0,
                               dt::Real=0.0001,
                               Tfinal::Real=1.0,
                               outpoints::Integer=0)

        Ntime = Int(floor(Tfinal/dt)) # Number of timesteps

        outsteps = 1
        if outpoints > 0

            try
                outsteps = Int(round(Tfinal / dt / outpoints, digits=3))
            catch InexactError
                @warn "The requested $outpoints output points does not divide
                the total time steps. Using the full time output."
            end
        end
        #outpoints = Ntime

        new(Nj, kind, kcoll, omega, eta, dt, Tfinal, Ntime, outpoints, outsteps)
    end
end

function get_time(mp::ModelParameters)
    t = (1 : mp.Ntime) * mp.dt
    t[mp._outsteps:mp._outsteps:end]
end

struct State
    ρ::BlockDiagonal
    dρ::BlockDiagonal
    τ::BlockDiagonal
    # Internal fields
    _tmp1::BlockDiagonal
    _tmp2::BlockDiagonal
    _new_ρ::BlockDiagonal

    State(ρ::BlockDiagonal) = new(ρ, zero(ρ), zero(ρ), similar(ρ), similar(ρ), similar(ρ))
end
# Copy constructor
State(state::State) = State(copy(state.ρ))

function coherentspinstate(Nj, dense=true)
    ρ0 = blockdiagonal(css(Nj), dense=dense)
    State(ρ0)
end

struct KrausOperator
    M0::BlockDiagonal
    M::BlockDiagonal
end

struct Model
    params::ModelParameters
    Jx::BlockDiagonal
    Jy::BlockDiagonal
    Jz::BlockDiagonal
    Jx2::BlockDiagonal
    Jy2::BlockDiagonal
    Jz2::BlockDiagonal
    second_term::SuperOperator
    M0::BlockDiagonal
    dM::BlockDiagonal
end

function InitializeModel(modelparams::ModelParameters, liouvillianfile::Union{String, Nothing}=nothing)
    Nj = modelparams.Nj
    dt = modelparams.dt
    kcoll = modelparams.kcoll
    kind = modelparams.kind
    ω = modelparams.omega
    η = modelparams.eta

    # Spin operators
    (Jx, Jy, Jz) = jspin(Nj)

    # TOODO: Find better name
    second_term = (1 - η) * dt * kcoll * sup_pre_post(sparse(Jy))

    let indprepost = isnothing(liouvillianfile) ? initliouvillian(Nj) : initliouvillian(Nj, liouvillianfile)
        second_term += dt * (kind / 2) * indprepost
    end

    dropzeros!(second_term)
    second_term = SuperOperator(second_term)

    (Jx, Jy, Jz) = map(blockdiagonal, (Jx, Jy, Jz))

    Jx2 = Jx^2
    Jy2 = Jy^2
    Jz2 = Jz^2
    H = ω * Jz
    dH = Jz

    # Kraus-like operator, trajectory-independent part
    M0 = (I - 1im * H * dt -
          0.25 * dt * kind * Nj * I - # The Id comes from the squares of sigmaz_j
          (kcoll/2) * Jy2 * dt)

    # Derivative of the Kraus-like operator wrt to ω
    dM = -1im * dH * dt

    Model(modelparams, Jx, Jy, Jz, Jx2, Jy2, Jz2, second_term, M0, dM)
end

get_time(m::Model) = get_time(m.params)

function initliouvillian(Nj::Integer)
    sys = piqs.Dicke(Nj)
    sys.dephasing = 4.

    liouvillian = tosparse(sys.liouvillian())
    return liouvillian + Nj*I
end

function initliouvillian(Nj::Integer, filename::String)
    liouvillian = sparse_fromfile(filename)
    return liouvillian + Nj*I
end

function measure_current(state::State, model::Model)
    @inline dW() = sqrt(model.params.dt) * randn() # Define the Wiener increment
    # Homodyne current (Eq. 35)
    # dy = 2 sqrt(kcoll * eta) * tr(ρ * Jy) * dt + dW
    mul!(state._tmp1, model.Jy, state.ρ)
    return 2 * sqrt(model.params.kcoll * model.params.eta) * real(tr(state._tmp1)) * model.params.dt + dW()
end

function get_kraus(model::Model, dy::Real)
    p = model.params
    kraus = similar(model.M0)
    # Kraus operator Eq. (36)
    @inbounds for i in eachindex(kraus.blocks)
        kraus.blocks[i] = model.M0.blocks[i] + sqrt(p.eta * p.kcoll) * model.Jy.blocks[i] * dy +
        p.eta * (p.kcoll / 2) * model.Jy2.blocks[i] * (dy^2 - p.dt)
    end
    return kraus
end

function updatestate!(state::State, model::Model, dy::Real)
    # Get Kraus operator for measured current dy
    @timeit_debug "get_kraus" M = get_kraus(model, dy)

    # Non-allocating code for
    # new_ρ = Mpre * Mpost * ρ + second_term * ρ

    @timeit_debug "rho" begin
        mul!(state._tmp1, state.ρ, M')
        mul!(state._new_ρ, M, state._tmp1)
        @timeit_debug "superop" apply_superop!(state._tmp1, model.second_term, state.ρ)

        # TODO: Replace with broadcasting once implemented
        @inbounds for i in eachindex(state._new_ρ.blocks)
            state._new_ρ.blocks[i] .+= state._tmp1.blocks[i]
        end

        zchop!(state._new_ρ) # Round off elements smaller than 1e-14
        tr_ρ = tr(state._new_ρ)
        # Evolve the unnormalized derivative wrt ω
    end
    # Non-allocating code for
    # τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
    #      second_term * τ )/ tr_ρ;
    @timeit_debug "tau" begin
        mul!(state._tmp1, state.ρ, model.dM')
        mul!(state._tmp1, state.τ, M', 1., 1.)
        @timeit_debug "superop" apply_superop!(state._tmp2, model.second_term, state.τ)
        mul!(state._tmp2, M, state._tmp1, 1., 1.)
        mul!(state._tmp1, state.ρ, M')
        mul!(state._tmp2, model.dM, state._tmp1, 1., 1.)

        # TODO: Use broadcasting when it is implemented
        @inbounds for i in eachindex(state.τ.blocks)
            state.τ.blocks[i] .= state._tmp2.blocks[i] ./ tr_ρ
        end
    end
    zchop!(state.τ) # Round off elements smaller than 1e-14

    tr_τ = tr(state.τ)
    # Now we can renormalize ρ and its derivative wrt ω
    # TODO: Use broadcasting when it is implemented
    @timeit_debug "norms" begin
        @inbounds for i = eachindex(state.ρ.blocks)
            state.ρ.blocks[i] .= state._new_ρ.blocks[i] ./ tr_ρ
        end
        @inbounds for i = eachindex(state.dρ.blocks)
            state.dρ.blocks[i] .= state.τ.blocks[i] .- tr_τ .* state.ρ.blocks[i]
        end
    end
    return tr_ρ, tr_τ
end

expectation_value!(state::State, op::AbstractArray) = real(tr(mul!(state._tmp1, op, state.ρ)))
