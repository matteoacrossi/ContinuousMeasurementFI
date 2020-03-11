using BlockDiagonals
using SparseArrays
using LinearAlgebra

struct ModelParameters
    Nj::Integer
    kind::Real
    kcoll::Real
    omega::Real
    eta::Real
    dt::Real
end

struct State
    ρ::BlockDiagonal
    dρ::BlockDiagonal
    τ::BlockDiagonal
    # Internal fields
    _tmp1::BlockDiagonal
    _tmp2::BlockDiagonal
    _new_ρ::BlockDiagonal

    State(ρ) = new(ρ, zero(ρ), zero(ρ), similar(ρ), similar(ρ), similar(ρ))
end

function coherentspinstate(Nj, dense=true)
    ρ0 = blockdiagonal(css(Nj), dense=dense)
    State(ρ0)
end

struct SimulationParameters
    Ntraj::Integer
    Tfinal::Real
    dt::Real
    outpoints::Integer
end

struct KrausOperator
    M0::BlockDiagonal
    M::BlockDiagonal
end

function updatekraus!(M::KrausOperator, A::BlockDiagonal)
    @inbounds for i in eachindex(M.M.blocks)
        M.M.blocks[i] = M.M0.blocks[i] + A.blocks[i]
    end
end

struct Model
    params::ModelParameters
    Jx::BlockDiagonal
    Jy::BlockDiagonal
    Jz::BlockDiagonal
    Jx2::BlockDiagonal
    Jy2::BlockDiagonal
    Jz2::BlockDiagonal
    H::BlockDiagonal
    dH::BlockDiagonal
    second_term::SuperOperator
    M::KrausOperator
    dM::BlockDiagonal
end

function InitializeModel(modelparams::ModelParameters)
    Nj = modelparams.Nj
    dt = modelparams.dt
    kcoll = modelparams.kcoll
    kind = modelparams.kind
    ω = modelparams.omega
    η = modelparams.eta

    # Spin operators
    (Jx, Jy, Jz) = map(blockdiagonal, jspin(Nj))

    sys = piqs.Dicke(Nj)
    sys.dephasing = 4.

    liouvillian = tosparse(sys.liouvillian())
    indprepost = liouvillian + Nj*I

    Jx2 = Jx^2
    Jy2 = Jy^2
    Jz2 = Jz^2

    H = ω * Jz
    dH = Jz

    # Kraus-like operator, trajectory-independent part
    M0 = blockdiagonal(I - 1im * H * dt -
                0.25 * dt * kind * Nj * I - # The Id comes from the squares of sigmaz_j
                (kcoll/2) * Jy2 * dt)

    # Derivative of the Kraus-like operator wrt to ω
    dM = -1im * dH * dt

    M = KrausOperator(M0, similar(M0))

    # TODO: Find better name
    second_term = ((1 - η) * dt * kcoll * sup_pre_post(sparse(Jy)) +
            dt * (kind / 2) * indprepost)
    dropzeros!(second_term)

    second_term = SuperOperator(second_term)

    Model(modelparams, Jx, Jy, Jz, Jx2, Jy2, Jz2, H, dH, second_term, M, dM)
end

function measure_current(state::State, model::Model)
    dW() = sqrt(model.params.dt) * randn() # Define the Wiener increment
    # Homodyne current (Eq. 35)
    mul!(state._tmp1, model.Jy, state.ρ)
    2 * sqrt(model.params.kcoll * model.params.eta) * real(tr(state._tmp1)) * model.params.dt + dW()
end

function updatekraus!(model::Model, dy::Real)
    p = model.params
    # Kraus operator Eq. (36)
    updatekraus!(model.M, sqrt(p.eta * p.kcoll) * model.Jy * dy +
                            p.eta * (p.kcoll / 2) * model.Jy2 * (dy^2 - p.dt))
end

function updatestate!(state::State, model::Model)
    # Non-allocating code for
    # new_ρ = Mpre * Mpost * ρ + second_term * ρ

    mul!(state._tmp1, state.ρ, model.M.M')
    mul!(state._new_ρ, model.M.M, state._tmp1)
    apply_superop!(state._tmp1, model.second_term, state.ρ)

    # TODO: Replace with broadcasting once implemented
    for (i, b) in enumerate(blocks(state._new_ρ))
        b .+= state._tmp1.blocks[i]
    end

    zchop!(state._new_ρ) # Round off elements smaller than 1e-14
    tr_ρ = tr(state._new_ρ)

    # Evolve the unnormalized derivative wrt ω

    # Non-allocating code for
    # τ = (Mpre * (Mpost * τ  +  dMpost * ρ) + dMpre * Mpost * ρ +
    #      second_term * τ )/ tr_ρ;
    mul!(state._tmp1, state.ρ, model.dM')
    mul!(state._tmp1, state.τ, model.M.M', 1., 1.)
    apply_superop!(state._tmp2, model.second_term, state.τ)
    mul!(state._tmp2, model.M.M, state._tmp1, 1., 1.)
    mul!(state._tmp1, state.ρ, model.M.M')
    mul!(state._tmp2, model.dM, state._tmp1, 1., 1.)

    state.τ .= state._tmp2

    # TODO: Use broadcasting when it is implemented
    for b in blocks(state.τ)
        b ./= tr_ρ
    end

    zchop!(state.τ) # Round off elements smaller than 1e-14

    tr_τ = tr(state.τ)

    # Now we can renormalize ρ and its derivative wrt ω
    # TODO: Use broadcasting when it is implemented
    for i = eachindex(state.ρ.blocks)
        state.ρ.blocks[i] .= state._new_ρ.blocks[i] ./ tr_ρ
    end
    for i = eachindex(state.dρ.blocks)
        state.dρ.blocks[i] .= state.τ.blocks[i] .- tr_τ .* state.ρ.blocks[i]
    end
    return tr_ρ, tr_τ
end

expectation_value!(state::State, op::AbstractArray) = real(tr(mul!(state._tmp1, op, state.ρ)))
