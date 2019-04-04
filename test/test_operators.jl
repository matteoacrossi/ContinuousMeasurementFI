using Test
using LinearAlgebra

include("../src/NoiseOperators.jl")
@testset "Trace for vectorized matrix" begin
    N = 4
    ρ = rand(N, N) + 1im * rand(N,N)
    @test tr(ρ) ≈ trace(ρ[:])
end

@testset "Spre, spost" begin
N = 4
ρ = rand(N, N) + 1im * rand(N,N)
A = rand(N, N) + 1im * rand(N,N)

@test ρ * A ≈ reshape(sup_post(A) * ρ[:], (N,N))
@test A * ρ ≈ reshape(sup_pre(A) * ρ[:], (N,N))

ρ = sprand(N, N, .1) + 1im *  sprand(N, N, .1)
A = sprand(N, N, .1) + 1im * sprand(N, N, .1)

@test ρ * A ≈ reshape(sup_post(A) * ρ[:], (N,N))
@test A * ρ ≈ reshape(sup_pre(A) * ρ[:], (N,N))

println(typeof(A))
println(typeof(sup_post(A)))
@test typeof(sup_post(A)) <: typeof(A)
@test typeof(sup_pre(A)) <: typeof(A)

ρ = rand(N, N) + 1im * rand(N,N)
A = rand(N, N) + 1im * rand(N,N)
B = rand(N, N) + 1im * rand(N,N)

@test A * ρ * B' ≈ reshape(sup_pre(A) * sup_post(B)' * ρ[:], (N,N))
@test A * ρ * B' ≈ reshape(sup_post(B)' * sup_pre(A) * ρ[:], (N,N))
@test A * ρ * B' ≈ reshape(sup_pre_post(A, B') * ρ[:], (N, N))
@test A * ρ * A' ≈ reshape(sup_pre_post(A) * ρ[:], (N, N))

end
