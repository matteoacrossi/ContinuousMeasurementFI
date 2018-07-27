using Base.Test

@testset "Trace for vectorized matrix" begin
    N = 4
    ρ = rand(N, N) + 1im * rand(N,N)
    @test trace(ρ) ≈ trace(ρ[:])
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

@test typeof(sup_post(A)) <: SparseMatrixCSC
@test typeof(sup_pre(A)) <: SparseMatrixCSC

ρ = rand(N, N) + 1im * rand(N,N)
A = rand(N, N) + 1im * rand(N,N)
B = rand(N, N) + 1im * rand(N,N)

@test A * ρ * B' ≈ reshape(sup_pre(A) * sup_post(B)' * ρ[:], (N,N))
@test A * ρ * B' ≈ reshape(sup_post(B)' * sup_pre(A) * ρ[:], (N,N))
@test A * ρ * B' ≈ reshape(sup_pre_post(A, B') * ρ[:], (N, N))
@test A * ρ * A' ≈ reshape(sup_pre_post(A) * ρ[:], (N, N))

end
