using ContinuousMeasurementFI

#using Plots
using Test
using Random

@testset "Dicke basis" begin
        Ntraj = 1
        Tfinal = 1.0
        dt = 0.001
        κ_ind = 1.5
        κ_coll = 1.
        θ = 0. # Angle of the independent noise (θ = 0 : parallel)
        ω = .1
        η = 1.0
        Nj = 3

        seed = 20

        Random.seed!(seed)
        @time res_dicke = Eff_QFI_HD_dicke(Nj, Ntraj,# Number of trajectories
                Tfinal,                              # Final time
                dt;                                  # Time step
                κ = κ_ind,
                κcoll = κ_coll,
                ω = ω,
                η = η)     

        Random.seed!(seed)
        @time res_sup = Eff_QFI_HD(Nj, Ntraj,# Number of trajectories
                        Tfinal,                              # Final time
                        dt;                                  # Time step
                        κ = κ_ind,
                        κcoll = κ_coll,
                        ω = ω,
                        η = η)      # Initial state
                
        er_FI = maximum(abs.(res_dicke.FI - res_sup.FI) ./ res_sup.FI)

        if er_FI > 1e-3
                @warn "Relative error FI high" er_FI
        end

        er_QFI = maximum(abs.(res_dicke.QFI - res_sup.QFI) ./ res_sup.QFI)
        if er_QFI > 1e-3
                @warn "Relative error QFI high" er_QFI
        end

        @test res_dicke.FI ≈ res_sup.FI rtol=1e-2 atol=dt
        @test res_dicke.QFI ≈ res_sup.QFI rtol=1e-2 atol=dt
        @test res_dicke.jx ≈ res_sup.jx rtol=1e-2 atol=dt
        @test res_dicke.jy ≈ res_sup.jy rtol=1e-2 atol=dt
        @test res_dicke.jz ≈ res_sup.jz rtol=1e-2 atol=dt
end