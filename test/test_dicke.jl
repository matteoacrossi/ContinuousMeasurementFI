using ContinuousMeasurementFI

#using Plots
using Test
using Random

@testset "Dicke basis" begin
    @testset "Nj = $Nj" for Nj in 1:5
        Ntraj = 1
        Tfinal = 1.0
        dt = 0.001
        κ_ind = 1.5
        κ_coll = 1.
        θ = 0. # Angle of the independent noise (θ = 0 : parallel)
        ω = .1
        η = 1.0
        #Nj = 5

        seed = 20

        Random.seed!(seed)
        @time res_dicke = Eff_QFI_HD_Dicke(Nj, Ntraj,# Number of trajectories
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

        #println(er_FI)
        if er_FI > 1e-3
                @warn "Relative error FI high" er_FI
        end

        er_QFI = maximum(abs.(res_dicke.QFI - res_sup.QFI) ./ res_sup.QFI)
        if er_QFI > 1e-3
                @warn "Relative error QFI high" er_QFI
        end

        #println(er_QFI)

        rtol = 1e-8
        @test res_dicke.FI ≈ res_sup.FI rtol=rtol atol=dt^2
        @test res_dicke.QFI ≈ res_sup.QFI rtol=rtol atol=dt^2
        @test res_dicke.jx ≈ res_sup.jx rtol=rtol atol=dt^2
        @test res_dicke.jy ≈ res_sup.jy rtol=rtol atol=dt^2
        @test res_dicke.jz ≈ res_sup.jz rtol=rtol atol=dt^2
    end
end