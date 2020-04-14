using ContinuousMeasurementFI
using Test

@testset "Pretty printing" begin
Nj = 2
modelparams = ModelParameters(Nj=Nj)

model = InitializeModel(modelparams)
initial_state = coherentspinstate(Nj)

println(model)
println(initial_state)

end