using ForwardDiff
include("NoiseOperators.jl")

function H(ω)
    ω * [0.5 0; 0 0.5]
end

dH = ω -> ForwardDiff.derivative(H, ω); # g = ∇f
