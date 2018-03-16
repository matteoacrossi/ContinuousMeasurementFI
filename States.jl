#=
Functions for constructing initial states
=#
using QuantumOptics

"""
    plus_state(n)

Return the state ``\\ket{+}^{\\otimes n}`` in the computational basis
"""
function plus_state(n::Int)
    b = SpinBasis(1//2)
    return tensor([(spinup(b) + spindown(b)) / sqrt(2) for i in 1:n]...).data
end

"""
    ghz_state(n)

Return the GHZ state ``(\\ket{00..0} + \ket{11...1}) / \\sqrt{2}`` in the computational basis
"""
function ghz_state(n::Int)
    b = SpinBasis(1//2)
    return (tensor([spinup(b) for i in 1:n]...).data + tensor([spindown(b) for i in 1:n]...).data)/sqrt(2)
end
