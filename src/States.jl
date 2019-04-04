#=
Functions for constructing initial states
=#
"""
    plus_state(n)

Return the state ``\\ket{+}^{\\otimes n}`` in the computational basis
"""
function plus_state(n::Int)
    @assert n > 0 "n must be a positive integer"
    spinup = Vector{Complex{Float64}}([1., 0.])
    spindown = Vector{Complex{Float64}}([0., 1.])

    return kron([(spinup + spindown) / sqrt(2.) for i in 1:n]...)
end

"""
    ghz_state(n)

Return the GHZ state ``(\\ket{00...0} + \\ket{11...1}) / \\sqrt{2}`` in the computational basis
"""
function ghz_state(n::Int)
    @assert n > 0 "n must be a positive integer"
    r = zeros(Complex{Float64}, 2^n)
    r[1] = r[end] = 1. / sqrt(2.)
    return r
end

function random_state(n::Int)
    @assert n > 0 "n must be a positive integer"
    r = rand(Complex{Float64}, 2^n)
    r /= norm(r)
    return r
end