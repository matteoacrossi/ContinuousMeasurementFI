function Uncond_qfi_parallel(t, N, κ, ω)
    return N^2* t.^2 .* exp.(-2 * κ * N * t)
end
