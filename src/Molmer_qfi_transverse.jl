"""
    Molmer_qfi_transverse(t, Nj, κ)

The Mølmer QFI for ``N_j`` atoms interacting with independent transverse
noise with coupling ``\\κ``.

`t` can be any `Array`.

This function effectively evaluates the formula

``Q(t) = 4  \\frac{N_j}{\\κ ^2} e^{-\\κ  t} \\left(2 (2- N_j) e^{\\frac{\\κ  t}{2}}+e^{\\κ
   t} (\\κ  t+ N_j-3)+N_j-1\\right)``

"""
function Molmer_qfi_transverse(t, Nj, κ)   
    return exp.(-2 * κ .* t) .* κ.^(-2) .* Nj  .*
     (-1 -2 .* exp.(κ .* t) .* (Nj - 2) + Nj + 
        exp.(2.* κ .* t ) .* (-3+Nj +2.* κ.*t));
end
