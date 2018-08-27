"""
    Molmer_QFI_GHZ(Nj, Tfinal, dt; kwargs...)

Numerically exact evaluation of the Mølmer QFI for the GHZ state for ``Nj`` atoms interacting with independent noise. `Tfinal` and `dt` specify the

Returns `(t, qfi)` where `qfi` is the Mølmer QFI evaluated at time instants in `t`.

# Keyword arguments

    * `κ = 1`: coupling with the noise
    * `θ = 0`: angle with the noise
    * `ω = 0`: magnetic field / frequency
    * `dω = 1e-5`: increment dω for numerically evaluating the derivative
"""
function Molmer_QFI_GHZ(
    Nj::Int64,      # Number of spins
    Tfinal,         # Final time
    dt::Float64;    # Time step
    κ = 1.,         # Field rate
    θ = 0.,         # Angle of the noise
    ω = 0.,         # Magnetic field
    dω=1e-5        # Increment to numerically compute the derivative (instable if too small wrt to ω)
    )

    Ntime = Int(floor(Tfinal/dt))
    t = (1 : Ntime) * dt
    θ = pi/2 - θ # Matteo chose the opposite convention for the effective QFI
    QFisherT = zero(t)

    lmat=[ 0 0 0 -1im.*dω ;
            0  -κ.*sin(θ).^2  -ω   κ.*cos(θ).*sin(θ) ;
            0 ω -κ 0 ;
             -1im.*dω  κ.*cos(θ).*sin(θ) 0 -κ.*cos(θ).^2]

    for jt=1:Ntime
        fmat=expm(lmat*t[jt])
        f00=sqrt(0.5).*fmat*[1. 0. 0. 1.]'
        f11=sqrt(0.5).*fmat*[1. 0. 0. -1.]'
        f01=sqrt(0.5).*fmat*[0. 1. 1im 0.]'
        f10=sqrt(0.5).*fmat*[0. 1. -1im 0.]'
        logtrρ=log.(abs(0.5 .*( (f00[1] .* sqrt(2.)) .^ Nj + (f11[1] .* sqrt(2.)) .^ Nj + (f01[1] .* sqrt(2.)) .^ Nj + (f10[1] .* sqrt(2.)) .^ Nj )))
        # check: the trace of ρ should be less then 1... it can be approximately 0, but positive:
        # in that case we manually set it to 0
        if logtrρ<0
            QFisherT[jt] = 2. .* exp.(log.(-logtrρ) - 2. *log.(dω))
        else
            QFisherT[jt] = 0.
        end
    end
    return (t,QFisherT)
end

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
        exp.(2. * κ .* t ) .* (-3+Nj +2. * κ .* t));
end
