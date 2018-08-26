function QFI_Molmer_GHZ(
    Nj::Int64,      # Number of spins
    Tfinal,         # Final time
    dt::Float64;    # Time step
    κ = 1.,     # Field rate
    θ = 0.,     # Angle of the noise
    om = 0.,        # Magnetic field
    dom=1e-5        # Increment to numerically compute the derivative (instable if too small wrt to om)
    )

    Ntime = Int(floor(Tfinal/dt))
    t = (1 : Ntime) * dt
    θ = pi/2 - θ # Matteo chose the opposite convention for the effective QFI
    QFisherT = zero(t)

    lmat=[ 0 0 0 -1im.*dom ; 
            0  -κ.*sin(θ).^2  -om   κ.*cos(θ).*sin(θ) ; 
            0 om -κ 0 ;
             -1im.*dom  κ.*cos(θ).*sin(θ) 0 -κ.*cos(θ).^2]

    for jt=1:Ntime
        fmat=expm(lmat*t[jt])
        f00=sqrt(1/2).*fmat*[1 0 0 1]'
        f11=sqrt(1/2).*fmat*[1 0 0 -1]'
        f01=sqrt(1/2).*fmat*[0 1 1im 0]'
        f10=sqrt(1/2).*fmat*[0 1 -1im 0]'
        logtrρ=log.(abs(0.5.*( (f00[1].*sqrt(2)).^Nj + (f11[1].*sqrt(2)).^Nj + (f01[1].*sqrt(2)).^Nj + (f10[1].*sqrt(2)).^Nj )))
        # check: the trace of ρ should be less then 1... it can be approximately 0, but positive:
        # in that case we manually set it to 0 
        if logtrρ<0
            QFisherT[jt] = 2.*exp.(log.(-logtrρ) - 2.*log.(dom))
        else
            QFisherT[jt] = 0.
        end
    end
    return (t,QFisherT)
end