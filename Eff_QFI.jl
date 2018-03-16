include("Eff_QFI_PD.jl")
include("Eff_QFI_PD_pure.jl")
include("Eff_QFI_HD.jl")
include("Eff_QFI_HD_pure.jl")

"""
    (t, FI, QFI) = Eff_QFI(kwargs...)

Evaluate the continuous-time FI and QFI of a final strong measurement for the
estimation of the frequency ω with continuous monitoring of each half-spin
particle affected by noise at an angle θ, with efficiency η using SME
(stochastic master equation).

The function returns a tuple `(t, FI, QFI)` containing the time vector and the
vectors containing the FI and average QFI

# Arguments

* `Nj`: number of spins
* `Ntraj`: number of trajectories for the SSE
* `Tfinal`: final time of evolution
* `measurement = :pd` measurement (either `:pd` or `:hd`)
* `dt`: timestep of the evolution
* `κ = 1`: the noise coupling
* `θ = 0`: noise angle (0 parallel, π/2 transverse)
* `ω = 0`: local value of the frequency
* `η = 1`: measurement efficiency
"""
function Eff_QFI(args::Dict{String, Any})
    kwargs = map(x -> (Symbol(x[1]), x[2]), collect(args))
    @show kwargs
    tic()
    (t, fi, qfi) = Eff_QFI(; kwargs...)
    eval_time = toc()
    return merge(args, Dict("t" => t, "fi" => fi,"qfi" => qfi, "eval_time" => eval_time))
end

function Eff_QFI(; Nj=1,        # Number of spins
                Ntraj=1,        # Number of trajectories
                Tfinal=1.,      # Final time
                dt=0.01,        # Time step
                measurement=:pd,# Measurement type
                η = 1.,         # Measurement efficiency
                κ = 1.,         # Field rate
                θ = pi/2,       # Angle of the noise
                ω = 1.,         # Magnetic field
                _watherver...)

        if measurement == :pd
            if η == 1.
                f = Eff_QFI_PD_pure
            else
                f = Eff_QFI_PD
            end
        elseif measurement == :hd
            if η == 1.
                f = Eff_QFI_HD_pure
            else
                f = Eff_QFI_HD
            end
        else
            throw(ArgumentError("Measurement must be either :pd or :hd"))
        end

        if η == 1.
            f(Nj, Ntraj, Tfinal, dt;
                κ = κ,
                θ = θ,
                ω = ω)
        else
            f(Nj, Ntraj, Tfinal, dt;
            κ = κ,
            θ = θ,
            ω = ω,
            η = η)
        end

    end
