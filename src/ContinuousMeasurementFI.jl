"""
Fisher information for magnetometry with continuously monitored spin systems, with independent Markovian noise acting on each spin
"""
module ContinuousMeasurementFI
    using PyCall
    const qutip = PyNULL()
    const piqs = PyNULL()
    const sp = PyNULL()
    const pystuff = PyNULL()

    function __init__()
        # The commands below import the modules, and make sure that they Arguments
        # installed using Conda.jl
        copy!(qutip, pyimport_conda("matplotlib", "matplotlib"))
        copy!(qutip, pyimport_conda("qutip", "qutip", "conda-forge"))
        copy!(piqs, pyimport_conda("qutip.piqs", "qutip", "conda-forge"))
        copy!(sp, pyimport_conda("scipy.sparse", "scipy"))

        pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)

        copy!(pystuff, pyimport("pystuff"))
    end

    include("piqs.jl")

    using SparseArrays
    using LinearAlgebra

    export Molmer_QFI_GHZ, Molmer_qfi_transverse, uncond_QFI_transverse
    export QFI, QFI!, squeezing_param
    export Unconditional_QFI, Unconditional_QFI_Dicke
    export Eff_QFI_HD
    export Eff_QFI_HD_Dicke, Eff_QFI_HD_Dicke_0
    export simulate_trajectory
    export liouvillian, jspin, css
    export Model, ModelParameters, State, updatestate!, updatekraus!, measure_current, expectation_value!
    export get_time
    export InitializeModel, coherentspinstate
    export write_to_file
    include("NoiseOperators.jl")
    include("model.jl")
    include("States.jl")
    include("Fisher.jl")
    include("Molmer_QFI.jl")
    include("Uncond_qfi_transverse.jl")

    include("Eff_QFI_HD_Dicke.jl")
    include("Eff_QFI_HD.jl")
    include("filewriter.jl")

    include("utils.jl")

    """
        (t, FI, QFI) = Eff_QFI_HD(Nj, Ntraj, Tfinal, dt; kwargs... )

    Evaluate the continuous-time FI and QFI of a final strong measurement for the
    estimation of the frequency ω with continuous homodyne monitoring of
    collective transverse noise, with each half-spin particle affected by
    parallel noise, with efficiency η using SME
    (stochastic master equation).

    The function returns a NamedTuple `(t, FI, QFI)` containing the time vector and the vectors containing the FI and average QFI

    # Arguments

    * `Nj`: number of spins
    * `Ntraj`: number of trajectories for the SSE
    * `Tfinal`: final time of evolution
    * `dt`: timestep of the evolution
    * `κ = 1`: the independent noise coupling
    * `κcoll = 1`: the collective noise coupling
    * `ω = 0`: local value of the frequency
    * `η = 1`: measurement efficiency
    """
    function Eff_QFI_HD(
        Nj::Int64, # Number of spins
        Ntraj::Int64,                    # Number of trajectories
        Tfinal::Real,                    # Final time
        dt::Real;                        # Time step
        κ::Real = 1.,                    # Independent noise strength
        κcoll::Real = 1.,                # Collective noise strength
        ω::Real = 0.0,                   # Frequency of the Hamiltonian
        η::Real = 1.
    )

    θ = 0
    # ω is the parameter that we want to estimate
    H = ω * σ(:z, Nj) / 2       # Hamiltonian of the spin system

    dH = σ(:z, Nj) / 2          # Derivative of H wrt the parameter ω

    #non_monitored_noise_op = []
    non_monitored_noise_op = (sqrt(κ/2) *
        [(cos(θ) * σ_j(:z, j, Nj) + sin(θ) * σ_j(:y, j, Nj)) for j = 1:Nj])

    monitored_noise_op = [sqrt(κcoll) * σ(:y, Nj)/2]

    res = Eff_QFI_HD(Ntraj,# Number of trajectories
        Tfinal,                           # Final time
        dt,                               # Time step
        H, dH,                            # Hamiltonian and its derivative wrt ω
        non_monitored_noise_op,           # Non monitored noise operators
        monitored_noise_op;               # Monitored noise operators
        initial_state = coherent_state,
        η=η)

    return res
    end
end
