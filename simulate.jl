using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere using ContinuousMeasurementFI
@everywhere using TimerOutputs

using HDF5
using DelimitedFiles
using Logging

using ArgParse
using ProgressMeter

# Parse the arguments passed to the script
s = ArgParseSettings()

@add_arg_table! s begin
    "--Nj"
        help = "Number of spins"
        default = 1
        arg_type = Int64
    "--Ntraj"
        help = "Number of trajectories"
        default = 1
        arg_type = Int64
    "--Tfinal", "-T"
        help = "Final time"
        arg_type = Float64
        default = 1.0
    "--dt"
        help = "dt"
        arg_type = Float64
        default = 0.0001
    "--kind"
        help = "Independent noise rate k_ind"
        arg_type = Float64
        default = 1.0
    "--kcoll"
        help = "Collective noise rate k_coll"
        arg_type = Float64
        default = 1.0
    "--omega"
        help = "Hamiltonian frequency omega"
        arg_type = Float64
        default = 1.0
    "--eta"
        help = "Measurement efficiency"
        arg_type = Float64
        default = 1.0
    "--outpoints"
        help = "Number of output points"
        arg_type = Int64
        default = 200

    "--liouvillianfile"
        help = "npz file with the liovuillian data"
        arg_type = String
        default = nothing
end

args = parse_args(s)

filename = string("sim_$(args["Nj"])_$(args["Ntraj"])_$(args["Tfinal"])_$(args["dt"])_$(args["kind"])_$(args["kcoll"])_$(args["omega"])_$(args["eta"])")

pkgs = Pkg.installed();

@info "ContinuousMeasurementFI version: $(pkgs["ContinuousMeasurementFI"])"

@info "Output filename: $filename"

outsteps = 1
Tfinal = args["Tfinal"]
dt = args["dt"]
outpoints = args["outpoints"]
Ntraj = args["Ntraj"]

if outpoints > 0
    try
        outsteps = Int(round(Tfinal / dt / outpoints, digits=3))
    catch InexactError
        @warn "The requested $outpoints output points does not divide
        the total time steps. Using the full time output."
    end
end
outsteps = Int(round(Tfinal / dt / outpoints, digits=3))

@info outsteps
@info "Output every $outsteps steps"

@everywhere to = TimerOutput()

modelparams = ModelParameters(args["Nj"], args["kind"], args["kcoll"], args["omega"], args["eta"], args["dt"])

Ntime = Int(floor(Tfinal/dt)) # Number of timesteps
t = (1 : Ntime) * dt
t = t[outsteps:outsteps:end]

@info "Initializing model..."
init_time = @elapsed begin
@everywhere model = InitializeModel($modelparams, args["liouvillianfile"])
@everywhere initial_state = coherentspinstate($modelparams.Nj)
end
@info "Done in $init_time seconds..."

function write_to_file(file_channel, timevec, Ntraj)
    fid = h5open("$filename.h5", "w")
    fid["t"] = collect(timevec)
    outpoints = length(timevec)

    datasets = Dict()
    #for quantity in ("FI", "QFI", "Jx", "Jy", "Jz", "Δjx", "Δjy", "Δjz", "xi2x", "xi2y", "xi2z")
    for quantity in ("FI", "QFI", "xi2y")
        ds = d_create(fid, quantity, Float64, ((outpoints, Ntraj), (outpoints, -1)), "chunk", (outpoints, 1))
        datasets[quantity] = ds
    end
    #p = Progress(Ntraj, 1)
    counter = 1
    @info "Writer ready!"
    while true
        try
            traj_result = take!(file_channel)
            for (d, data) in pairs(traj_result)
                if string(d) in keys(datasets)
                    try
                        datasets[string(d)][:, counter] = data
                    catch er
                        @error er
                    end
                end
            end
            #next!(p)
            counter += 1
        catch InvalidStateException
            @info "Channel closed"
            return fid
        end
    end
    return fid
end

# Starts the writer function on the main process

file_channel = RemoteChannel(() -> Channel{NamedTuple}(200))
writer = @spawnat 1 write_to_file(file_channel, t, args["Ntraj"])

progress_channel = RemoteChannel(() -> Channel{Bool}(200))

p = Progress(Ntraj * length(t))
@async while take!(progress_channel)
    next!(p)
end

@info "Starting trajectories simulation..."
@sync @distributed for ktraj = 1 : Ntraj
    state = State(copy(initial_state.ρ))

    # Output variables
    jx = similar(t)
    jy = similar(t)
    jz = similar(t)

    jx2 = similar(t)
    jy2 = similar(t)
    jz2 = similar(t)

    FisherT = similar(t)
    QFisherT = similar(t)

    jto = 1 # Counter for the output
    for jt = 1 : Ntime
        dy = measure_current(state, model)
        updatekraus!(model, dy)
        tr_ρ, tr_τ = updatestate!(state, model)

        if jt % outsteps == 0

            jx[jto] = expectation_value!(state, model.Jx)
            jy[jto] = expectation_value!(state, model.Jy)
            jz[jto] = expectation_value!(state, model.Jz)

            jx2[jto] = expectation_value!(state, model.Jx2)
            jy2[jto] = expectation_value!(state, model.Jy2)
            jz2[jto] = expectation_value!(state, model.Jz2)

            # We evaluate the classical FI for the continuous measurement
            FisherT[jto] = real(tr_τ^2)

            # We evaluate the QFI for a final strong measurement done at time t
            QFisherT[jto] = QFI(state)

            jto += 1
            put!(progress_channel, true)
        end
    end

    Δjx2 = jx2 - jx.^2
    Δjy2 = jy2 - jy.^2
    Δjz2 = jz2 - jz.^2

    xi2x = squeezing_param(model.params.Nj, Δjx2, jy, jz)
    xi2y = squeezing_param(model.params.Nj, Δjy2, jx, jz)
    xi2z = squeezing_param(model.params.Nj, Δjz2, jx, jy)

    if !isnothing(file_channel)
        result = (FI=FisherT, QFI=QFisherT,
                    Jx=jx, Jy=jy, Jz=jz,
                    Δjx=Δjx2, Δjy=Δjy2, Δjz=Δjz2,
                    xi2x=xi2x, xi2y=xi2y, xi2z=xi2z)
        put!(file_channel, result)
    else
        @info "No file channel!"
    end
end
put!(progress_channel, false)
for i in eachindex(workers())
    fetch(@spawnat i begin
        result = read(`grep VmHWM /proc/$(getpid())/status`, String)
        peakmem = tryparse(Int, String(match(r"(\d+)", result)[1]))

        @info "Peak memory in GB:" peakmem / 1024^3
    end)
end

close(file_channel)

fid = fetch(writer)
@show fid

close(fid)
