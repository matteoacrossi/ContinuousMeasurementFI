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

Ntraj = args["Ntraj"]

@everywhere to = TimerOutput()

modelparams = ModelParameters(Nj=args["Nj"],
                              kind=args["kind"],
                              kcoll=args["kcoll"],
                              omega=args["omega"],
                              eta=args["eta"],
                              dt=args["dt"],
                              Tfinal=args["Tfinal"],
                              outpoints=args["outpoints"])

@info "Output every $(modelparams._outsteps) steps"

@info "Initializing model..."

init_time = @elapsed begin
@everywhere model = InitializeModel($modelparams, $args["liouvillianfile"])
@everywhere initial_state = coherentspinstate($modelparams.Nj)
end

@info "Done in $init_time seconds..."


# Starts the writer function on the main process

file_channel = RemoteChannel(() -> Channel{NamedTuple}(200))
writer = @async write_to_file(file_channel, get_time(model), args["Ntraj"])

progress_channel = RemoteChannel(() -> Channel{Bool}(200))

p = Progress(Ntraj * length(get_time(model)))
@async while take!(progress_channel)
    next!(p)
end

numthreads = parse(Int, ENV["JULIA_NUM_THREADS"])

@everywhere function thread_simulate_trajectory(model, initial_state, file_channel, progress_channel, ntraj)
    pid = Distributed.myid()
    nth = Threads.nthreads()
    Threads.@threads for i in 1:ntraj
        tid = Threads.threadid()
        println("Hello from thread $tid of $nth on worker $pid.")
        simulate_trajectory(model, initial_state, file_channel, progress_channel)
    end
end

trajectory_time = @elapsed begin
    pmap(x -> thread_simulate_trajectory(model, initial_state, file_channel, progress_channel, numthreads), 1:numthreads:Ntraj)
end

@info "Trajectory simulation time: $trajectory_time"

put!(progress_channel, false)

for i in eachindex(workers())
    fetch(@spawnat i begin
        try
            result = read(`grep VmHWM /proc/$(getpid())/status`, String)
            peakmem = tryparse(Int, String(match(r"(\d+)", result)[1]))

            @info "Peak memory in GB:" peakmem / 1024^3
        catch er
            @info er
        end
    end)
end

close(file_channel)

fid = fetch(writer)
@show fid

close(fid)
