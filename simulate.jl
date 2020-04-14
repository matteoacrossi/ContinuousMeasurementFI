using Distributed
using ClusterManagers

try
    num_tasks = parse(Int, ENV["SLURM_NTASKS"])
    @info "Number of slurm tasks: $num_tasks"

    addprocs(SlurmManager(num_tasks), topology=:master_worker)
catch KeyError
    @warn "Not inside a slurm job"
end

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

jobid = "SLURM_JOB_ID" in keys(ENV) ? jobid = ENV["SLURM_JOB_ID"] : ""
filename = string("sim_$(args["Nj"])_$(args["Ntraj"])_$(args["Tfinal"])_$(args["dt"])_$(args["kind"])_$(args["kcoll"])_$(args["omega"])_$(args["eta"])_$(jobid)")

let pkgstr = ""
    for (uuid, pkg) in Pkg.dependencies()
        if pkg.is_direct_dep
            pkgstr *= "\t$(pkg.name) $(pkg.version)"
            if pkg.is_tracking_repo
                pkgstr *= " #$(pkg.git_revision) ($(pkg.git_source))"
            elseif pkg.is_tracking_path
                pkgstr *= " $(pkg.source)"
            end
            pkgstr *= "\n"
        end
    end
    @info "Installed packages\n$pkgstr"
end

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

println(model)
println(initial_state)

@info "Done in $init_time seconds..."

# Opens the FileWriter
writer = FileWriter("$filename.h5", modelparams, args["Ntraj"], ["FI", "QFI", "xi2y"])

progress_channel = RemoteChannel(() -> Channel{Bool}(200))

p = Progress(Ntraj * length(get_time(model)), barglyphs=BarGlyphs("[=> ]"), barlen=10)

@async while take!(progress_channel)
    next!(p)
end

if "JULIA_NUM_THREADS" in keys(ENV)
    @everywhere numthreads = parse(Int, ENV["JULIA_NUM_THREADS"])
    @everywhere @info("Number of threads: $numthreads")
else
    @everywhere @info("JULIA_NUM_THREADS not set, using 1 thread")
    @everywhere numthreads = 1
end


@everywhere function thread_simulate_trajectory(model, initial_state, ntraj, filewriter, progress_channel)
    pid = Distributed.myid()
    nth = Threads.nthreads()
    Threads.@threads for i in 1:ntraj
        tid = Threads.threadid()
        #println("Hello from thread $tid of $nth on worker $pid.")
        simulate_trajectory(model, initial_state, filewriter, progress_channel)
    end
end

trajectory_time = @elapsed begin
    pmap(x -> thread_simulate_trajectory(model, initial_state, numthreads, writer, progress_channel), 1:numthreads:Ntraj)
end

put!(progress_channel, false)

@info "Trajectory simulation time: $trajectory_time"

for i in eachindex(workers())
    fetch(@spawnat i begin
        try
            result = read(`grep VmHWM /proc/$(getpid())/status`, String)
            peakmem = tryparse(Int, String(match(r"(\d+)", result)[1]))

            @info "Peak memory in GB: $(peakmem / 1024^2)"
        catch er
            @info er
        end
    end)
end

close(writer)
