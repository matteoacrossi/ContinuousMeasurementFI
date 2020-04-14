using Test
using ContinuousMeasurementFI
using HDF5
using Random

typedict(x) = Dict(string(fn)=>getfield(x, fn) for fn ∈ fieldnames(typeof(x)))

# Opens the FileWriter
@testset "FileWriter" begin
    Ntraj = 100

    params = ModelParameters(outpoints=200)

    timevector = collect(get_time(params))

    filename = tempname() # Get a temporary filename

    @info "Temporary file: $filename"
    FI = rand(length(timevector), Ntraj)
    QFI = rand(length(timevector), Ntraj)
    xi2y = rand(length(timevector), Ntraj)

    writer = FileWriter(filename, params, Ntraj, ["FI", "QFI", "xi2y", "something"])

    for i=1:Ntraj
        tuple = (FI=FI[:, i],
                 QFI=QFI[:, i],
                 xi2y=xi2y[:, i],
                 extrafield=xi2y[:, i]) # This field should be ignored
        put!(writer, tuple)
    end

    close(writer)

    fid = h5open(filename, "r")

    @testset "datasets" begin
        @test read(fid["t"]) == timevector
        @test read(fid["FI"]) == FI
        @test read(fid["QFI"]) == QFI
        @test read(fid["xi2y"]) == xi2y
        @test iszero(read(fid["something"]))
        @test read(attrs(fid)["stored_traj"]) == Ntraj
        close(fid)
    end
    @testset "attrs" begin
        h5attr = h5readattr(filename, "/")
        pop!(h5attr, "stored_traj")
        @test h5attr == typedict(params)
    end

end