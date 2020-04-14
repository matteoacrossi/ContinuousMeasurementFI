using Test
using ContinuousMeasurementFI
using HDF5
using Random

# Opens the FileWriter
@testset "FileWriter" begin
    Ntraj = 100
    timevector = collect(1.:200.)
    filename = tempname(cleanup=true) # Get a temporary filename

    @info "Temporary file: $filename"
    FI = rand(length(timevector), Ntraj)
    QFI = rand(length(timevector), Ntraj)
    xi2y = rand(length(timevector), Ntraj)

    writer = FileWriter(filename, timevector, Ntraj, ["FI", "QFI", "xi2y", "something"])

    for i=1:Ntraj
        tuple = (FI=FI[:, i],
                 QFI=QFI[:, i],
                 xi2y=xi2y[:, i],
                 extrafield=xi2y[:, i]) # This field should be ignored
        put!(writer, tuple)
    end

    close(writer)

    fid = h5open(filename, "r")

    @test read(fid["t"]) == timevector
    @test read(fid["FI"]) == FI
    @test read(fid["QFI"]) == QFI
    @test read(fid["xi2y"]) == xi2y
    @test iszero(read(fid["something"]))
    @test read(attrs(fid)["stored_traj"]) == Ntraj
end