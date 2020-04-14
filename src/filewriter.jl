using HDF5
using Distributed

struct FileWriter
    fid::HDF5File
    channel::Union{Channel,RemoteChannel}
    total_trajectories::Int64
    timevector::Array{Float64,1}
    datasets::Dict
    writer::Task

    function FileWriter(filename::String, timevector::Array{Float64,1}, total_trajectories::Int64, quantities)
        fid = h5open(filename, "w")

        fid["t"] = timevector
        outpoints = length(timevector)

        datasets = Dict()
        #for quantity in ("FI", "QFI", "Jx", "Jy", "Jz", "Δjx", "Δjy", "Δjz", "xi2x", "xi2y", "xi2z")
        for quantity in quantities
            ds = d_create(fid, quantity, Float64, ((outpoints, total_trajectories), (outpoints, -1)), "chunk", (outpoints, 1))
            datasets[quantity] = ds
        end

        file_channel = RemoteChannel(() -> Channel{NamedTuple}(200))
        writer = @async writer_task(fid, datasets, file_channel)

        new(fid, file_channel, total_trajectories, timevector, datasets, writer)
    end
end

function writer_task(fid, datasets, channel)
    counter = 1
    attrs(fid)["stored_traj"] = 0
    @info "Writer ready!"
    while true
        try
            traj_result = take!(channel)
            for (d, data) in pairs(traj_result)
                if string(d) in keys(datasets)
                    try
                        datasets[string(d)][:, counter] = data
                    catch er
                        @error er
                    end
                end
            end
            @debug "Entry written to $(fid)"
            counter += 1
            write(attrs(fid)["stored_traj"], counter - 1)
            flush(fid)

        catch InvalidStateException
            @info "Channel closed"
            return fid
        end
    end
    return fid
end

function Base.close(fw::FileWriter)
    close(fw.channel)
    fid = fetch(fw.writer)
    close(fid)
    @info "FileWriter closed: $fid"
end

function Base.put!(fw::FileWriter, v)
    Base.put!(fw.channel, v)
end