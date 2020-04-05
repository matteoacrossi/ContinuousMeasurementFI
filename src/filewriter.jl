using HDF5

function write_to_file(fid::HDF5File, file_channel, timevec, Ntraj)

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