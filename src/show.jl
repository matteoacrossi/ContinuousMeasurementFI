using PrettyTables

Base.show(io::IO, model::Model) = begin
    type = typeof(model.Jx)
    m = Base.format_bytes(Base.summarysize(model))
    println(io, "Continuous monitoring model with $(model.params.Nj)-spin Dicke state\n")

    println(io, "Operator type: $(typeof(model.M.M))")
    sup_size = model.second_term.size
    nvals = length(model.second_term.values)
    density = nvals / (*(sup_size...))^2
    println(io, "Superoperator: $sup_size with $nvals stored entries (density $density)")

    println(io, "Total memory: $m\n")

    show(io, model.params)

end

Base.show(io::IO, state::State) = begin
    n = size(state.ρ, 1)
    type = typeof(state.ρ)
    m = Base.format_bytes(Base.summarysize(state))
    println(io, "Dicke state for $(nspins(n)) spins. $n × $n matrix of type $type. Total memory: $m")
end

Base.show(io::IO, x::ModelParameters) = begin
    println(io, "Model Parameters")
    data = ["Nj"    x.Nj    "Number of spins";
            "κ_i"   x.kind  "Independent noise rate";
            "κ_c"   x.kcoll "Collective noise rate";
            "ω"     x.omega "Frequency";
            "η"     x.eta   "Measurement efficiency"]
    pretty_table(io, data, ["Param.", "Value", "Description"], alignment=[:l, :r, :l])

    println(io, "\nSimulation Parameters")
    data = ["Tfinal" x.Tfinal "Final time";
            "dt"     x.dt     "Time step";
            "Ntime"  x.Ntime  "N. of simulation points";
            "outpoints" x.outpoints "N. of output points"]
    pretty_table(io, data, ["Parameter", "Value", "Description"], alignment=[:l, :r, :l])
end

