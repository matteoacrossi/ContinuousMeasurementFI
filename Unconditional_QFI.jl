include("Eff_QFI_PD.jl")

function Unconditional_QFI(
                    Tfinal::Number,                         # Final time
                    dt::Number,                             # Time step
                    H, dH,                                  # Hamiltonian and its derivative wrt ω
                    non_monitored_noise_op;                 # Non monitored noise operators
                    initial_state = ghz_state)              # Initial state

     t, fi, qfi = Eff_QFI_PD(               # Number of spins
        1,                                  # Number of trajectories
        Tfinal,                             # Final time
        dt,                                 # Time step
        H, dH,                              # Hamiltonian and its derivative wrt ω
        non_monitored_noise_op,            # Non monitored noise operators
        [],                                # Monitored noise operators
        initial_state = initial_state,
        η = 0.)

        return (t, qfi)
end
