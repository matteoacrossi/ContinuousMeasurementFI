using BenchmarkTools
using ContinuousMeasurementFI
using Logging

# To remove output
disable_logging(Logging.Warn)

# Define a parent BenchmarkGroup to contain our suite
const SUITE = BenchmarkGroup()

# Precompile
Eff_QFI_HD_Dicke(2, 1, .01, 0.001);

# Add some benchmarks to the "trig" group
for n in (20, 40, 60)
    SUITE[n] = @benchmarkable Eff_QFI_HD_Dicke($n, 1, .5, 0.001, outsteps=10)
end