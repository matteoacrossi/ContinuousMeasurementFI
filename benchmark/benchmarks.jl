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
for n in (15, 20, 25, 30)
    SUITE[n] = @benchmarkable Eff_QFI_HD_Dicke($n, 1, .5, 0.001, outsteps=50)
end