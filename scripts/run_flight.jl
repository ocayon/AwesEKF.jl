# Run the EKF on a processed flight CSV and save the results file.
#
#   julia --project scripts/run_flight.jl --csv <file> --config <kite folder> \
#         [--system-yaml system_flown_2019.yaml] [--start 60] [--end 100] --out results.h5 \
#         [--x0-from ref.h5] [--pitot-from ref.h5]
#
# --x0-from / --pitot-from take the initial state / pitot calibration from another
# results file (meta/x0, meta/pitot_calibration) instead of computing them: used
# to run in lockstep with the Python reference.

using AwesEKF, HDF5

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("Unexpected argument $(args[i])")
        opts[args[i][3:end]] = args[i+1]
        i += 2
    end
    return opts
end

opts = parse_args(ARGS)
kwargs = Dict{Symbol,Any}()
haskey(opts, "x0-from") && (kwargs[:x0] = h5read(opts["x0-from"], "meta/x0"))
if haskey(opts, "pitot-from")
    k, b = h5read(opts["pitot-from"], "meta/pitot_calibration")
    kwargs[:pitot] = AwesEKF.PitotCalibration(k, b)
end
run_flight(opts["csv"], opts["config"]; system_yaml=get(opts, "system-yaml", nothing),
    start_minute=parse(Float64, get(opts, "start", "0")), end_minute=parse(Float64, get(opts, "end", "0")),
    out=opts["out"], kwargs...)
