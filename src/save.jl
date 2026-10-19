# Results file: the EKF-AWE HDF5 layout (groups ekf_output, flight_data,
# config_data), plus the state history and run metadata.

h5value(v::Bool) = Int8(v)
h5value(v::Real) = v
h5value(v::AbstractString) = String(v)
h5value(v::AbstractVector) = all(x -> x isa Real, v) ? [Float64(x) for x in v] : string.(v)
h5value(v::Nothing) = "None"
h5value(v) = string(v)

function write_dict!(group, d::AbstractDict)
    for (key, value) in d
        if value isa AbstractDict
            write_dict!(create_group(group, string(key)), value)
        else
            attrs(group)[string(key)] = h5value(value)
        end
    end
end

"""
    save_results(path, result, flight_data, cfg)

Write `ekf_output` (one dataset per output column), `flight_data` (the input
rows of the kept steps), `config_data` (the merged configuration), `state`
(x, diag P, names) and `meta` (x0, kept/failed samples, IEKF iterations,
calibrations, timing).
"""
function save_results(path::AbstractString, r::FilterResult, fd::FlightData, cfg::Config)
    mkpath(dirname(abspath(path)))
    rows = r.kept .+ 1
    h5open(path, "w") do f
        g = create_group(f, "ekf_output")
        attrs(g)["description"] = "Extended Kalman Filter output (AwesEKF.jl)"
        for c in OUTPUT_COLUMNS
            g[c] = r.outputs[c]
        end
        g = create_group(f, "flight_data")
        attrs(g)["description"] = "Experimental data of the kept steps"
        for name in fd.names
            col = fd[name][rows]
            clean = replace(name, " " => "_", "(" => "", ")" => "", "/" => "_", "°" => "deg")
            g[clean] = col
        end
        write_dict!(create_group(f, "config_data"), cfg.raw)
        g = create_group(f, "state")
        g["x"] = permutedims(r.x)          # HDF5.jl is column-major: stored as steps x states
        g["P_diag"] = permutedims(r.P_diag)
        g["names"] = r.state_names
        g = create_group(f, "meta")
        g["x0"] = r.x0
        g["kept"] = r.kept
        g["failed"] = r.failed
        g["iekf_iterations"] = r.iekf_iterations
        pitot = get(r.calibrations, :pitot, nothing)
        g["pitot_calibration"] = pitot === nothing ? [NaN, NaN] : [pitot.k, pitot.b]
        attrs(g)["seconds_prerun"] = r.seconds_prerun
        attrs(g)["seconds_main"] = r.seconds_main
        attrs(g)["n_samples"] = length(fd)
    end
    println("Saved $path")
    return path
end
