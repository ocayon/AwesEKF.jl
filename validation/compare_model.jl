# Unit-level comparison of the Julia model functions with the Python ones, on the
# probe file written by python_model_probe.py.
#
#   julia --project validation/compare_model.jl <probe.h5> <csv> <config_folder> <system_yaml> <start> <end>

using AwesEKF, HDF5, LinearAlgebra, Printf
const A = AwesEKF

relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

function main(probe, csv, folder, system_yaml, start_min, end_min)
    cfg = load_config(folder; system_yaml=system_yaml)
    fd = slice_minutes(read_flight_csv(csv), start_min, end_min)
    inputs = create_inputs(fd, cfg.sim)
    model = SystemModel(cfg)
    solver = A.TetherShapeSolver()
    x0 = initial_state(solver, model, inputs[1]; wind_velocity=cfg.sim.initial_wind_velocity)

    worst = 0.0
    report(name, a, b) = begin
        e = relerr(a, b)
        worst = max(worst, e)
        @printf("  %-12s rel. error %.2e\n", name, e)
    end
    h5open(probe) do f
        @printf("KCU drag areas: axial %.3e, crossflow %.3e (rel. error)\n",
            abs(model.tether.kcu.cd_area_axial / read_attribute(f, "cdt_area") - 1),
            abs(model.tether.kcu.cd_area_crossflow / read_attribute(f, "cdp_area") - 1))
        println("Initial state (tether solve: scipy TRF xtol 1e-5 vs converged LM)")
        report("x0", x0, read(f, "x0"))
        for name in sort(filter(startswith("probe_"), keys(f)); by=n -> parse(Int, split(n, "_")[2]))
            g = f[name]
            k = read_attribute(g, "sample")
            x = read(g, "x")
            u = read(g, "u")
            println("$name (sample $k)")
            report("u", A.input_vector(model, inputs[k+1]), u)
            report("f", A.dynamics(model, x, u, x), read(g, "f"))
            report("df/dx", A.state_jacobian(model, x, u), permutedims(read(g, "F")))
            h, H = A.observation_and_jacobian(model, x, u)
            report("h", h, read(g, "h"))
            report("dh/dx", H, permutedims(read(g, "H")))
            report("propagate", A.propagate(model, x, u, read_attribute(g, "ts")), read(g, "x_next"))
            # Output row through a filter object holding x and u
            ekf = A.ExtendedKalmanFilter(model, A.model_stdv(cfg.tuning, cfg.sim), A.meas_stdv(cfg.tuning, cfg.sim), x, u)
            row = A.ekf_output(ekf)
            py = Dict(k => read_attribute(g["output"], k) for k in keys(attrs(g["output"])))
            bad = String[]
            for (key, value) in py
                key in ("nis", "norm_epsilon_norm", "mahalanobis_distance", "exceeds_threshold") && continue
                haskey(row, key) || (push!(bad, "$key missing"); continue)
                (isnan(value) && isnan(row[key])) && continue
                e = abs(row[key] - value) / max(abs(value), 1e-12)
                e > 1e-9 && push!(bad, @sprintf("%s %.3e vs %.3e", key, row[key], value))
            end
            println("  outputs     ", isempty(bad) ? "all agree to 1e-9" : join(bad, "; "))
        end
    end
    @printf("Worst relative error: %.2e\n", worst)
end

main(ARGS[1], ARGS[2], ARGS[3], ARGS[4], parse(Float64, ARGS[5]), parse(Float64, ARGS[6]))
