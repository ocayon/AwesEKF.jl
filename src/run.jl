# Running the filter over a flight (EKF-AWE examples/run_analysis.py, without
# the interactive prompts and the postprocessing).

"Results of a run: per kept step outputs and states, and run metadata."
struct FilterResult
    outputs::Dict{String,Vector{Float64}}
    x::Matrix{Float64}             # corrected state per kept step (steps x states)
    P_diag::Matrix{Float64}
    state_names::Vector{String}
    kept::Vector{Int}              # 0-based sample index of each output row
    failed::Vector{Int}            # 0-based samples where the filter re-initialised
    iekf_iterations::Vector{Int}
    x0::Vector{Float64}
    calibrations::Dict{Symbol,Any}
    seconds_prerun::Float64
    seconds_main::Float64
end

"""
    run_filter(cfg, inputs; x0=nothing, pitot=nothing, calibrate=true)

Initial state, sensor calibration pre-run and the filter loop. A step that fails
(integrator failure, singular innovation covariance, ...) re-initialises the
filter from a fresh tether-shape solve at that sample, as the Python code does.

`x0` and `pitot` (a `PitotCalibration`) replace the initial-state solve and the
pitot pre-run, e.g. to run in lockstep with another implementation.
"""
function run_filter(cfg::Config, inputs::Vector{EKFInput}; x0=nothing, pitot=nothing, calibrate=true)
    model = SystemModel(cfg)
    solver = TetherShapeSolver()
    q = model_stdv(cfg.tuning, cfg.sim)
    r = meas_stdv(cfg.tuning, cfg.sim)
    x0 = x0 === nothing ? initial_state(solver, model, inputs[1]; wind_velocity=cfg.sim.initial_wind_velocity) :
         collect(Float64, x0)
    if x0 !== nothing && solver.guess === nothing
        # Seed the shape solver for later re-initialisations
        sx = model.layout.sx
        solver.guess = [x0[sx["elevation_0"]], x0[sx["azimuth_0"]], x0[sx["tether_length"]]]
    end

    t_start = time()
    calibrations = Dict{Symbol,Any}()
    if pitot !== nothing
        apply_pitot!(inputs, pitot)
        calibrations[:pitot] = pitot
    elseif calibrate
        calibrations = calibrate_sensors!(cfg, inputs, x0)
    end
    seconds_prerun = time() - t_start

    ekf = ExtendedKalmanFilter(model, q, r, x0, input_vector(model, inputs[1]))
    n = length(inputs)
    outputs = Dict(c => Float64[] for c in OUTPUT_COLUMNS)
    states = Vector{Vector{Float64}}()
    p_diag = Vector{Vector{Float64}}()
    kept, failed, iterations = Int[], Int[], Int[]

    t_start = time()
    for k in 1:n
        try
            step!(ekf, inputs[k])
            row = ekf_output(ekf)
            for c in OUTPUT_COLUMNS
                push!(outputs[c], row[c])
            end
            push!(states, copy(ekf.x_k1_k1))
            push!(p_diag, diag(ekf.P_k1_k1))
            push!(kept, k - 1)
            push!(iterations, ekf.iekf_iterations)
        catch e
            e isa InterruptException && rethrow()
            println("Step $(k-1) failed: $(sprint(showerror, e))")
            push!(failed, k - 1)
            x_restart = try
                initial_state(solver, model, inputs[k])
            catch
                ekf.x_k1_k1
            end
            ekf = ExtendedKalmanFilter(model, q, r, x_restart, input_vector(model, inputs[k]))
        end
        if (k - 1) % 3000 == 0
            @printf("step %d/%d  %.1f s\n", k - 1, n, time() - t_start)
            flush(stdout)
        end
    end
    seconds_main = time() - t_start
    @printf("Pre-run %.1f s, main loop %.1f s\n", seconds_prerun, seconds_main)

    x = isempty(states) ? zeros(0, nstates(model.layout)) : permutedims(reduce(hcat, states))
    P = isempty(p_diag) ? zeros(0, nstates(model.layout)) : permutedims(reduce(hcat, p_diag))
    return FilterResult(outputs, x, P, copy(model.layout.state_names), kept, failed, iterations,
        x0, calibrations, seconds_prerun, seconds_main)
end

"""
    run_flight(csv, config_folder; system_yaml=nothing, start_minute=0, end_minute=0,
               out=nothing, kwargs...)

Load the configuration and a processed flight CSV, run the filter on the minute
slice and, with `out`, save the results file. Keyword arguments are passed to
`run_filter`. Returns `(result, flight_data, cfg)`.
"""
function run_flight(csv::AbstractString, config_folder::AbstractString; system_yaml=nothing,
    start_minute=0, end_minute=0, out=nothing, kwargs...)
    cfg = load_config(config_folder; system_yaml=system_yaml)
    println("Using system config: $(cfg.raw["system_yaml_used"]); kite model: $(cfg.kite.model_name)")
    fd = slice_minutes(read_flight_csv(csv), start_minute, end_minute)
    inputs = create_inputs(fd, cfg.sim)
    result = run_filter(cfg, inputs; kwargs...)
    out === nothing || save_results(out, result, fd, cfg)
    return result, fd, cfg
end
