# Sensor calibration pre-run (awes_ekf initialize_ekf / calibration_series and
# postprocessing.find_pitot_calibration / find_offset).
#
# The filter is run without the sensor over a window of the flight; the sensor
# is then calibrated against the filter's estimate of the same quantity.

"Calibration coefficients of a pitot tube: va_meas^2 = k va_ref^2 + b."
struct PitotCalibration
    k::Float64
    b::Float64
end

"Calibrated apparent wind speed from the indicated one."
apply(c::PitotCalibration, va) = sqrt(max((va^2 - c.b) / c.k, 0.0))

"""
Least-squares pitot calibration in dynamic-pressure space (through the origin,
or with the transducer zero `b` when `fit_zero`), k clipped to `k_range`.
"""
function find_pitot_calibration(reference_va, measured_va; fit_zero=false, k_range=(0.25, 4.0))
    mask = isfinite.(reference_va) .& isfinite.(measured_va) .& (reference_va .> 0) .& (measured_va .> 0)
    any(mask) || error("No valid samples to calibrate the pitot tube")
    q_ref = reference_va[mask] .^ 2
    q_meas = measured_va[mask] .^ 2
    if fit_zero
        k, b = [q_ref ones(length(q_ref))] \ q_meas
    else
        k, b = dot(q_ref, q_meas) / dot(q_ref, q_ref), 0.0
    end
    k_clipped = clamp(k, k_range...)
    k_clipped != k && @printf("Warning: pitot calibration coefficient %.4f clipped to %g\n", k, k_clipped)
    return PitotCalibration(k_clipped, b)
end

"Additive offset (grid search, 1000 values) minimising the mean squared difference."
function find_offset(signal1, signal2; offset_range=(-15.0, 15.0))
    offsets = range(offset_range[1], offset_range[2]; length=1000)
    mse(o) = mean(filter(!isnan, (signal1 .- (signal2 .+ o)) .^ 2))
    return offsets[argmin(mse.(offsets))]
end

"First and last sample (0-based, end exclusive) of the calibration window."
function calibration_window(inputs, sim::SimulationConfig)
    n = length(inputs)
    dt = mean(s.timestep for s in inputs[1:min(100, n)])
    per_minute = 60.0 / dt
    start = trunc(Int, sim.calibration_start_minutes * per_minute)
    duration = trunc(Int, sim.calibration_duration_minutes * per_minute)
    margin = trunc(Int, sim.calibration_end_margin_minutes * per_minute)
    stop = min(start + duration, max(n - margin, 0))
    if stop <= start
        start, stop = n ÷ 4, (3n) ÷ 4
        println("Warning: flight too short for the configured calibration window, " *
                "falling back to samples $start to $stop")
    end
    return start, stop
end

"""
Run the filter without `variable` over the calibration window; return the
estimated and measured series on the window (NaN where a step failed).
"""
function calibration_series(cfg::Config, inputs, x0, variable::Symbol)
    output_column = variable == :bridle_angle_of_attack ? "kite_angle_of_attack" : String(variable)
    start, stop = calibration_window(inputs, cfg.sim)
    dt = mean(s.timestep for s in inputs[1:min(100, length(inputs))])
    @printf("Calibrating %s on minutes %.1f to %.1f of the flight\n", variable, start * dt / 60, stop * dt / 60)

    sim = deepcopy(cfg.sim)
    setfield!(sim.obs, variable, false)
    sim.enforce_vertical_wind_to_0 = true
    model = SystemModel(cfg, sim)
    ekf = ExtendedKalmanFilter(model, model_stdv(cfg.tuning, sim), meas_stdv(cfg.tuning, sim), x0,
        input_vector(model, inputs[1]))

    estimated = fill(NaN, stop)
    n_failed = 0
    last_good = nothing
    t_start = time()
    for k in 1:stop
        try
            step!(ekf, inputs[k])
            estimated[k] = ekf_output(ekf)[output_column]
            last_good = (copy(ekf.x_k1_k1), copy(ekf.P_k1_k1))
        catch e
            # Keep the series aligned; roll back a non-finite state
            n_failed += 1
            n_failed <= 10 && println("  Skipping timestep $(k-1) of the pre-run: $(sprint(showerror, e))")
            if last_good !== nothing && !all(isfinite, ekf.x_k1_k1)
                ekf.x_k1_k1 = copy(last_good[1])
                ekf.P_k1_k1 = copy(last_good[2])
            end
        end
        if k > 1 && (k - 1) % 3000 == 0
            @printf("  Pre-run at minute %.1f, %.1f s elapsed\n", (k - 1) * dt / 60, time() - t_start)
            flush(stdout)
        end
    end
    n_failed > 0 && println("  $n_failed of $stop pre-run samples were unusable and are left out of the fit")
    measured = [Float64(getfield(s, variable)) for s in inputs[1:stop]]
    return estimated[start+1:end], measured[start+1:end]
end

"""
    calibrate_sensors!(cfg, inputs, x0) -> Dict

Pre-run calibration of the enabled sensors; corrects `inputs` in place and
returns the calibrations found (`:pitot => PitotCalibration`, `:aoa_offset`).
"""
function calibrate_sensors!(cfg::Config, inputs, x0)
    sim = cfg.sim
    found = Dict{Symbol,Any}()
    if sim.obs.kite_apparent_windspeed && sim.calibrate_apparent_windspeed
        estimated, measured = calibration_series(cfg, inputs, x0, :kite_apparent_windspeed)
        usable = isfinite.(estimated) .& isfinite.(measured)
        if any(usable)
            println("Fitting kite_apparent_windspeed on $(sum(usable)) of $(length(estimated)) samples of the calibration window")
            c = find_pitot_calibration(estimated, measured; fit_zero=sim.pitot_calibration_fit_zero)
            @printf("Pitot calibration for kite_apparent_windspeed: k = %.4f, zero = %.4f m2/s2 (speed scale %.4f)\n",
                c.k, c.b, 1 / sqrt(c.k))
            apply_pitot!(inputs, c)
            found[:pitot] = c
        else
            println("No usable samples to calibrate kite_apparent_windspeed, skipping")
        end
    end
    if sim.obs.bridle_angle_of_attack && sim.calibrate_angle_of_attack
        estimated, measured = calibration_series(cfg, inputs, x0, :bridle_angle_of_attack)
        if any(isfinite.(estimated) .& isfinite.(measured))
            offset = find_offset(estimated, measured)
            println("Offset for bridle_angle_of_attack: $offset")
            for s in inputs
                s.bridle_angle_of_attack += offset
            end
            found[:aoa_offset] = offset
        else
            println("No usable samples to calibrate bridle_angle_of_attack, skipping")
        end
    end
    return found
end

function apply_pitot!(inputs, c::PitotCalibration)
    for s in inputs
        s.kite_apparent_windspeed = apply(c, s.kite_apparent_windspeed)
    end
    return inputs
end
