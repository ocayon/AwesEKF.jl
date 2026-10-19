# Flight data (processed CSV, EKF-AWE input spec) -> one EKFInput per sample
# (awes_ekf.load_data.create_input_from_csv).

"Columns of a processed flight CSV, in file order."
struct FlightData
    names::Vector{String}
    columns::Dict{String,AbstractVector}
end

Base.length(fd::FlightData) = length(fd.columns[fd.names[1]])
Base.haskey(fd::FlightData, name) = haskey(fd.columns, name)
Base.getindex(fd::FlightData, name::AbstractString) = fd.columns[name]

function Base.getindex(fd::FlightData, rows::AbstractVector{<:Integer})
    return FlightData(copy(fd.names), Dict(n => fd.columns[n][rows] for n in fd.names))
end

"""
    read_flight_csv(path)

Read a processed flight CSV. Numeric columns become `Vector{Float64}` (missing
values NaN, as pandas reads them); other columns stay strings. An `index`
column with the 0-based row number is prepended (pandas `reset_index`).
"""
function read_flight_csv(path::AbstractString)
    file = CSV.File(path; ntasks=1)
    names = String.(propertynames(file))
    columns = Dict{String,AbstractVector}()
    for name in names
        col = getproperty(file, Symbol(name))
        T = nonmissingtype(eltype(col))
        if T <: Real
            columns[name] = Float64[ismissing(v) ? NaN : Float64(v) for v in col]
        else
            columns[name] = String[ismissing(v) ? "" : string(v) for v in col]
        end
    end
    n = length(columns[names[1]])
    columns["index"] = collect(0.0:n-1)
    return FlightData(vcat("index", names), columns)
end

"""
    slice_minutes(fd, start_minute, end_minute)

Rows between two flight minutes, with the row arithmetic of EKF-AWE's
run_analysis.py (0 = no limit).
"""
function slice_minutes(fd::FlightData, start_minute::Real, end_minute::Real)
    (start_minute > 0 || end_minute > 0) || return fd
    dt = mean(filter(!isnan, diff(fd["time"])))
    first = trunc(Int, start_minute * 60 / dt)
    if end_minute > 0 && end_minute <= length(fd) / 60 * dt
        last = trunc(Int, end_minute * 60 / dt)
    else
        last = length(fd)
    end
    return fd[first+1:last]
end

"One sample of filter input. Optional scalars are NaN when not measured."
Base.@kwdef mutable struct EKFInput
    kite_position::Vec3{Float64}
    kite_velocity::Vec3{Float64}
    kite_acceleration::Vec3{Float64}
    kcu_velocity::Vec3{Float64} = zero(Vec3{Float64})
    kcu_acceleration::Vec3{Float64} = zero(Vec3{Float64})
    kite_thrust_force::Vec3{Float64} = zero(Vec3{Float64})
    timestep::Float64
    tether_force::Float64
    tether_length::Float64
    tether_reelout_speed::Float64
    kite_apparent_windspeed::Float64 = NaN
    bridle_angle_of_attack::Float64 = NaN
    tether_elevation_ground::Float64 = NaN
    tether_azimuth_ground::Float64 = NaN
    steering_input::Float64 = NaN
    depower_input::Float64 = NaN
    delta_up::Float64 = NaN
end

"Python's builtin max over a sequence (a NaN is skipped unless it comes first)."
function python_max(v)
    m = v[1]
    for x in Iterators.drop(v, 1)
        x > m && (m = x)
    end
    return m
end

function column_or(fd, name, n, required::Bool, what)
    haskey(fd, name) && return fd[name]
    required && error("No $what data found, but required by the config file")
    return zeros(n)
end

function vec3_columns(fd, prefix, n, required::Bool, what)
    names = ["$(prefix)_x", "$(prefix)_y", "$(prefix)_z"]
    if all(c -> haskey(fd, c), names)
        return [Vec3(fd[names[1]][i], fd[names[2]][i], fd[names[3]][i]) for i in 1:n]
    end
    required && error("No $what data found, but required by the config file")
    return fill(zero(Vec3{Float64}), n)
end

"""
    create_inputs(fd, sim)

Filter inputs for every sample: measured signals, the frozen pitot calibration
when configured, normalised and lag-filtered steering, normalised depower and
its rate.
"""
function create_inputs(fd::FlightData, sim::SimulationConfig)
    n = length(fd)
    o = sim.obs
    time = fd["time"]

    kite_position = vec3_columns(fd, "kite_position", n, true, "kite position")
    kite_velocity = vec3_columns(fd, "kite_velocity", n, true, "kite velocity")
    kite_acceleration = vec3_columns(fd, "kite_acceleration", n, o.kite_acceleration, "kite acceleration")
    kcu_velocity = vec3_columns(fd, "kcu_velocity", n, o.kcu_velocity, "KCU velocity")
    kcu_acceleration = vec3_columns(fd, "kcu_acceleration", n, o.kcu_acceleration, "KCU acceleration")
    thrust = vec3_columns(fd, "thrust_force", n, o.kite_thrust_force, "thrust force")

    haskey(fd, "ground_tether_force") || error("No tether force data found")
    haskey(fd, "tether_reelout_speed") || error("No tether reelout speed data found")
    tether_force = fd["ground_tether_force"]
    tether_reelout_speed = fd["tether_reelout_speed"]
    tether_length = column_or(fd, "tether_length", n, o.tether_length, "tether length")
    va = column_or(fd, "kite_apparent_windspeed", n, o.kite_apparent_windspeed, "apparent wind speed")
    aoa = column_or(fd, "bridle_angle_of_attack", n, o.bridle_angle_of_attack, "angle of attack")
    elevation = column_or(fd, "tether_elevation_ground", n, o.tether_elevation, "tether elevation")
    azimuth = column_or(fd, "tether_azimuth_ground", n, o.tether_azimuth, "tether azimuth")

    # Frozen pitot calibration: va_meas^2 = k va_ref^2 + b, solved for va_ref
    if sim.pitot_calibration_k !== nothing
        k, b = sim.pitot_calibration_k, sim.pitot_calibration_b
        va = sqrt.(max.((va .^ 2 .- b) ./ k, 0.0))
        @printf("Frozen pitot calibration from config: k = %.4f, zero = %.4f m2/s2\n", k, b)
    end

    if haskey(fd, "kcu_actual_depower")
        d = fd["kcu_actual_depower"]
        depower = d ./ python_max(abs.(d))
    else
        println("No depower input data found")
        depower = zeros(n)
    end
    if haskey(fd, "kcu_actual_steering")
        s = fd["kcu_actual_steering"]
        steering = s ./ python_max(abs.(s))
    else
        println("No steering input data found")
        steering = zeros(n)
    end

    # First-order lag: the aero response lags the measured steering
    tau = sim.steering_input_lag
    if tau > 0
        filtered = copy(steering)
        for i in 2:n
            dt_i = time[i] - time[i-1]
            alpha = dt_i / (tau + dt_i)
            filtered[i] = filtered[i-1] + alpha * (steering[i] - filtered[i-1])
        end
        steering = filtered
        @printf("Steering input filtered with a %.2f s first-order lag\n", tau)
    end

    timestep = numpy_gradient(time)
    delta_up = numpy_gradient(depower, time)

    inputs = Vector{EKFInput}(undef, n)
    for i in 1:n
        a = kite_acceleration[i]
        # The tether model divides by the acceleration norm
        vnorm(a) < 1e-5 && (a = Vec3(1e-5, 1e-5, 1e-5))
        inputs[i] = EKFInput(; kite_position=kite_position[i], kite_velocity=kite_velocity[i],
            kite_acceleration=a, kcu_velocity=kcu_velocity[i], kcu_acceleration=kcu_acceleration[i],
            kite_thrust_force=thrust[i], timestep=timestep[i], tether_force=tether_force[i],
            tether_length=tether_length[i], tether_reelout_speed=tether_reelout_speed[i],
            kite_apparent_windspeed=va[i], bridle_angle_of_attack=aoa[i],
            tether_elevation_ground=elevation[i], tether_azimuth_ground=azimuth[i],
            steering_input=steering[i], depower_input=depower[i], delta_up=delta_up[i])
    end
    return inputs
end
