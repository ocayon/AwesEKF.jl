# Point-mass kite model with tether: the process model f(x, u, x0) and the
# observation model h(x, u, x0) of awes_ekf (PointMassEKF.get_fx and
# ObservationModel.get_hx).
#
# `x0` is the state at the start of the step. The tether shape is evaluated with
# the kite position, velocity and wind of `x0` and only the tether states
# (elevation, azimuth, length) of `x`; the filter's Jacobians differentiate with
# respect to `x` alone, holding `x0` fixed, exactly as the Python code does.

struct SystemModel
    sim::SimulationConfig
    layout::Layout
    tether::TetherModel
    mass::Float64
    area::Float64
end

function SystemModel(cfg::Config, sim::SimulationConfig=cfg.sim)
    tm = TetherModel(cfg.tether, cfg.kcu, cfg.kite, sim.obs)
    return SystemModel(sim, Layout(sim), tm, cfg.kite.mass, cfg.kite.area)
end

"Wind velocity of state vector `x` at the kite."
function wind_velocity(m::SystemModel, x)
    sx = m.layout.sx
    if m.sim.log_profile
        return log_wind_velocity(x[sx["uf"]], x[sx["wdir"]], x[sx["vw_2"]], x[sx["r_2"]])
    end
    return xvec3(x, m.layout, "vw")
end

"Measured accelerations/velocities the tether model takes from the input vector."
function tether_kinematic_inputs(m::SystemModel, u)
    o = m.sim.obs
    l = m.layout
    z = zero(Vec3{eltype(u)})
    a_kite = o.kite_acceleration ? uvec3(u, l, "a_kite") : z
    a_kcu = o.kcu_acceleration ? uvec3(u, l, "a_kcu") : z
    v_kcu = o.kcu_velocity ? uvec3(u, l, "v_kcu") : z
    return a_kite, a_kcu, v_kcu
end

"Tether shape with the tether states of `x`, the kite kinematics of `x0`."
function tether_shape_for(m::SystemModel, x, u, x0)
    l = m.layout
    sx = l.sx
    a_kite, a_kcu, v_kcu = tether_kinematic_inputs(m, u)
    return tether_shape(m.tether, x[sx["elevation_0"]], x[sx["azimuth_0"]], x[sx["tether_length"]],
        u[l.ux["ground_tether_force"]], xvec3(x0, l, "r"), xvec3(x0, l, "v"), wind_velocity(m, x0),
        a_kite, a_kcu, v_kcu)
end

"""
Full aerodynamic coefficients of the wing: the CL/CD/CS walk states plus the
deterministic steering-dependent and drag-polar terms that are enabled.
"""
function effective_coefficients(sim::SimulationConfig, sx, x, us)
    CL = x[sx["CL"]]
    CD = x[sx["CD"]]
    CS = x[sx["CS"]]
    if sim.steering_dependent_clcd
        CL = CL + x[sx["k_cl_us"]] * abs(us)
        CD = CD + x[sx["k_cd_us"]] * us * us
    end
    if sim.steering_dependent_cl_asym
        CL = CL + x[sx["k_cl_us_odd"]] * us
    end
    if sim.drag_polar
        # Induced drag follows the complete lift coefficient
        CD = CD + x[sx["k_cd_cl2"]] * CL * CL
    end
    if sim.steering_dependent_cs
        # Aerodynamic sideslip proportional to the steering input
        CS = CL * tan(x[sx["k_phi_us"]] * us) + CS
    end
    return CL, CD, CS
end

steering_input(m::SystemModel, u) = haskey(m.layout.ux, "us") ? u[m.layout.ux["us"]] : zero(eltype(u))

"State derivative f(x, u, x0)."
function dynamics(m::SystemModel, x, u, x0)
    sim = m.sim
    l = m.layout
    sx = l.sx
    T = promote_type(eltype(x), eltype(u), eltype(x0))

    shape = tether_shape_for(m, x, u, x0)
    tether_force = shape.tension_kite

    r = xvec3(x, l, "r")
    v = xvec3(x, l, "v")
    va = wind_velocity(m, x) - v
    dir_D = va / vnorm(va)
    t_unit = tether_force / vnorm(tether_force)
    dir_L = t_unit - dot(t_unit, dir_D) * dir_D
    dir_S = cross(dir_L, dir_D)

    us = steering_input(m, u)
    CL, CD, CS = effective_coefficients(sim, sx, x, us)
    q = 0.5 * RHO * m.area * vnorm(va)^2
    L = CL * q * dir_L
    D = CD * q * dir_D
    S = CS * q * dir_S
    Fg = Vec3(0.0, 0.0, -m.mass * G)
    F = -tether_force + L + D + S + Fg
    if sim.obs.kite_thrust_force
        F = F + uvec3(u, l, "thrust")
    end
    acceleration = F / m.mass

    # Tether elevation/azimuth rates from the tangential kite velocity
    r0 = xvec3(x0, l, "r")
    v0 = xvec3(x0, l, "v")
    r_norm = vnorm(r0)
    r_normal = r0 / r_norm
    vert = project_onto_plane(Vec3(0.0, 0.0, 1.0), r_normal)
    horz = cross(vert, r_normal)
    elevation_rate = dot(v0, vert) / r_norm
    azimuth_rate = dot(v0, horz) / (r_norm * cos(x[sx["elevation_0"]]))

    CL_dot = zero(T)
    CD_dot = zero(T)
    if sim.obs.dynamic_depower
        delta_up = u[l.ux["delta_up"]]
        CL_dot = x[sx["k_cl_up"]] * delta_up
        # With the drag polar the depower-driven CD change flows through CL
        CD_dot = sim.drag_polar ? zero(T) : x[sx["k_cd_up"]] * delta_up
    end

    fx = zeros(T, nstates(l))
    fx[1:3] .= v
    fx[4:6] .= acceleration
    fx[sx["CL"]] = CL_dot
    fx[sx["CD"]] = CD_dot
    fx[sx["tether_length"]] = u[l.ux["reelout_speed"]]
    fx[sx["elevation_0"]] = elevation_rate
    fx[sx["azimuth_0"]] = azimuth_rate
    return fx
end

"Predicted measurements h(x, u, x0), in the order of `measurement_vector`."
function observation(m::SystemModel, x, u, x0)
    sim = m.sim
    o = sim.obs
    l = m.layout
    sx = l.sx
    T = promote_type(eltype(x), eltype(u), eltype(x0))

    shape = tether_shape_for(m, x, u, x0)
    r = xvec3(x, l, "r")
    v = xvec3(x, l, "v")
    va = wind_velocity(m, x) - v

    h = T[]
    o.kite_position && append!(h, r)
    o.kite_velocity && append!(h, v)
    # Tether-model closure: the modelled kite position must match the state
    append!(h, r - shape.kite_position)
    o.tether_length && push!(h, x[sx["tether_length"]] - x[sx["tether_length_offset"]])
    o.tether_elevation && push!(h, x[sx["elevation_0"]] - x[sx["tether_elevation_offset"]])
    o.tether_azimuth && push!(h, x[sx["azimuth_0"]] - x[sx["tether_azimuth_offset"]])
    sim.enforce_vertical_wind_to_0 && push!(h, x[sx["vw_2"]])
    o.kite_apparent_windspeed && push!(h, vnorm(va))
    if o.bridle_angle_of_attack
        r0 = xvec3(x0, l, "r")
        v0 = xvec3(x0, l, "v")
        frames = tether_derived(m.tether, shape, r0, v0)
        push!(h, airflow_angles(frames.dcm_b2w, va)[1])
    end
    return h
end

"Input vector u of one sample (ExtendedKalmanFilter.update_input_vector)."
function input_vector(m::SystemModel, s)
    o = m.sim.obs
    u = [s.tether_reelout_speed, s.tether_force]
    o.kite_acceleration && append!(u, s.kite_acceleration)
    o.kcu_acceleration && append!(u, s.kcu_acceleration)
    o.kcu_velocity && append!(u, s.kcu_velocity)
    o.kite_thrust_force && append!(u, s.kite_thrust_force)
    uses_steering(m.sim) && push!(u, isnan(s.steering_input) ? 0.0 : s.steering_input)
    o.dynamic_depower && append!(u, (s.depower_input, s.delta_up))
    return u
end

"Measurement vector z of one sample (ExtendedKalmanFilter.update_measurement_vector)."
function measurement_vector(m::SystemModel, s)
    sim = m.sim
    o = sim.obs
    z = Float64[]
    o.kite_position && append!(z, s.kite_position)
    o.kite_velocity && append!(z, s.kite_velocity)
    append!(z, (0.0, 0.0, 0.0))
    o.tether_length && push!(z, s.tether_length)
    o.tether_elevation && push!(z, s.tether_elevation_ground)
    o.tether_azimuth && push!(z, s.tether_azimuth_ground)
    sim.enforce_vertical_wind_to_0 && push!(z, 0.0)
    o.kite_apparent_windspeed && push!(z, s.kite_apparent_windspeed)
    o.bridle_angle_of_attack && push!(z, s.bridle_angle_of_attack)
    return z
end
