# State, input and measurement vector layouts. The order matches awes_ekf
# (PointMassEKF.get_state / get_input, ExtendedKalmanFilter.update_measurement_vector),
# so state vectors are interchangeable between the two implementations.

struct Layout
    state_names::Vector{String}
    input_names::Vector{String}
    sx::Dict{String,Int}   # state name -> index
    ux::Dict{String,Int}   # input name -> index
end

function Layout(sim::SimulationConfig)
    o = sim.obs
    s = ["r_0", "r_1", "r_2", "v_0", "v_1", "v_2"]
    append!(s, sim.log_profile ? ["uf", "wdir", "vw_2"] : ["vw_0", "vw_1", "vw_2"])
    append!(s, ["CL", "CD", "CS", "tether_length", "elevation_0", "azimuth_0"])
    o.tether_length && push!(s, "tether_length_offset")
    o.tether_elevation && push!(s, "tether_elevation_offset")
    o.tether_azimuth && push!(s, "tether_azimuth_offset")
    o.dynamic_depower && append!(s, ["k_cl_up", "k_cd_up"])
    sim.steering_dependent_cs && push!(s, "k_phi_us")
    sim.steering_dependent_clcd && append!(s, ["k_cl_us", "k_cd_us"])
    sim.steering_dependent_cl_asym && push!(s, "k_cl_us_odd")
    sim.drag_polar && push!(s, "k_cd_cl2")

    u = ["reelout_speed", "ground_tether_force"]
    o.kite_acceleration && append!(u, ["a_kite_0", "a_kite_1", "a_kite_2"])
    o.kcu_acceleration && append!(u, ["a_kcu_0", "a_kcu_1", "a_kcu_2"])
    o.kcu_velocity && append!(u, ["v_kcu_0", "v_kcu_1", "v_kcu_2"])
    o.kite_thrust_force && append!(u, ["thrust_0", "thrust_1", "thrust_2"])
    uses_steering(sim) && push!(u, "us")
    o.dynamic_depower && append!(u, ["depower_input", "delta_up"])

    return Layout(s, u, Dict(n => i for (i, n) in enumerate(s)), Dict(n => i for (i, n) in enumerate(u)))
end

nstates(l::Layout) = length(l.state_names)
ninputs(l::Layout) = length(l.input_names)

# Typed accessors used by the models
@inline xvec3(x, l::Layout, prefix) = Vec3(x[l.sx[prefix*"_0"]], x[l.sx[prefix*"_1"]], x[l.sx[prefix*"_2"]])
@inline uvec3(u, l::Layout, prefix) = Vec3(u[l.ux[prefix*"_0"]], u[l.ux[prefix*"_1"]], u[l.ux[prefix*"_2"]])
