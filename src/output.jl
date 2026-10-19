# Filter outputs per step (awes_ekf.ekf.ekf_output.create_ekf_output).

"""
Output columns, in the order of the Python EKFOutput dataclass followed by the
innovation statistics it appends. Columns the Python code never fills are kept
(NaN) so both result files have the same layout.
"""
const OUTPUT_COLUMNS = [
    "kite_position_x", "kite_position_y", "kite_position_z",
    "kite_velocity_x", "kite_velocity_y", "kite_velocity_z",
    "kite_roll", "kite_pitch", "kite_yaw", "kite_yaw_kin", "kite_elevation", "kite_thrust_force",
    "wind_speed_horizontal", "wind_direction", "wind_speed_vertical",
    "tether_length", "tether_elevation", "tether_azimuth",
    "tether_length_offset", "tether_elevation_offset", "tether_azimuth_offset",
    "tether_force_kite", "tether_roll", "tether_pitch", "tether_yaw",
    "radial_roll", "radial_pitch", "radial_yaw",
    "kite_apparent_windspeed", "kite_angle_of_attack", "kite_sideslip_angle",
    "wing_lift_coefficient", "wing_drag_coefficient", "wing_sideforce_coefficient",
    "tether_angle_of_attack", "tether_sideslip_angle",
    "tether_drag_coefficient", "kcu_drag_coefficient", "bridles_drag_coefficient",
    "steering_law_coefficient", "normalized_innovation_squared", "mahalanobis_distance",
    "normalized_residual_norm", "k_cl_up", "k_cd_up",
    "k_phi_us", "k_cl_us", "k_cd_us", "k_cl_us_odd", "k_cd_cl2",
    "wing_lift_coefficient_residual", "wing_parasitic_drag_coefficient",
    "wing_sideforce_coefficient_residual",
    "norm_epsilon_norm", "nis", "exceeds_threshold",
]

"""
    ekf_output(ekf)

Output row of the current corrected state. Note (kept for identical results
files): offsets and constants that are not states (`tether_elevation_offset`
without elevation measurement, `k_cl_up` without dynamic depower,
`kite_thrust_force`) repeat state 1, the kite x position, as in the Python code.
"""
function ekf_output(ekf::ExtendedKalmanFilter)
    m = ekf.model
    sim = m.sim
    x = ekf.x_k1_k1
    u = ekf.u
    sx = m.layout.sx
    state_or_first(name) = x[get(sx, name, 1)]

    r = xvec3(x, m.layout, "r")
    v = xvec3(x, m.layout, "v")
    vw = wind_velocity(m, x)
    shape = tether_shape_for(m, x, u, x)
    d = tether_derived(m.tether, shape, r, v)

    roll, pitch, yaw = euler_ned(d.dcm_b2w)
    yaw_kin = euler_ned(d.dcm_b2vel)[3]
    t_roll, t_pitch, t_yaw = euler_ned(d.dcm_t2w)
    r_roll, r_pitch, r_yaw = euler_ned(d.dcm_r2w)
    z_axis = d.dcm_b2w[:, 3]
    aoa, sideslip = airflow_angles(d.dcm_b2w, vw - v)
    CL, CD, CS = effective_coefficients(sim, sx, x, steering_input(m, u))

    row = Dict{String,Float64}(c => NaN for c in OUTPUT_COLUMNS)
    row["kite_position_x"], row["kite_position_y"], row["kite_position_z"] = r
    row["kite_velocity_x"], row["kite_velocity_y"], row["kite_velocity_z"] = v
    row["kite_roll"], row["kite_pitch"], row["kite_yaw"] = roll, pitch, yaw
    row["kite_yaw_kin"] = yaw_kin
    row["kite_elevation"] = atan(z_axis[3], vnorm(z_axis))
    row["kite_thrust_force"] = state_or_first("kite_thrust_force")
    row["wind_speed_horizontal"] = sqrt(vw[1]^2 + vw[2]^2)
    row["wind_direction"] = atan(vw[2], vw[1])
    row["wind_speed_vertical"] = vw[3]
    row["tether_length"] = x[sx["tether_length"]]
    row["tether_elevation"] = x[sx["elevation_0"]]
    row["tether_azimuth"] = x[sx["azimuth_0"]]
    row["tether_length_offset"] = state_or_first("tether_length_offset")
    row["tether_elevation_offset"] = state_or_first("tether_elevation_offset")
    row["tether_azimuth_offset"] = state_or_first("tether_azimuth_offset")
    row["tether_force_kite"] = vnorm(shape.tension_kite)
    row["tether_roll"], row["tether_pitch"], row["tether_yaw"] = t_roll, t_pitch, t_yaw
    row["radial_roll"], row["radial_pitch"], row["radial_yaw"] = r_roll, r_pitch, r_yaw
    row["kite_apparent_windspeed"] = vnorm(vw - v)
    row["kite_angle_of_attack"] = aoa
    row["kite_sideslip_angle"] = sideslip
    row["wing_lift_coefficient"], row["wing_drag_coefficient"], row["wing_sideforce_coefficient"] = CL, CD, CS
    row["wing_lift_coefficient_residual"] = x[sx["CL"]]
    row["wing_parasitic_drag_coefficient"] = x[sx["CD"]]
    row["wing_sideforce_coefficient_residual"] = x[sx["CS"]]
    row["tether_drag_coefficient"] = d.cd_tether
    row["kcu_drag_coefficient"] = d.cd_kcu
    row["bridles_drag_coefficient"] = d.cd_bridles
    row["k_cl_up"] = state_or_first("k_cl_up")
    row["k_cd_up"] = state_or_first("k_cd_up")
    for name in ("k_phi_us", "k_cl_us", "k_cd_us", "k_cl_us_odd", "k_cd_cl2")
        haskey(sx, name) && (row[name] = x[sx[name]])
    end
    row["mahalanobis_distance"] = ekf.mahalanobis_distance
    row["norm_epsilon_norm"] = ekf.norm_epsilon_norm
    row["nis"] = ekf.nis
    row["exceeds_threshold"] = ekf.norm_epsilon_norm >= 1 ? 1.0 : 0.0
    return row
end
