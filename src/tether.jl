# Quasi-static lumped-mass tether model (awes_ekf.setup.tether.Tether).
#
# Starting from the measured ground tension and the elevation/azimuth of the
# first element, the force balance of each point mass gives the tension in the
# next element, and the tension direction the position of the next node. With a
# KCU, the last two nodes are the KCU and the kite (bridle point), joined by the
# bridle segment of length `distance_kcu_kite`.
#
# All functions are generic in the number type, so ForwardDiff differentiates
# straight through them.

struct TetherModel
    p::TetherParams
    kcu::Union{Nothing,KCUParams}
    kite_mass::Float64
    kite_area::Float64
    kite_acceleration::Bool   # use the measured kite acceleration (KCU kinematics)
    kcu_acceleration::Bool
    kcu_velocity::Bool
    elastic::Bool
    cf::Float64               # skin-friction drag coefficient
    EA::Float64
end

function TetherModel(p::TetherParams, kcu, kite::KiteParams, obs::ObservationData; elastic=true)
    area = π * (p.diameter / 2)^2
    return TetherModel(p, kcu, kite.mass, kite.area, obs.kite_acceleration, obs.kcu_acceleration,
        obs.kcu_velocity, elastic, 0.01, p.youngs_modulus * area)
end

"""
    tether_shape(tm, elevation_0, azimuth_0, tether_length, tension_ground,
                 r_kite, v_kite, vw, a_kite, a_kcu, v_kcu)

Shape and loads of the tether for the given ground boundary condition. `r_kite`,
`v_kite` set the tether rotation (and, with a measured kite acceleration, the
KCU kinematics); `vw` is the wind at the kite, scaled down the tether with a log
profile. Returns a named tuple with the modelled kite position, the tensions at
the kite and below the KCU, the aerodynamic force on the kite, and the parasitic
drag terms.
"""
function tether_shape(tm::TetherModel, el, az, len, tension_ground, r_kite, v_kite, vw,
    a_kite, a_kcu_in, v_kcu_in)
    T = promote_type(typeof(el), typeof(az), typeof(len), typeof(tension_ground),
        eltype(r_kite), eltype(v_kite), eltype(vw), eltype(a_kite), eltype(a_kcu_in), eltype(v_kcu_in))
    V = Vec3{T}
    p = tm.p
    kcu = tm.kcu
    has_kcu = kcu !== nothing

    l_unstrained = len / p.n_elements
    m_s = π * p.diameter^2 / 4 * l_unstrained * p.density
    n = p.n_elements + (has_kcu ? 1 : 0)

    wvel = vnorm(vw)
    wdir = vw / wvel

    rn = vnorm(r_kite)
    vtau_kite = project_onto_plane(v_kite, r_kite / rn)
    omega_tether = cross(r_kite, vtau_kite) / rn^2

    if tm.kite_acceleration
        # Instantaneous centre of rotation and angular velocity of the kite
        vk = vnorm(v_kite)
        at = dot(a_kite, v_kite / vk) * v_kite / vk
        omega_kite = -cross(a_kite - at, v_kite) / vk^2
        icr = -cross(v_kite, omega_kite) / vnorm(omega_kite)^2
        alpha = cross(at, icr) / vnorm(icr)^2
    else
        omega_kite = zero(V)
        alpha = zero(V)
    end

    # Python index i (0-based) is stored at i + 1
    tensions = Vector{V}(undef, n)
    positions = Vector{V}(undef, n + 1)
    dir0 = V(cos(el) * cos(az), cos(el) * sin(az), sin(el))
    tensions[1] = V(dir0[1] * tension_ground, dir0[2] * tension_ground, dir0[3] * tension_ground)
    l_s = tm.elastic ? (tension_ground / tm.EA + 1) * l_unstrained : l_unstrained
    positions[1] = zero(V)
    positions[2] = V(dir0[1] * l_s, dir0[2] * l_s, dir0[3] * l_s)

    vj = zero(V)
    vwj = zero(V)
    vaj = zero(V)
    aero_force = zero(V)
    D_kcu = zero(T)
    drag_bridles = zero(V)
    lift_bridles = zero(V)
    drag_tether = zero(T)
    stretched_length = l_s

    log_kite = nlog(r_kite[3] / Z0)
    for j in 0:n-1
        last_element = j == n - 1
        kcu_element = has_kcu && j == n - 2
        pj = positions[j+2]                      # node j + 1 (the point mass)

        vj = cross(omega_tether, pj)
        aj = cross(omega_tether, vj)
        delta_p = pj - positions[j+1]
        ej = delta_p / vnorm(delta_p)            # axial direction of the element
        vwj = wvel * nlog(pj[3] / Z0) / log_kite * wdir

        if kcu_element
            if tm.kcu_acceleration
                aj = a_kcu_in
            elseif tm.kite_acceleration
                aj = a_kite + cross(alpha, pj - r_kite) + cross(omega_kite, cross(omega_kite, pj - r_kite))
            end
            if tm.kcu_velocity
                vj = v_kcu_in
            elseif tm.kite_acceleration
                vj = v_kite + cross(omega_kite, pj - r_kite)
            end
            ej = (r_kite - pj) / vnorm(r_kite - pj)
        end

        # Flow at point mass j
        vaj = vwj - vj
        vajp = dot(vaj, ej) * ej                 # along the element
        vajn = vaj - vajp                        # across the element
        vaj_sq = vnorm(vaj)^2

        theta = angle_between(vaj, ej)
        cd_t = p.cd * sin(theta)^3 + π * tm.cf * cos(theta)^3
        cl_t = p.cd * sin(theta)^2 * cos(theta) - π * tm.cf * sin(theta) * cos(theta)^2
        dir_D = vaj / vnorm(vaj)
        lift_perp = ej - dot(ej, dir_D) * dir_D
        dir_L = -lift_perp / (vnorm(lift_perp) + 1e-9)
        q_area = 0.5 * RHO * vnorm(vaj)^2 * l_unstrained * p.diameter
        lift_j = q_area * cl_t * dir_L
        drag_j = q_area * cd_t * dir_D

        if !has_kcu
            # Boundary (kite) node carries half an element
            faj = (p.n_elements == 1 || last_element) ? 0.5 * drag_j + 0.5 * lift_j : drag_j + lift_j
        elseif last_element
            faj = 0.5 * drag_bridles + 0.5 * lift_bridles
        elseif kcu_element
            bridle_q = 0.5 * RHO * kcu.total_length_bridle_lines * kcu.diameter_bridle_lines * vaj_sq
            drag_bridles = bridle_q * cd_t * dir_D
            lift_bridles = bridle_q * cl_t * dir_L
            # KCU (+ onboard turbine): each drag area on its own flow component
            dp_kcu = 0.5 * RHO * vnorm(vajn) * vajn * kcu.cd_area_crossflow
            dt_kcu = 0.5 * RHO * vnorm(vajp) * vajp * kcu.cd_area_axial
            D_kcu = vnorm(dp_kcu + dt_kcu)
            faj = dp_kcu + dt_kcu + 0.5 * drag_bridles + 0.5 * lift_bridles + 0.5 * drag_j + 0.5 * lift_j
        else
            faj = lift_j + drag_j
            drag_tether += vnorm(drag_j)
        end

        point_mass = if !has_kcu
            last_element ? m_s / 2 + tm.kite_mass : m_s
        else
            last_element ? tm.kite_mass : kcu_element ? m_s / 2 + kcu.mass : m_s
        end
        fgj = V(0, 0, -point_mass * G)

        # Force balance gives the tension in the next element ...
        if !last_element
            tensions[j+2] = point_mass * aj + tensions[j+1] - fgj - faj
        end
        # ... and its direction the position of the next node
        if kcu_element
            t = tensions[j+2]
            positions[j+3] = pj + t / vnorm(t) * kcu.distance_kcu_kite
        elseif !last_element
            t = tensions[j+2]
            l_s = tm.elastic ? (vnorm(t) / tm.EA + 1) * l_unstrained : l_unstrained
            stretched_length += l_s
            positions[j+3] = pj + t / vnorm(t) * l_s
        else
            next_tension = tensions[j+1] - fgj - faj
            if tm.kite_acceleration
                next_tension += point_mass * aj
            end
            aero_force = next_tension
        end
    end

    return (kite_position=positions[n+1], tension_kite=tensions[n], tension_below_kite=tensions[n-1],
        va=vwj - vj, aero_force=aero_force, D_kcu=D_kcu, drag_bridles=drag_bridles,
        drag_tether=drag_tether, stretched_length=stretched_length, positions=positions)
end

"Right-handed frame from a z axis and a vector in the x-z plane."
function frame_from_z(ez, toward)
    ey = cross(ez, toward) / vnorm(cross(ez, toward))
    ex = cross(ey, ez)
    return hcat(ex, ey, ez)
end

"""
Frames and aerodynamic coefficients derived from a tether shape (the
`bridle_frame_va`, `bridle_frame_vk`, `tether_frame`, `radial_frame`, `CL`, `CD`,
`CS`, `cd_*` functions of the Python tether model).
"""
function tether_derived(tm::TetherModel, shape, r_kite, v_kite)
    va = shape.va
    ez_bridle = -shape.tension_kite / vnorm(shape.tension_kite)
    dcm_b2w = frame_from_z(ez_bridle, -va)
    dcm_b2vel = frame_from_z(ez_bridle, v_kite)
    ez_tether = -shape.tension_below_kite / vnorm(shape.tension_below_kite)
    dcm_t2w = frame_from_z(ez_tether, -va)
    dcm_r2w = frame_from_z(-r_kite / vnorm(r_kite), -va)

    q = 0.5 * RHO * tm.kite_area * vnorm(va)^2
    dir_D = va / vnorm(va)
    tension_unit = shape.tension_kite / vnorm(shape.tension_kite)
    dir_L = tension_unit - dot(tension_unit, dir_D) * dir_D
    dir_L = dir_L / (vnorm(dir_L) + 1e-9)
    dir_S = cross(dir_L, dir_D)
    F = shape.aero_force
    # Parasitic drag, normalised with the kite area and the flow at the kite node
    q_node = 0.5 * RHO * vnorm(va)^2 * tm.kite_area
    has_kcu = tm.kcu !== nothing
    return (dcm_b2w=dcm_b2w, dcm_b2vel=dcm_b2vel, dcm_t2w=dcm_t2w, dcm_r2w=dcm_r2w,
        CL=dot(F, dir_L) / q, CD=dot(F, dir_D) / q, CS=dot(F, dir_S) / q,
        cd_kcu=has_kcu ? shape.D_kcu / q_node : 0.0,
        cd_bridles=has_kcu ? vnorm(shape.drag_bridles) / q_node : 0.0,
        cd_tether=shape.drag_tether / q_node)
end
