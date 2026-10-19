# (Iterated) extended Kalman filter (awes_ekf.ekf.kalman_filter).

mutable struct ExtendedKalmanFilter
    model::SystemModel
    Q::Matrix{Float64}
    R::Matrix{Float64}
    x_k1_k::Vector{Float64}     # predicted state
    x_k1_k1::Vector{Float64}    # corrected state
    P_k1_k::Matrix{Float64}
    P_k1_k1::Matrix{Float64}
    u::Vector{Float64}
    z::Vector{Float64}
    iekf_iterations::Int
    nis::Float64
    mahalanobis_distance::Float64
    norm_epsilon_norm::Float64
end

function ExtendedKalmanFilter(model::SystemModel, stdv_model, stdv_meas, x0, u0)
    n = length(stdv_model)
    n == nstates(model.layout) || error("$(n) process noise values for $(nstates(model.layout)) states")
    return ExtendedKalmanFilter(model, diagm(stdv_model .^ 2), diagm(stdv_meas .^ 2),
        copy(x0), copy(x0), Matrix(1.0I, n, n), Matrix(1.0I, n, n), u0, Float64[], 0, NaN, NaN, NaN)
end

# Jacobians (Cayon et al. 2025, WES 10, Sect. 4.3 and Fig. 6): the tether model
# inside f and h is differentiated only with respect to the tether length and the
# first-segment elevation/azimuth. The kite position, velocity and wind it uses
# come from the nominal state x*, passed as the fixed `x0` argument, so their
# dependence through the tether model is deliberately left out of F and H.

"Jacobian of f with respect to the state, x0 held at `x`."
state_jacobian(m::SystemModel, x, u) = ForwardDiff.jacobian(xx -> dynamics(m, xx, u, x), x)

"Observation h and its Jacobian with respect to the state, x0 held at `x`."
function observation_and_jacobian(m::SystemModel, x, u)
    h = observation(m, x, u, x)
    H = ForwardDiff.jacobian(xx -> observation(m, xx, u, x), x)
    return h, H
end

"Covariance prediction with the zero-order-hold discretisation of the Jacobian."
function predict!(ekf::ExtendedKalmanFilter, ts)
    F = state_jacobian(ekf.model, ekf.x_k1_k, ekf.u)
    Phi = exp(F * ts)
    ekf.P_k1_k = Phi * ekf.P_k1_k1 * Phi' + ekf.Q
    return ekf
end

"Measurement update; iterated (IEKF) when the configuration asks for it."
function update!(ekf::ExtendedKalmanFilter)
    m = ekf.model
    sim = m.sim
    valid = findall(!isnan, ekf.z)
    length(valid) < length(ekf.z) &&
        println("Warning: NaN values found in the measurements at indices $(findall(isnan, ekf.z) .- 1)")
    z = ekf.z[valid]
    R = ekf.R[valid, valid]
    P = ekf.P_k1_k
    x_pred = ekf.x_k1_k

    local K, H, z_pred, P_zz
    if sim.do_iekf
        eta2 = x_pred
        err = 2 * sim.epsilon
        iterations = 0
        while err > sim.epsilon
            if iterations >= sim.max_iterations
                println("Terminating IEKF: exceeded max iterations ($(sim.max_iterations))")
                break
            end
            iterations += 1
            eta1 = eta2
            h_full, H_full = observation_and_jacobian(m, eta1, ekf.u)
            H = H_full[valid, :]
            z_pred = h_full[valid]
            P_zz = H * P * H' + R
            K = P * H' * inv(P_zz)
            eta2 = x_pred + K * (z - z_pred - H * (x_pred - eta1))
            err = norm(eta2 - eta1) / max(norm(eta1), 1e-10)
        end
        ekf.iekf_iterations = iterations
        ekf.x_k1_k1 = eta2
    else
        h_full, H_full = observation_and_jacobian(m, x_pred, ekf.u)
        H = H_full[valid, :]
        z_pred = h_full[valid]
        P_zz = H * P * H' + R
        K = P * H' * inv(P_zz)
        ekf.x_k1_k1 = x_pred + K * (z - z_pred)
    end

    # Joseph form
    IKH = I - K * H
    ekf.P_k1_k1 = IKH * P * IKH' + K * R * K'

    # Innovation statistics of the last iteration
    innovation = z - z_pred
    ekf.nis = innovation' * inv(P_zz) * innovation
    ekf.mahalanobis_distance = sqrt(ekf.nis)
    ekf.norm_epsilon_norm = norm(innovation ./ sqrt.(diag(P_zz)))
    return ekf
end

"""
    step!(ekf, sample)

One filter step, in the order of awes_ekf's propagate_state_EKF: propagate the
corrected state with the PREVIOUS input, then load this sample's input and
measurements, predict the covariance and update.
"""
function step!(ekf::ExtendedKalmanFilter, s)
    m = ekf.model
    ekf.x_k1_k = propagate(m, ekf.x_k1_k1, ekf.u, s.timestep)
    ekf.u = input_vector(m, s)
    ekf.z = measurement_vector(m, s)
    predict!(ekf, s.timestep)
    update!(ekf)
    return ekf
end
