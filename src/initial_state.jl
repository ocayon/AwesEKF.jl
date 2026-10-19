# Initial state: solve the tether shape that ends at the measured kite position
# (awes_ekf find_initial_state_vector / Tether.solve_tether_shape).

"Tether-shape solver; keeps the last solution as the next initial guess."
mutable struct TetherShapeSolver
    guess::Union{Nothing,Vector{Float64}}
end
TetherShapeSolver() = TetherShapeSolver(nothing)

# The tether closure is solved with a port of scipy.optimize.least_squares as the
# Python code calls it (method "trf" without bounds, x_scale 1, 2-point
# finite-difference Jacobian, xtol = ftol = 1e-5). The closure has no exact
# solution when the tether sags strongly (low force); the solver then stops in a
# local minimum that depends on the algorithm, and the filter restarts from it.
# Using the same algorithm keeps those restarts identical to the Python code.

"Forward-difference Jacobian, scipy approx_derivative(method='2-point')."
function fd_jacobian(fun, x, f0)
    J = Matrix{Float64}(undef, length(f0), length(x))
    for i in eachindex(x)
        h = sqrt(eps(Float64)) * (x[i] >= 0 ? 1.0 : -1.0) * max(1.0, abs(x[i]))
        x1 = copy(x)
        x1[i] += h
        dx = x1[i] - x[i]          # exactly representable step
        J[:, i] = (fun(x1) - f0) / dx
    end
    return J
end

"Trust-region subproblem via one SVD (scipy _lsq.common.solve_lsq_trust_region)."
function solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha; rtol=0.01, max_iter=10)
    suf = s .* uf
    function phi_and_derivative(alpha)
        denom = s .^ 2 .+ alpha
        p_norm = norm(suf ./ denom)
        return p_norm - Delta, -sum(suf .^ 2 ./ denom .^ 3) / p_norm
    end
    full_rank = m >= n && s[end] > eps(Float64) * m * s[1]
    if full_rank
        p = -V * (uf ./ s)
        norm(p) <= Delta && return p, 0.0, 0
    end
    alpha_upper = norm(suf) / Delta
    if full_rank
        phi, phi_prime = phi_and_derivative(0.0)
        alpha_lower = -phi / phi_prime
    else
        alpha_lower = 0.0
    end
    alpha = (!full_rank && initial_alpha == 0) ? max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper)) :
            initial_alpha
    it = 0
    for outer it in 0:max_iter-1
        if alpha < alpha_lower || alpha > alpha_upper
            alpha = max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper))
        end
        phi, phi_prime = phi_and_derivative(alpha)
        phi < 0 && (alpha_upper = alpha)
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta
        abs(phi) < rtol * Delta && break
    end
    p = -V * (suf ./ (s .^ 2 .+ alpha))
    p *= Delta / norm(p)
    return p, alpha, it + 1
end

function update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit)
    ratio = predicted_reduction > 0 ? actual_reduction / predicted_reduction :
            (predicted_reduction == actual_reduction == 0 ? 1.0 : 0.0)
    if ratio < 0.25
        Delta = 0.25 * step_norm
    elseif ratio > 0.75 && bound_hit
        Delta *= 2.0
    end
    return Delta, ratio
end

function check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol)
    ftol_ok = dF < ftol * F && ratio > 0.25
    xtol_ok = dx_norm < xtol * (xtol + x_norm)
    return ftol_ok && xtol_ok ? 4 : ftol_ok ? 2 : xtol_ok ? 3 : nothing
end

"""
    solve_least_squares(fun, x0; ftol=1e-5, xtol=1e-5, gtol=1e-8)

scipy.optimize.least_squares(fun, x0, ftol=ftol, xtol=xtol) with its defaults
(trf_no_bounds, exact trust-region solver, max_nfev = 100 n).
"""
function solve_least_squares(fun, x0; ftol=1e-5, xtol=1e-5, gtol=1e-8)
    x = collect(Float64, x0)
    n = length(x)
    f = fun(x)
    all(isfinite, f) || error("Residuals are not finite in the initial point")
    m = length(f)
    J = fd_jacobian(fun, x, f)
    nfev = 1
    max_nfev = 100n
    cost = 0.5 * dot(f, f)
    g = J' * f
    Delta = norm(x)
    Delta == 0 && (Delta = 1.0)
    alpha = 0.0
    status = nothing
    while true
        maximum(abs, g) < gtol && (status = 1)
        (status !== nothing || nfev == max_nfev) && break
        F = svd(J)
        uf = F.U' * f
        actual_reduction = -1.0
        x_new, f_new, cost_new = x, f, cost
        while actual_reduction <= 0 && nfev < max_nfev
            step, alpha, _ = solve_lsq_trust_region(n, m, uf, F.S, F.V, Delta, alpha)
            Js = J * step
            predicted_reduction = -(0.5 * dot(Js, Js) + dot(step, g))
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1
            step_norm = norm(step)
            if !all(isfinite, f_new)
                Delta = 0.25 * step_norm
                continue
            end
            cost_new = 0.5 * dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(Delta, actual_reduction, predicted_reduction,
                step_norm, step_norm > 0.95 * Delta)
            status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            status !== nothing && break
            alpha *= Delta / Delta_new
            Delta = Delta_new
        end
        if actual_reduction > 0
            x, f, cost = x_new, f_new, cost_new
            J = fd_jacobian(fun, x, f)
            g = J' * f
        end
    end
    return x
end

"""
    solve_tether_shape!(solver, m, sample, wind)

Elevation, azimuth and length of the tether whose end matches the measured kite
position, for the measured ground tension and the given wind at the kite.
"""
function solve_tether_shape!(solver::TetherShapeSolver, m::SystemModel, s::EKFInput, wind)
    r = s.kite_position
    if solver.guess === nothing
        guess = [atan(r[3], vnorm(r)), atan(r[2], r[1]), vnorm(r)]
        solver.guess = all(isfinite, guess) ? guess : [0.0, 0.0, 0.0]
    end
    residual(p) = r - tether_shape(m.tether, p[1], p[2], p[3], s.tether_force, r, s.kite_velocity,
        wind, s.kite_acceleration, s.kcu_acceleration, s.kcu_velocity).kite_position
    try
        p = solve_least_squares(residual, solver.guess)
        solver.guess = p
        return p[1], p[2], p[3]
    catch
        return 1.0, 1.0, 1.0
    end
end

"""
    initial_state(solver, m, sample; wind_velocity, CL=1, CD=0.15, CS=0)

Initial state vector: measured kite position and velocity, the guessed wind,
the given aero coefficients, the solved tether shape and zero offsets/constants.
"""
function initial_state(solver::TetherShapeSolver, m::SystemModel, s::EKFInput;
    wind_velocity=[1e-3, 8.0, 0.0], CL=1.0, CD=0.15, CS=0.0)
    sim = m.sim
    wind = Vec3(wind_velocity...)
    solve_wind = wind
    if sim.log_profile
        # The guess is the wind at 10 m: solve with it scaled to kite height
        wvel = vnorm(wind) * log(s.kite_position[3] / Z0) / log(10 / Z0)
        wdir = atan(wind[2], wind[1])
        solve_wind = Vec3(wvel * cos(wdir), wvel * sin(wdir), 0.0)
    end
    elevation, azimuth, len = solve_tether_shape!(solver, m, s, solve_wind)

    x0 = zeros(nstates(m.layout))
    sx = m.layout.sx
    x0[1:3] .= s.kite_position
    x0[4:6] .= s.kite_velocity
    if sim.log_profile
        x0[7:9] .= (vnorm(wind) * KAPPA / log(10 / Z0), atan(wind[2], wind[1]), 0.0)
    else
        x0[7:9] .= wind
    end
    x0[sx["CL"]] = CL
    x0[sx["CD"]] = CD
    x0[sx["CS"]] = CS
    x0[sx["tether_length"]] = len
    x0[sx["elevation_0"]] = elevation
    x0[sx["azimuth_0"]] = azimuth
    return x0
end
