# State propagation over one sample: CVODES (BDF, dense direct solver with the
# exact ForwardDiff Jacobian) with the tolerances of CasADi's cvodes plugin.

"""
    propagate(m, x, u, ts)

Integrate f(x, u, x_start) from 0 to `ts` (rounded to ms, as the Python code
does), holding the input and the start state fixed. Throws on an integrator
failure, which the filter loop answers with a re-initialisation.
"""
function propagate(m::SystemModel, x::AbstractVector, u::AbstractVector, ts::Real)
    ts = round(ts; digits=3)
    x_start = collect(Float64, x)
    rhs(xx) = dynamics(m, xx, u, x_start)
    f!(dx, xx, p, t) = (dx .= rhs(xx); nothing)
    jac!(J, xx, p, t) = (ForwardDiff.jacobian!(J, rhs, xx); nothing)
    prob = ODEProblem(ODEFunction(f!; jac=jac!), x_start, (0.0, ts))
    sol = solve(prob, CVODE_BDF(linear_solver=:Dense, max_order=5);
        abstol=CVODE_ABSTOL, reltol=CVODE_RELTOL, maxiters=CVODE_MAX_STEPS,
        save_everystep=false, save_start=false, verbose=false)
    SciMLBase.successful_retcode(sol) || error("Integration failed ($(sol.retcode))")
    x_end = sol.u[end]
    all(isfinite, x_end) || error("Integration produced a non-finite state")
    return x_end
end
