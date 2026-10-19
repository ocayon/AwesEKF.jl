# Run the filter on the bundled LEI V3 example: 10 minutes of the 8 October 2019
# flight (minutes 60-70), with the as-flown system and the tuned filter
# configuration. Prints the estimated wind and aerodynamic coefficients per
# minute and writes the full results to results/v3_example.h5.
#
#   julia --project examples/run_v3_example.jl

using AwesEKF, Printf, Statistics

const HERE = @__DIR__

result, flight_data, cfg = run_flight(
    joinpath(HERE, "data", "v3_2019-10-08_min60-70.csv"),
    joinpath(HERE, "LEI-V3-KITE");
    out=joinpath(HERE, "..", "results", "v3_example.h5"),
)

o = result.outputs
t = flight_data["time"][result.kept.+1]
minute = floor.(Int, (t .- t[1]) ./ 60)
# Meteorological convention: direction the wind comes from, clockwise from north
wind_from = mod.(270 .- rad2deg.(o["wind_direction"]), 360)

println()
@printf("%d steps, %d re-initialisations, pre-run %.0f s + filter %.0f s\n",
    length(result.kept), length(result.failed), result.seconds_prerun, result.seconds_main)
pitot = get(result.calibrations, :pitot, nothing)
if pitot !== nothing
    @printf("Pitot calibration (pre-run fit): k = %.3f, zero = %.1f m²/s²\n", pitot.k, pitot.b)
elseif cfg.sim.pitot_calibration_k !== nothing
    @printf("Pitot calibration (frozen, from config): k = %.4f, zero = %.2f m²/s²\n",
        cfg.sim.pitot_calibration_k, cfg.sim.pitot_calibration_b)
end
println()
println("minute  height  wind speed  wind from  w_z      CL     CD     CS")
println("          [m]     [m/s]      [deg]    [m/s]")
for m in sort(unique(minute))
    i = minute .== m
    @printf("%4d   %6.0f   %7.2f   %8.1f  %6.2f  %5.2f  %5.3f  %6.3f\n", m,
        mean(o["kite_position_z"][i]), mean(o["wind_speed_horizontal"][i]), mean(wind_from[i]),
        mean(o["wind_speed_vertical"][i]), mean(o["wing_lift_coefficient"][i]),
        mean(o["wing_drag_coefficient"][i]), mean(o["wing_sideforce_coefficient"][i]))
end
println()
println("Identified steering constants (end of run): ",
    join((@sprintf("%s = %.3f", k, o[k][end]) for k in ("k_phi_us", "k_cl_us", "k_cd_us", "k_cl_us_odd")), ", "))
