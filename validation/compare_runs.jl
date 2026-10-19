# Compare two results files (Julia vs Python reference) sample by sample.
#
#   julia --project validation/compare_runs.jl <julia.h5> <python.h5> [tolerance]
#
# For every output column and state: the RMS and maximum difference, relative to
# the signal's standard deviation over the run (so a 1e-3 means the two series
# differ by 0.1 % of how much the signal itself varies). The run passes when every
# compared channel stays under the tolerance in RMS (default 1e-3).

using HDF5, Statistics, Printf

const SKIP = Set(["exceeds_threshold"])   # a 0/1 flag at a threshold, flips on rounding

function read_run(path)
    h5open(path) do f
        g = f["ekf_output"]
        out = Dict(k => read(g[k]) for k in keys(g))
        return (outputs=out, x=read(f["state/x"]), names=read(f["state/names"]),
            kept=read(f["meta/kept"]), failed=read(f["meta/failed"]),
            iterations=read(f["meta/iekf_iterations"]), x0=read(f["meta/x0"]),
            pitot=read(f["meta/pitot_calibration"]),
            seconds=(read_attribute(f["meta"], "seconds_prerun"), read_attribute(f["meta"], "seconds_main")))
    end
end

function channel_stats(a, b)
    ok = isfinite.(a) .& isfinite.(b)
    any(ok) || return (rms=NaN, max=NaN, scale=NaN, n=0)
    d = a[ok] - b[ok]
    scale = std(b[ok])
    scale = scale > 0 ? scale : max(maximum(abs, b[ok]), 1e-12)
    return (rms=sqrt(mean(d .^ 2)) / scale, max=maximum(abs, d) / scale, scale=scale, n=count(ok))
end

function main(jl_path, py_path, tol)
    jl = read_run(jl_path)
    py = read_run(py_path)
    @printf("Samples: Julia %d kept / %d failed, Python %d kept / %d failed\n",
        length(jl.kept), length(jl.failed), length(py.kept), length(py.failed))
    jl.failed == py.failed || println("  failed samples differ: Julia $(jl.failed), Python $(py.failed)")
    @printf("Pitot calibration: Julia k=%.6f b=%.4f | Python k=%.6f b=%.4f\n", jl.pitot..., py.pitot...)
    @printf("x0 max abs difference: %.2e\n", maximum(abs, jl.x0 - py.x0))
    @printf("Runtime pre-run + main loop: Julia %.0f + %.0f s, Python %.0f + %.0f s\n", jl.seconds..., py.seconds...)

    common = intersect(jl.kept, py.kept)
    ij = indexin(common, jl.kept)
    ip = indexin(common, py.kept)
    same_iter = mean(jl.iterations[ij] .== py.iterations[ip])
    @printf("IEKF iterations identical on %.2f %% of steps (mean %.2f vs %.2f)\n",
        100same_iter, mean(jl.iterations[ij]), mean(py.iterations[ip]))

    worst = 0.0
    println("\nStates (RMS / max difference relative to the signal std):")
    jl.names == py.names || error("state layouts differ")
    for (i, name) in enumerate(py.names)
        s = channel_stats(jl.x[i, ij], py.x[i, ip])
        worst = max(worst, s.rms)
        @printf("  %-24s rms %.1e  max %.1e  (std %.3g)\n", name, s.rms, s.max, s.scale)
    end
    println("\nOutputs:")
    failures = String[]
    for col in sort(collect(keys(py.outputs)))
        col in SKIP && continue
        haskey(jl.outputs, col) || (push!(failures, "$col missing in Julia"); continue)
        a, b = jl.outputs[col][ij], py.outputs[col][ip]
        all(isnan, b) && all(isnan, a) && continue
        s = channel_stats(a, b)
        nan_mismatch = count(isnan.(a) .!= isnan.(b))
        worst = max(worst, s.rms)
        @printf("  %-36s rms %.1e  max %.1e  (std %.3g)%s\n", col, s.rms, s.max, s.scale,
            nan_mismatch > 0 ? "  NaN mismatch on $nan_mismatch" : "")
        s.rms > tol && push!(failures, col)
    end
    @printf("\nWorst RMS difference / std: %.2e (tolerance %.0e)\n", worst, tol)
    println(isempty(failures) ? "PASS" : "FAIL: $(join(failures, ", "))")
end

main(ARGS[1], ARGS[2], length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 1e-3)
