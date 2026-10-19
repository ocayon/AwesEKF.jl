# Unit tests on a synthetic kite (no flight data). The comparison against the
# Python implementation on real flights lives in validation/.

using AwesEKF, Test, LinearAlgebra, StaticArrays
const A = AwesEKF

const CFG = load_config(joinpath(@__DIR__, "data", "SYNTHETIC-KITE"))

"A plausible sample: kite at 200 m range, crosswind velocity."
function synthetic_sample(; ts=0.1)
    r = SVector(150.0, 20.0, 130.0)
    return A.EKFInput(; kite_position=r, kite_velocity=SVector(-5.0, 25.0, 3.0),
        kite_acceleration=SVector(1.0, -2.0, 0.5), timestep=ts, tether_force=4000.0,
        tether_length=200.0, tether_reelout_speed=1.5, kite_apparent_windspeed=30.0,
        steering_input=0.2, depower_input=0.3, delta_up=0.0)
end

@testset "AwesEKF" begin
    @testset "config and layout" begin
        m = SystemModel(CFG)
        @test m.layout.state_names[1:6] == ["r_0", "r_1", "r_2", "v_0", "v_1", "v_2"]
        @test length(A.model_stdv(CFG.tuning, CFG.sim)) == A.nstates(m.layout)
        @test CFG.kcu.distance_kcu_kite == 10.0          # wing CG height
        @test CFG.sim.epsilon == 1e-6                     # "1e-6" string in YAML
    end

    @testset "numpy gradient" begin
        x = [0.0, 0.1, 0.25, 0.3, 0.5]
        f = 3 .* x .^ 2 .+ 1
        g = A.numpy_gradient(f, x)
        @test g[2:end-1] ≈ 6 .* x[2:end-1]               # exact for quadratics
        @test A.numpy_gradient([1.0, 2.0, 4.0]) == [1.0, 1.5, 2.0]
    end

    @testset "initial state closes the tether" begin
        m = SystemModel(CFG)
        s = synthetic_sample()
        x0 = initial_state(A.TetherShapeSolver(), m, s)
        u = A.input_vector(m, s)
        shape = A.tether_shape_for(m, x0, u, x0)
        @test norm(shape.kite_position - s.kite_position) < 1e-8
        @test 180 < x0[m.layout.sx["tether_length"]] < 200   # range 199.5 m minus the 10 m bridle segment
    end

    @testset "Jacobians match finite differences" begin
        m = SystemModel(CFG)
        s = synthetic_sample()
        x = initial_state(A.TetherShapeSolver(), m, s)
        x[m.layout.sx["k_phi_us"]] = -0.1
        u = A.input_vector(m, s)
        F = A.state_jacobian(m, x, u)
        h, H = A.observation_and_jacobian(m, x, u)
        for i in eachindex(x)
            dx = zeros(length(x))
            dx[i] = 1e-6 * max(abs(x[i]), 1.0)
            fd_f = (A.dynamics(m, x + dx, u, x) - A.dynamics(m, x - dx, u, x)) / (2dx[i])
            fd_h = (A.observation(m, x + dx, u, x) - A.observation(m, x - dx, u, x)) / (2dx[i])
            @test isapprox(F[:, i], fd_f; rtol=1e-5, atol=1e-5)
            @test isapprox(H[:, i], fd_h; rtol=1e-5, atol=1e-5)
        end
    end

    @testset "tether Jacobian excludes the kite kinematics (paper Sect. 4.3, Fig. 6)" begin
        # The tether model is differentiated only with respect to tether length,
        # elevation and azimuth; r, v and the wind inside it come from x* (held fixed)
        m = SystemModel(CFG)
        s = synthetic_sample()
        x = initial_state(A.TetherShapeSolver(), m, s)
        u = A.input_vector(m, s)
        F_partial = A.state_jacobian(m, x, u)
        F_full = A.ForwardDiff.jacobian(xx -> A.dynamics(m, xx, u, xx), x)
        sx = m.layout.sx
        kinematic = [sx[n] for n in ("r_0", "r_1", "r_2", "v_0", "v_1", "v_2", "vw_0", "vw_1", "vw_2")]
        tether = [sx[n] for n in ("tether_length", "elevation_0", "azimuth_0")]
        @test F_partial[:, tether] ≈ F_full[:, tether] rtol = 1e-12
        @test norm(F_partial[4:6, kinematic] - F_full[4:6, kinematic]) > 1e-3
        h_partial = A.observation_and_jacobian(m, x, u)[2]
        h_full = A.ForwardDiff.jacobian(xx -> A.observation(m, xx, u, xx), x)
        @test h_partial[:, tether] ≈ h_full[:, tether] rtol = 1e-12
        @test norm(h_partial[:, kinematic] - h_full[:, kinematic]) > 1e-3
    end

    @testset "filter steps" begin
        m = SystemModel(CFG)
        s = synthetic_sample()
        x0 = initial_state(A.TetherShapeSolver(), m, s)
        ekf = A.ExtendedKalmanFilter(m, A.model_stdv(CFG.tuning, CFG.sim), A.meas_stdv(CFG.tuning, CFG.sim),
            x0, A.input_vector(m, s))
        for _ in 1:5
            A.step!(ekf, s)
        end
        @test all(isfinite, ekf.x_k1_k1)
        @test ekf.P_k1_k1 ≈ ekf.P_k1_k1'
        @test all(>(0), diag(ekf.P_k1_k1))
        row = A.ekf_output(ekf)
        @test all(c -> haskey(row, c), A.OUTPUT_COLUMNS)
        @test isfinite(row["wing_lift_coefficient"])
    end
end
