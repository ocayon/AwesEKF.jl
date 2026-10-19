# Configuration: ekf_config.yaml + an awesIO system*.yaml, with the same
# defaults and extraction as awes_ekf.setup.settings.

"Which signals are measured (and therefore used) by the filter."
Base.@kwdef mutable struct ObservationData
    tether_length::Bool = true
    tether_elevation::Bool = true
    tether_azimuth::Bool = true
    kite_position::Bool = true
    kite_velocity::Bool = true
    tether_force::Bool = true
    kite_acceleration::Bool = false
    kcu_position::Bool = false
    kcu_acceleration::Bool = false
    kcu_velocity::Bool = false
    kite_apparent_windspeed::Bool = false
    bridle_angle_of_attack::Bool = false
    bridle_angle_of_sideslip::Bool = false
    kite_yaw_angle::Bool = false
    kite_thrust_force::Bool = false
    dynamic_depower::Bool = false
end

"Model options and filter settings (`simulation_parameters` block)."
Base.@kwdef mutable struct SimulationConfig
    do_iekf::Bool = true
    epsilon::Float64 = 1e-6
    max_iterations::Int = 200
    log_profile::Bool = false
    initial_wind_velocity::Vector{Float64} = [1e-3, 8.0, 0.0]
    enforce_vertical_wind_to_0::Bool = false
    model_yaw::Bool = false
    steering_dependent_cs::Bool = false
    steering_dependent_clcd::Bool = false
    steering_dependent_cl_asym::Bool = false
    steering_input_lag::Float64 = 0.0
    drag_polar::Bool = false
    # Pre-run sensor calibration
    calibrate_apparent_windspeed::Bool = true
    calibrate_angle_of_attack::Bool = true
    pitot_calibration_fit_zero::Bool = false
    pitot_calibration_k::Union{Nothing,Float64} = nothing
    pitot_calibration_b::Float64 = 0.0
    calibration_start_minutes::Float64 = 5.0
    calibration_duration_minutes::Float64 = 15.0
    calibration_end_margin_minutes::Float64 = 3.0
    obs::ObservationData = ObservationData()
end

"True when the steering input enters the model."
uses_steering(c::SimulationConfig) =
    c.model_yaw || c.steering_dependent_cs || c.steering_dependent_clcd ||
    c.steering_dependent_cl_asym

struct KiteParams
    model_name::String
    mass::Float64
    area::Float64
    span::Float64
end

"Kite control unit; drag coefficients from the Applied Fluid Dynamics Handbook."
struct KCUParams
    length::Float64
    diameter::Float64
    mass::Float64
    distance_kcu_kite::Float64
    total_length_bridle_lines::Float64
    diameter_bridle_lines::Float64
    cd_area_axial::Float64        # drag area for flow along the KCU axis [m^2]
    cd_area_crossflow::Float64    # drag area for flow across it (+ turbine) [m^2]
end

const KCU_LD_T = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
const KCU_CD_T = [1.15, 1.1, 0.93, 0.85, 0.83, 0.85, 0.85, 0.85]
const KCU_LD_P = [1, 1.98, 2.96, 5, 10, 20, 40, 1e6]
const KCU_CD_P = [0.64, 0.68, 0.74, 0.74, 0.82, 0.91, 0.98, 1.2]

function KCUParams(; length, diameter, mass, distance_kcu_kite, total_length_bridle_lines,
    diameter_bridle_lines, turbine_diameter=0.0, turbine_thrust_coefficient=0.0)
    # Interpolating cubic splines (FITPACK, as scipy's splrep(s=0)/splev)
    ld = length / diameter
    cdt = Spline1D(KCU_LD_T, KCU_CD_T; k=3, s=0.0, bc="extrapolate")(ld)
    cdp = Spline1D(Float64.(KCU_LD_P), KCU_CD_P; k=3, s=0.0, bc="extrapolate")(ld)
    area_frontal = π * (diameter / 2)^2
    area_broadside = diameter * length
    area_turbine = π * (turbine_diameter / 2)^2
    return KCUParams(length, diameter, mass, distance_kcu_kite, total_length_bridle_lines,
        diameter_bridle_lines, cdt * area_frontal,
        cdp * area_broadside + turbine_thrust_coefficient * area_turbine)
end

const TETHER_MATERIALS = Dict(
    "Dyneema-SK78" => (density=970.0, cd=1.1, youngs_modulus=132e9),
    "Dyneema-SK75" => (density=970.0, cd=1.1, youngs_modulus=109e9),
    "Kitekraft" => (density=1500.0, cd=1.1, youngs_modulus=70e9),
)

struct TetherParams
    material_name::String
    diameter::Float64
    n_elements::Int
    density::Float64
    cd::Float64
    youngs_modulus::Float64
end

function TetherParams(material_name, diameter, n_elements)
    haskey(TETHER_MATERIALS, material_name) || error("Invalid tether material $material_name")
    diameter > 0 && n_elements > 0 || error("diameter and n_elements must be positive")
    m = TETHER_MATERIALS[material_name]
    return TetherParams(material_name, diameter, n_elements, m.density, m.cd, m.youngs_modulus)
end

"Process (model) and measurement noise standard deviations."
struct TuningParameters
    model_stdv::Dict{String,Any}
    meas_stdv::Dict{String,Any}
end

"Everything a run needs: the raw config dicts and the parsed parameters."
struct Config
    sim::SimulationConfig
    tuning::TuningParameters
    kite::KiteParams
    kcu::Union{Nothing,KCUParams}
    tether::TetherParams
    raw::Dict{String,Any}   # merged config, as written to the results file
end

# YAML 1.1 reads "1e-6" (no dot) as a string; the Python code float()s these
tofloat(x::Real) = Float64(x)
tofloat(x::AbstractString) = parse(Float64, x)

getor(d, key, default) = (v = get(d, key, nothing); v === nothing ? default : v)

function SimulationConfig(p::Dict)
    measurements = getor(p, "measurements", Dict{String,Any}())
    obs = ObservationData()
    for (key, value) in measurements
        sym = Symbol(key)
        hasfield(ObservationData, sym) || error("Unknown measurement '$key'")
        setfield!(obs, sym, Bool(value))
    end
    k = getor(p, "pitot_calibration_k", nothing)
    c = SimulationConfig(
        do_iekf=getor(p, "doIEKF", true),
        epsilon=tofloat(getor(p, "epsilon", 1e-6)),
        max_iterations=getor(p, "max_iterations", 200),
        log_profile=getor(p, "log_profile", false),
        initial_wind_velocity=tofloat.(getor(p, "initial_wind_velocity", [1e-3, 8, 0])),
        enforce_vertical_wind_to_0=getor(p, "enforce_vertical_wind_to_0", false),
        model_yaw=getor(p, "model_yaw", false),
        steering_dependent_cs=getor(p, "steering_dependent_cs", false),
        steering_dependent_clcd=getor(p, "steering_dependent_clcd", false),
        steering_dependent_cl_asym=getor(p, "steering_dependent_cl_asym", false),
        steering_input_lag=tofloat(getor(p, "steering_input_lag", 0.0)),
        drag_polar=getor(p, "drag_polar", false),
        calibrate_apparent_windspeed=getor(p, "calibrate_apparent_windspeed", true),
        calibrate_angle_of_attack=getor(p, "find_offset_angle_of_attack", true),
        pitot_calibration_fit_zero=getor(p, "pitot_calibration_fit_zero", false),
        pitot_calibration_k=(k === nothing || k == 0) ? nothing : tofloat(k),
        pitot_calibration_b=tofloat(getor(p, "pitot_calibration_b", 0.0)),
        calibration_start_minutes=tofloat(getor(p, "calibration_start_minutes", 5)),
        calibration_duration_minutes=tofloat(getor(p, "calibration_duration_minutes", 15)),
        calibration_end_margin_minutes=tofloat(getor(p, "calibration_end_margin_minutes", 3)),
        obs=obs,
    )
    # A frozen pitot calibration replaces the pre-run fit
    c.pitot_calibration_k !== nothing && (c.calibrate_apparent_windspeed = false)
    c.model_yaw && error("model_yaw is not supported by AwesEKF")
    return c
end

function primary_component(components, plural, singular)
    list = get(components, plural, nothing)
    list isa AbstractVector && !isempty(list) && return list[1]
    return get(components, singular, plural == "kites" ? components : Dict())
end

nested(d, keys...) = foldl((acc, k) -> acc isa AbstractDict ? getor(acc, k, Dict()) : Dict(), keys; init=d)

"KCU-to-wing distance: the wing CG height above the bridle point (awesIO body origin)."
function distance_kcu_kite(wing_struct, control_struct)
    cg = get(wing_struct, "center_of_mass", nothing)
    cg !== nothing && cg[3] != 0 && return tofloat(cg[3])
    return tofloat(getor(control_struct, "distance_kcu_kite", 0.0))
end

"""
    load_config(folder; system_yaml=nothing)

Load `ekf_config.yaml` and one awesIO `system*.yaml` from a kite folder
(EKF-AWE/AWETrim convention `data/<KITE-NAME>/`). With several system variants
the file must be named with `system_yaml`.
"""
function load_config(folder::AbstractString; system_yaml=nothing)
    ekf_path = joinpath(folder, "ekf_config.yaml")
    isfile(ekf_path) || error("ekf_config.yaml not found in $folder")
    candidates = sort(filter(f -> startswith(f, "system") && (endswith(f, ".yaml") || endswith(f, ".yml")),
        readdir(folder)))
    isempty(candidates) && error("No system*.yaml in $folder")
    if system_yaml === nothing
        length(candidates) == 1 ||
            error("Several system files in $folder ($(join(candidates, ", "))); pass system_yaml")
        system_yaml = candidates[1]
    end
    system_yaml in candidates || error("$system_yaml not found in $folder")

    raw = YAML.load_file(ekf_path; dicttype=Dict{String,Any})
    system = YAML.load_file(joinpath(folder, system_yaml); dicttype=Dict{String,Any})

    components = getor(system, "components", Dict{String,Any}())
    kite_node = primary_component(components, "kites", "kite")
    wing = nested(kite_node, "wing", "structure")
    bridle = nested(kite_node, "bridle", "structure")
    control_struct = nested(kite_node, "control_system", "structure")
    turbine = getor(getor(kite_node, "control_system", Dict()), "turbine", Dict())
    tether_struct = nested(primary_component(components, "tethers", "tether"), "structure")

    raw["kite"] = Dict{String,Any}(
        "model_name" => getor(kite_node, "name", "unknown"),
        "mass" => getor(wing, "mass", 0.0),
        "area" => getor(wing, "projected_surface_area", getor(wing, "planform_surface_area", 0.0)),
        "span" => getor(wing, "span", 0.0),
        "sensor_ids" => [0, 1],
    )
    raw["kcu"] = Dict{String,Any}(
        "length" => getor(control_struct, "length", 1.0),
        "diameter" => getor(control_struct, "diameter", 0.48),
        "mass" => getor(control_struct, "mass", 0.0),
        "distance_kcu_kite" => distance_kcu_kite(wing, control_struct),
        "total_length_bridle_lines" => getor(bridle, "total_nominal_line_length", 0.0),
        "diameter_bridle_lines" => getor(bridle, "avg_line_diameter", 0.0),
        "turbine_diameter" => getor(turbine, "diameter", 0.0),
        "turbine_thrust_coefficient" => getor(turbine, "thrust_coefficient", 0.0),
    )
    raw["tether"] = Dict{String,Any}(
        "material_name" => getor(nested(tether_struct, "material"), "type", "Dyneema-SK78"),
        "diameter" => getor(tether_struct, "diameter", 0.01),
        "n_elements" => 30,
    )
    raw["system_yaml_used"] = system_yaml

    sim = SimulationConfig(raw["simulation_parameters"])
    tp = raw["tuning_parameters"]
    tuning = TuningParameters(tp["model_stdv"], tp["meas_stdv"])
    k = raw["kite"]
    kite = KiteParams(string(k["model_name"]), tofloat(k["mass"]), tofloat(k["area"]), tofloat(k["span"]))
    c = raw["kcu"]
    kcu = KCUParams(; length=tofloat(c["length"]), diameter=tofloat(c["diameter"]),
        mass=tofloat(c["mass"]), distance_kcu_kite=tofloat(c["distance_kcu_kite"]),
        total_length_bridle_lines=tofloat(c["total_length_bridle_lines"]),
        diameter_bridle_lines=tofloat(c["diameter_bridle_lines"]),
        turbine_diameter=tofloat(c["turbine_diameter"]),
        turbine_thrust_coefficient=tofloat(c["turbine_thrust_coefficient"]))
    t = raw["tether"]
    tether = TetherParams(string(t["material_name"]), tofloat(t["diameter"]), Int(t["n_elements"]))
    return Config(sim, tuning, kite, kcu, tether, raw)
end

"""
Process noise standard deviations, in state order (TuningParameters in awes_ekf).
"""
function model_stdv(tuning::TuningParameters, sim::SimulationConfig)
    d = tuning.model_stdv
    keys = sim.log_profile ?
           ["x", "x", "x", "v", "v", "v", "uf", "wdir", "vwz", "CL", "CD", "CS",
        "tether_length", "tether_elevation", "tether_azimuth"] :
           ["x", "x", "x", "v", "v", "v", "vw", "vw", "vwz", "CL", "CD", "CS",
        "tether_length", "tether_elevation", "tether_azimuth"]
    s = [tofloat(d[k]) for k in keys]
    # Offsets and identified constants: near-zero process noise
    n_constants = sim.obs.tether_length + sim.obs.tether_elevation + sim.obs.tether_azimuth +
                  2 * sim.obs.dynamic_depower + sim.steering_dependent_cs +
                  2 * sim.steering_dependent_clcd + sim.steering_dependent_cl_asym + sim.drag_polar
    append!(s, fill(1e-6, n_constants))
    return s
end

"Measurement noise standard deviations, in measurement order."
function meas_stdv(tuning::TuningParameters, sim::SimulationConfig)
    d = tuning.meas_stdv
    o = sim.obs
    s = Float64[]
    o.kite_position && append!(s, fill(tofloat(d["x"]), 3))
    o.kite_velocity && append!(s, fill(tofloat(d["v"]), 3))
    append!(s, fill(tofloat(d["least_squares"]), 3))
    o.tether_length && push!(s, tofloat(d["tether_length"]))
    o.tether_elevation && push!(s, tofloat(d["tether_elevation"]))
    o.tether_azimuth && push!(s, tofloat(d["tether_azimuth"]))
    sim.enforce_vertical_wind_to_0 && push!(s, tofloat(d["z_wind"]))
    o.kite_apparent_windspeed && push!(s, tofloat(d["va"]))
    o.bridle_angle_of_attack && push!(s, tofloat(d["aoa"]))
    return s
end
