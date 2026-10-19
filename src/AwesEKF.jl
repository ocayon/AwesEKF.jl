"""
    AwesEKF

Extended Kalman filter for airborne-wind-energy flight data: a native-Julia port
of the Python package `awes_ekf` (EKF-AWE). The models are plain Julia functions;
Jacobians come from ForwardDiff and the state propagation from CVODES (Sundials),
with the same method and tolerances as the CasADi integrator the Python code uses.

Pipeline (`run_flight`):

  config folder ──► `load_config`         ekf_config.yaml + awesIO system yaml
  flight CSV    ──► `read_flight_csv`     columns, optional minute slice
                    `create_inputs`       one `EKFInput` per sample
                    `initial_state`       tether-shape solve → x0
                    `calibrate_sensors!`  pre-run sensor calibration (pitot, aoa)
                    filter loop           propagate → predict → update, with
                                          re-initialisation on failure
                    `save_results`        HDF5 in the EKF-AWE results layout
"""
module AwesEKF

using LinearAlgebra
using Printf
using Statistics
using StaticArrays
using ForwardDiff
using Sundials
using SciMLBase
using YAML
using CSV
using HDF5
using Dierckx

include("constants.jl")
include("math.jl")
include("config.jl")
include("layout.jl")
include("tether.jl")
include("model.jl")
include("propagation.jl")
include("filter.jl")
include("input.jl")
include("initial_state.jl")
include("calibration.jl")
include("output.jl")
include("run.jl")
include("save.jl")

export load_config, read_flight_csv, slice_minutes, create_inputs, SystemModel,
    initial_state, run_flight, run_filter, save_results

end
