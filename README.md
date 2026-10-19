# AwesEKF.jl

[![CI](https://github.com/ocayon/AwesEKF.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ocayon/AwesEKF.jl/actions/workflows/CI.yml)

Iterated extended Kalman filter for wind and state estimation of tethered flying
systems (airborne wind energy), in native Julia.

This is a port of the Python package [EKF-AWE](https://github.com/awegroup/EKF-AWE)
(`awes_ekf`), the filter described in

> O. Cayon, S. Watson, R. Schmehl: *Kite as a sensor: wind and state estimation in
> tethered flying systems*, Wind Energ. Sci. 10, 2161–2188, 2025,
> https://doi.org/10.5194/wes-10-2161-2025

It contains only what is needed to run and use the filter: no plotting and no
postprocessing. Given the same configuration and flight data, it reproduces the
Python results (see [Validation](#validation)).

**Reference implementation.** Ported from, and validated against, EKF-AWE commit
[`f1b95a0`](https://github.com/awegroup/EKF-AWE/commit/f1b95a08c913b1d361fde51583d305f0de03fce4)
(`f1b95a08c913b1d361fde51583d305f0de03fce4`, 2026-10-19, "fix(tether): pair KCU
drag coefficients with their areas; onboard turbine drag"). Changes to the
Python filter after that commit are not in this port.

## Method in brief

- **State**: kite position and velocity, wind velocity, aerodynamic coefficients
  (CL, CD, CS), tether length and the elevation/azimuth of the first tether
  element. Optionally it also holds sensor offsets and identified constants:
  depower and the steering-dependent aero terms.
- **Dynamics**: the kite is a point mass. The tether force at the kite comes from
  a quasi-static lumped-mass tether, with the KCU as a separate point mass at the
  end of a bridle segment.
- **Jacobians** (paper Sect. 4.3, Fig. 6): the tether model inside f and h is
  differentiated only with respect to tether length, elevation and azimuth. The
  kite kinematics and wind it uses are taken from the nominal state x\*, which is
  held fixed. In code, `dynamics(m, x, u, x0)` and `observation(m, x, u, x0)`
  read the tether states from `x` and the rest from `x0`, and ForwardDiff
  differentiates with respect to `x` only.
- **Propagation**: CVODES (Sundials.jl), BDF with a dense linear solver and the
  exact ForwardDiff Jacobian. Tolerances are abstol 1e-8 and reltol 1e-6, the
  defaults of the CasADi integrator in the Python version.
- **Update**: an iterated EKF with the Joseph-form covariance update.
- **Sensor calibration**: an optional pre-run without the pitot tube (or the
  angle-of-attack vane), then a dynamic-pressure calibration (or offset) fitted
  against the filter estimate.

## Installation

Requires Julia ≥ 1.10. The package is not in the General registry; install it
from GitHub:

```julia
julia> ] add https://github.com/ocayon/AwesEKF.jl
```

or, for development, clone the repository and instantiate it:

```julia
julia> ] activate .
(AwesEKF) pkg> instantiate
```

## Quick start

The repository includes a 10-minute example from the public LEI V3 flight of
8 October 2019. It contains the flight data, the as-flown system and the tuned
filter configuration:

```
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project examples/run_v3_example.jl
```

This takes about 30 s, including compilation. It prints the estimated wind
speed, wind direction and aerodynamic coefficients per minute, and writes the
full results to `results/v3_example.h5`. Julia and Python agree on this example
to within 2e-5 of each signal's standard deviation (`validation/compare_runs.jl`).

| Path | Contents |
|---|---|
| `examples/data/` | The flight data slice, its source and licence (CC BY 4.0) |
| `examples/LEI-V3-KITE/` | Filter configuration and system description |

**Other flights.** The filter needs a processed flight CSV in the EKF-AWE input
format (the columns listed in `examples/data/README.md`) and a kite folder with
`ekf_config.yaml` and a `system*.yaml`. Raw logs are converted with the
preprocessing scripts of [EKF-AWE](https://github.com/awegroup/EKF-AWE)
(`data/data_preprocessors/`); its `data/` folder holds the kite configurations.

## Usage

The configuration follows the EKF-AWE/AWETrim convention: a kite folder holding
`ekf_config.yaml` (filter settings and tuning) and one or more awesIO
`system*.yaml` files (the physical system).

```julia
using AwesEKF

result, flight_data, cfg = run_flight(
    "path/to/processed_flight.csv",       # EKF-AWE input CSV spec
    "path/to/LEI-V3-KITE";                # kite config folder
    system_yaml = "system_flown_2019.yaml",
    start_minute = 60, end_minute = 100,  # optional slice
    out = "results/run.h5",               # optional results file
)

result.outputs["wind_speed_horizontal"]   # one value per kept step
result.x                                  # state history (steps x states)
```

From the command line:

```
julia --project scripts/run_flight.jl --csv <file> --config <kite folder> \
      --system-yaml system_flown_2019.yaml --start 60 --end 100 --out run.h5
```

The results file uses the EKF-AWE HDF5 layout (`ekf_output`, `flight_data`,
`config_data`). It adds `state` (state history, covariance diagonal) and `meta`
(initial state, kept and failed samples, IEKF iterations, calibrations).

## Code structure

| File | Contents |
|---|---|
| `src/config.jl` | Config loading (ekf_config.yaml + awesIO system yaml), KCU and tether parameters, noise settings |
| `src/layout.jl` | State, input and measurement vector layouts (same order as the Python code) |
| `src/tether.jl` | Quasi-static tether model with KCU and bridle; derived frames and coefficients |
| `src/model.jl` | Process model f(x, u, x0) and observation model h(x, u, x0) |
| `src/propagation.jl` | One-step state propagation with CVODES |
| `src/filter.jl` | (Iterated) EKF: covariance prediction, measurement update, `step!` |
| `src/input.jl` | Flight CSV reading, minute slicing, per-sample filter inputs |
| `src/initial_state.jl` | Tether-shape solve for the initial state |
| `src/calibration.jl` | Sensor calibration pre-run (pitot, angle of attack) |
| `src/output.jl` | Output row per step (wind, aero coefficients, orientations, ...) |
| `src/run.jl` | `run_filter` (with re-initialisation on failure), `run_flight` |
| `src/save.jl` | HDF5 results file |

## Validation

`validation/` compares the port with the Python implementation. Flight data and
results stay local; they are never committed.

- `python_reference.py` runs the Python pipeline non-interactively and stores
  states, outputs and run metadata (run it with the EKF-AWE environment).
- `python_model_probe.py` + `compare_model.jl` compare f, ∂f/∂x, h, ∂h/∂x, one
  propagation step and the output row at given states.
- `compare_runs.jl` compares two results files sample by sample.

Validated against EKF-AWE `f1b95a0` on two LEI V3 flights, with the
configuration in EKF-AWE's `data/LEI-V3-KITE`: all steering-dependent stages,
dynamic depower, pitot pre-run calibration and the zero-vertical-wind
pseudo-measurement on.

**Model functions** (at the initial state and at perturbed states of 3 samples):
f, h and both Jacobians agree to ≤ 1e-15 relative and the output rows to 1e-9.
One propagation step agrees to ~1e-7, the integrator tolerance. The initial
state agrees to 4e-16.

**Full runs**, Julia against Python. "Independent" means each implementation
computes its own initial state and pitot calibration. "Lockstep" means Julia
takes both from the Python run.

| | 2019-10-08, min 60–100 | 2025-10-09, min 0–28 |
|---|---|---|
| Filter steps | 24 000 | 16 799 |
| Failed steps (re-initialisations) | 0 / 0 | 1 / 1, same sample |
| IEKF iterations identical | 100 % of steps | 100 % of steps |
| Pitot calibration k (Julia / Python) | 0.680554 / 0.680540 | 0.804635 / 0.804635 |
| Worst channel, RMS difference / signal std, independent | 9.2e-4 | 1.3e-4 |
| Worst channel, RMS difference / signal std, lockstep | 4.2e-5 | 1.3e-4 |
| Horizontal wind speed, max abs difference | 0.45 mm/s | 0.21 mm/s |
| Wind direction, max abs difference | 8.5e-5 rad | 1.2e-4 rad |
| Lift coefficient, max abs difference | 8.4e-5 | 1.7e-5 |
| Drag coefficient, max abs difference | 1.2e-5 | 1.9e-5 |
| Runtime, pre-run + main loop, Python | 168 + 139 s | 133 + 184 s |
| Runtime, pre-run + main loop, Julia | 14 + 18 s | 18 + 15 s |
| Speed-up | 9.6× | 9.6× |

Timings are for one process on the same Windows 11 laptop, with Julia 1.13.2
against Python 3.13 / CasADi 3.7.0. The Julia pre-run time includes JIT
compilation.

In the independent 2019 run, the larger differences come from the pre-run: its
pitot coefficient differs by 2e-5 relative, which then scales the measured
apparent wind speed of the whole run. With the calibration shared (lockstep), the
remaining differences are integrator-level.

Reproduce (Python with the EKF-AWE environment, files go to `validation/out/`):

```
python validation/python_reference.py --csv <flight.csv> --config-folder <EKF-AWE>/data/LEI-V3-KITE \
    --system-yaml system_flown_2019.yaml --start-min 60 --end-min 100 --out validation/out/py.h5
julia --project scripts/run_flight.jl --csv <flight.csv> --config <EKF-AWE>/data/LEI-V3-KITE \
    --system-yaml system_flown_2019.yaml --start 60 --end 100 --out validation/out/jl.h5
julia --project validation/compare_runs.jl validation/out/jl.h5 validation/out/py.h5
```

Add `--x0-from validation/out/py.h5 --pitot-from validation/out/py.h5` to the
Julia run for lockstep.

## Known behaviours kept for identical results

These are reproduced on purpose so both implementations give the same results.
Fix them in both codes together:

- Output columns for states that are not in the configuration
  (`tether_elevation_offset` / `tether_azimuth_offset` without angle
  measurements, `k_cl_up` / `k_cd_up` without dynamic depower,
  `kite_thrust_force`) repeat the first state, the kite x position.
- When a step fails, the filter is re-initialised from a fresh tether-shape solve
  with the default wind guess `[0.001, 8, 0]`, not the configured one.

- The tether-shape solve for the (re-)initial state is a port of scipy's
  `least_squares` as the Python code calls it: TRF without bounds, a 2-point
  finite-difference Jacobian, and xtol = ftol = 1e-5. At low tether force the
  closure has no exact solution (the modelled tether sags to the ground). The
  solver then stops in an algorithm-dependent local minimum, which can be tens
  of metres off, and the filter restarts from it. A different solver would
  restart elsewhere.
- In the calibration pre-run, a failed integration leaves the state and the
  input unchanged. The next sample therefore retries the identical integration,
  so once the pre-run fails it keeps failing for the rest of the window.
  (2019-10-08 from minute 0 triggers this in both implementations.)

## Not ported

Plotting, postprocessing (IMU offset removal, flight-phase labels, the
turn-rate law), the data preprocessors, and the `model_yaw` option (flagged
"don't use" in the Python code).

## Citation

If you use this code, please cite the paper above and the original
implementation: O. Cayon, *Extended Kalman Filter for Airborne Wind Energy Systems
(AWES)*, Zenodo, https://doi.org/10.5281/zenodo.15269632.

## Acknowledgements

This work was supported by the European Union's Horizon Europe programme
(Climate, Energy and Mobility) through the MERIDIONAL project, grant agreement
No. 101084216.

## License

This work is licensed under the MIT open source licence. See the [LICENSE](LICENSE)
file for details.

### Copyright

Technische Universiteit Delft hereby disclaims all copyright interest in the program "AwesEKF.jl" (an iterated extended Kalman filter for wind and state estimation of airborne wind energy systems, ported to Julia from EKF-AWE) written by the Author(s).

Prof. Dr. H.G.C. (Henri) Werij, Dean of the Faculty of Aerospace Engineering

Copyright &copy; 2024-2026 Oriol Cayon

AwesEKF.jl is a derivative of EKF-AWE (Copyright &copy; 2024 Oriol Cayon, MIT
License, for which TU Delft likewise disclaims copyright interest).
