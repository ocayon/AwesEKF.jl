"""Evaluate the Python EKF model functions at a few states, for a unit-level
comparison with the Julia port (validation/compare_model.jl).

For samples of a flight slice, at the initial state and at perturbed states:
f(x,u,x0), df/dx, h(x,u,x0), dh/dx, one propagation step, and the EKFOutput
row. Run with the EKF-AWE venv, arguments as python_reference.py.
"""

import argparse

import h5py
import numpy as np
import pandas as pd

from awes_ekf.ekf import DynamicModel, ObservationModel
from awes_ekf.ekf.ekf_output import create_ekf_output
from awes_ekf.load_data.create_input_from_csv import (
    create_input_from_csv,
    find_initial_state_vector,
)
from awes_ekf.setup.kcu import KCU
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.settings import SimulationConfig, TuningParameters
from awes_ekf.setup.tether import Tether
from awes_ekf.ekf.kalman_filter import ExtendedKalmanFilter
from python_reference import load_config_noninteractive, slice_flight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config-folder", required=True)
    parser.add_argument("--system-yaml", required=True)
    parser.add_argument("--start-min", type=float, default=0)
    parser.add_argument("--end-min", type=float, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_data = load_config_noninteractive(args.config_folder, args.system_yaml)
    flight_data = slice_flight(pd.read_csv(args.csv).reset_index(), args.start_min, args.end_min)
    simConfig = SimulationConfig(**config_data["simulation_parameters"])
    kite = PointMassEKF(simConfig, **config_data["kite"])
    kcu = KCU(**config_data["kcu"])
    tether = Tether(kite, kcu, simConfig.obsData, **config_data["tether"])
    kite.calc_fx = kite.get_fx_fun(tether)
    tuningParams = TuningParameters(config_data["tuning_parameters"], simConfig)
    inputs = create_input_from_csv(flight_data, kite, kcu, tether, simConfig)
    x0 = find_initial_state_vector(tether, inputs[0], simConfig, wind_velocity=simConfig.initial_wind_velocity)

    dyn = DynamicModel(kite, tether, simConfig)
    obs = ObservationModel(dyn.x, dyn.u, simConfig, kite, tether, kcu)
    ekf = ExtendedKalmanFilter(
        tuningParams.stdv_dynamic_model, tuningParams.stdv_measurements,
        dyn, obs, kite, tether, kcu, simConfig,
    )

    rng = np.random.default_rng(0)
    samples = [0, 1000, 5000]
    records = []
    for k in samples:
        ekf.update_input_vector(inputs[k])
        u = np.array(ekf.u, dtype=float)
        # A state near the measurements of sample k, plus a perturbation
        xk = np.array(x0, dtype=float)
        xk[0:3] = inputs[k].kite_position
        xk[3:6] = inputs[k].kite_velocity
        for scale in [0.0, 1.0]:
            x = xk + scale * rng.normal(0, 1, xk.size) * (np.abs(xk) * 0.01 + 0.05)
            rec = {
                "sample": k,
                "x": x,
                "u": u,
                "f": np.array(dyn.calc_fx(x, u, x)).ravel(),
                "F": np.array(ekf.calc_Fx(x, u, x)),
                "h": np.array(ekf.calc_hx(x, u, x)).ravel(),
                "H": np.array(ekf.calc_Hx(x, u, x)),
                "ts": inputs[k].timestep,
                "x_next": np.array(kite.propagate(x, u, inputs[k].timestep)).ravel(),
            }
            out = create_ekf_output(x, u, inputs[k], tether, kite, simConfig)
            rec["output"] = {key: (np.nan if v is None else float(v)) for key, v in vars(out).items()}
            records.append(rec)

    with h5py.File(args.out, "w") as hf:
        hf.attrs["cdt_area"] = kcu.cd_area_axial
        hf.attrs["cdp_area"] = kcu.cd_area_crossflow
        hf.create_dataset("x0", data=np.array(x0, dtype=float))
        for i, rec in enumerate(records):
            g = hf.create_group(f"probe_{i}")
            g.attrs["sample"] = rec["sample"]
            g.attrs["ts"] = rec["ts"]
            for key in ["x", "u", "f", "F", "h", "H", "x_next"]:
                g.create_dataset(key, data=rec[key])
            go = g.create_group("output")
            for key, value in rec["output"].items():
                go.attrs[key] = value
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
