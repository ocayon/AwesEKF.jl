"""Reference run of the Python EKF (awes_ekf) for validating the Julia port.

Runs the same pipeline as EKF-AWE's examples/run_analysis.py (config folder ->
CSV slice -> initial state -> sensor calibration pre-run -> filter loop with
re-initialisation on failure), non-interactively and without postprocessing,
and stores everything the comparison needs in one HDF5 file:

    ekf_output/<column>   the EKFOutput columns, one row per kept step
    state/x               filter state after every kept step (N x n)
    state/P_diag          diagonal of the state covariance after every kept step
    state/names           state names
    meta                  x0, kept sample indices, failed samples, IEKF
                          iterations, the pitot calibration found by the pre-run

The port was validated against EKF-AWE commit f1b95a08c913b1d361fde51583d305f0de03fce4;
the commit of the awes_ekf checkout used is stored in meta/ekf_awe_commit.

Run with the EKF-AWE venv:

    python validation/python_reference.py --csv <processed csv> \
        --config-folder <EKF-AWE>/data/LEI-V3-KITE --system-yaml system_flown_2019.yaml \
        --start-min 60 --end-min 100 --out validation/out/py_v3_60_100.h5
"""

import argparse
import builtins
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import awes_ekf.ekf.initialize_and_update_ekf as init_module
from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
from awes_ekf.ekf.initialize_and_update_ekf import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.create_input_from_csv import (
    create_input_from_csv,
    find_initial_state_vector,
)
from awes_ekf.setup.kcu import KCU
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.settings import SimulationConfig, TuningParameters, load_config
from awes_ekf.setup.tether import Tether


def load_config_noninteractive(folder, system_yaml):
    """load_config, answering its system-variant prompt with the given file."""
    candidates = sorted(
        set(Path(folder).glob("system*.yaml")) | set(Path(folder).glob("system*.yml"))
    )
    names = [path.name for path in candidates]
    original_input = builtins.input
    builtins.input = lambda *_: str(names.index(system_yaml) + 1)
    try:
        return load_config(folder)
    finally:
        builtins.input = original_input


def ekf_awe_commit():
    """Commit of the awes_ekf checkout in use (+ "-dirty" with local changes)."""
    import subprocess
    import awes_ekf

    repo = Path(awes_ekf.__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "src"], text=True
        ).strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def slice_flight(flight_data, start_minute, end_minute):
    """The minute slice of run_analysis.py (same row arithmetic)."""
    dt = flight_data["time"].diff().mean()
    if start_minute > 0 or end_minute > 0:
        if end_minute > 0 and end_minute <= len(flight_data) / 60 * dt:
            return flight_data.iloc[
                int(start_minute * 60 / dt) : int(end_minute * 60 / dt)
            ].reset_index(drop=True)
        return flight_data.iloc[int(start_minute * 60 / dt) :].reset_index(drop=True)
    return flight_data


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

    flight_data = pd.read_csv(args.csv).reset_index()
    flight_data = slice_flight(flight_data, args.start_min, args.end_min)

    simConfig = SimulationConfig(**config_data["simulation_parameters"])
    kite = PointMassEKF(simConfig, **config_data["kite"])
    kcu = KCU(**config_data["kcu"]) if config_data["kcu"] else None
    tether = Tether(kite, kcu, simConfig.obsData, **config_data["tether"])
    kite.calc_fx = kite.get_fx_fun(tether)
    tuningParams = TuningParameters(config_data["tuning_parameters"], simConfig)

    ekf_input_list = create_input_from_csv(
        flight_data, kite, kcu, tether, simConfig, kite_sensor=0
    )
    x0 = find_initial_state_vector(
        tether, ekf_input_list[0], simConfig, wind_velocity=simConfig.initial_wind_velocity
    )
    x0_initial = np.array(x0, dtype=float)

    # Record the pitot calibration the pre-run finds
    calibrations = []
    original_fit = init_module.find_pitot_calibration

    def recording_fit(*fit_args, **fit_kwargs):
        calibration = original_fit(*fit_args, **fit_kwargs)
        calibrations.append((calibration.k, calibration.b))
        return calibration

    init_module.find_pitot_calibration = recording_fit
    t_start = time.time()
    ekf, ekf_input_list = initialize_ekf(
        ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether
    )
    t_prerun = time.time() - t_start
    init_module.find_pitot_calibration = original_fit

    # Main loop, as in run_analysis.py
    outputs, states, p_diag, kept, failed, iterations = [], [], [], [], [], []
    t_start = time.time()
    for k, ekf_input in enumerate(ekf_input_list):
        try:
            ekf, ekf_output = propagate_state_EKF(
                ekf, ekf_input, simConfig, tether, kite, kcu
            )
            outputs.append(ekf_output)
            states.append(np.array(ekf.x_k1_k1, dtype=float))
            p_diag.append(np.diag(ekf.P_k1_k1).copy())
            kept.append(k)
            iterations.append(ekf.IEKF_itts)
        except Exception as e:
            print(f"Step {k} failed: {e}")
            failed.append(k)
            try:
                x0 = find_initial_state_vector(tether, ekf_input, simConfig)
            except Exception:
                x0 = ekf.x_k1_k1
            ekf, ekf_input_list[k::] = initialize_ekf(
                ekf_input_list[k::],
                simConfig,
                tuningParams,
                x0,
                kite,
                kcu,
                tether,
                find_offsets=False,
            )
        if k % 3000 == 0:
            print(f"step {k}/{len(ekf_input_list)}  {time.time() - t_start:.1f} s")
    t_main = time.time() - t_start
    print(f"Pre-run {t_prerun:.1f} s, main loop {t_main:.1f} s")

    df = convert_ekf_output_to_df(outputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as hf:
        group = hf.create_group("ekf_output")
        for column in df.columns:
            values = df[column].to_numpy()
            if values.dtype == object:
                values = np.array(
                    [np.nan if v is None else float(v) for v in values], dtype=float
                )
            group.create_dataset(column, data=values.astype(float))
        state = hf.create_group("state")
        state.create_dataset("x", data=np.array(states))
        state.create_dataset("P_diag", data=np.array(p_diag))
        state.create_dataset("names", data=np.array(kite.state_names, dtype="S"))
        meta = hf.create_group("meta")
        meta.create_dataset("x0", data=x0_initial)
        meta.create_dataset("kept", data=np.array(kept, dtype=np.int64))
        meta.create_dataset("failed", data=np.array(failed, dtype=np.int64))
        meta.create_dataset("iekf_iterations", data=np.array(iterations, dtype=np.int64))
        meta.create_dataset(
            "pitot_calibration", data=np.array(calibrations[0] if calibrations else [np.nan, np.nan])
        )
        meta.attrs["seconds_prerun"] = t_prerun
        meta.attrs["seconds_main"] = t_main
        meta.attrs["n_samples"] = len(ekf_input_list)
        meta.attrs["ekf_awe_commit"] = ekf_awe_commit()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
