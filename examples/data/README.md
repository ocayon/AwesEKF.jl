# Example flight data

`v3_2019-10-08_min60-70.csv` holds 10 minutes (6000 samples at 10 Hz) of the LEI
V3 kite flight of 8 October 2019, in the filter's input format.

**Source:** M. Schelbergen, R. Schmehl, B. Buchholz, J. Breuer, J. Peschel (2024):
*Kite power flight data acquired on 8 October 2019* (Version 1), 4TU.ResearchData,
https://doi.org/10.4121/19376174.v1. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); this file is
redistributed under the same licence.

**Changes from the source:**

- Processed into the EKF-AWE input format with the EKF-AWE V3 preprocessing:
  - position and velocity in local east–north–up coordinates, averaged over the
    two kite sensors;
  - tether force in N;
  - time in s from the start of the log.
- Cut to minutes 60–70 of the flight.
- Reduced to the columns the filter reads, plus `flight_phase_index`.

| Column | Unit | Content |
|---|---|---|
| `time` | s | time since the start of the log |
| `kite_position_x/y/z` | m | kite position, east/north/up |
| `kite_velocity_x/y/z` | m/s | kite velocity |
| `kite_acceleration_x/y/z` | m/s² | kite acceleration |
| `ground_tether_force` | N | tether force at the ground station |
| `tether_length` | m | deployed tether length |
| `tether_reelout_speed` | m/s | reel-out speed (positive out) |
| `kite_apparent_windspeed` | m/s | pitot reading (uncalibrated) |
| `kcu_actual_steering` | - | steering actuator position |
| `kcu_actual_depower` | - | depower actuator position |
| `flight_phase_index` | - | flight phase label from the logger |
