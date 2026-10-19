# Atmospheric constants (identical to awes_ekf.setup.settings)
const RHO = 1.225    # Air density [kg/m^3]
const KAPPA = 0.4    # Von Karman constant [-]
const G = 9.81       # Gravity acceleration [m/s^2]
const Z0 = 0.01      # Surface roughness [m]

# Integrator settings: the defaults of CasADi's cvodes plugin, which the
# Python implementation uses (checked against casadi 3.7 output)
const CVODE_ABSTOL = 1e-8
const CVODE_RELTOL = 1e-6
const CVODE_MAX_STEPS = 10_000
