# Small vector helpers shared by the models.
#
# The Python models are CasADi expressions, which evaluate log/acos/sqrt outside
# their domain to NaN instead of raising. The filter relies on that: a NaN state
# makes the next integration fail, which triggers a re-initialisation. The
# `n*` functions keep that behaviour (Julia's own throw a DomainError).

const Vec3{T} = SVector{3,T}

nlog(x) = x < 0 ? oftype(x, NaN) : log(x)
nacos(x) = abs(x) > 1 ? oftype(x, NaN) : acos(x)
nasin(x) = abs(x) > 1 ? oftype(x, NaN) : asin(x)

vnorm(v) = sqrt(dot(v, v))

"Component of `v` in the plane with unit normal `n`."
project_onto_plane(v, n) = v - dot(v, n) * n

"Angle between two vectors [rad]."
angle_between(a, b) = nacos(dot(a, b) / (vnorm(a) * vnorm(b)))

"Wind velocity of a logarithmic profile at height `z`."
function log_wind_velocity(uf, wdir, wvel_z, z)
    wvel = uf / KAPPA * nlog(z / Z0)
    return Vec3(wvel * cos(wdir), wvel * sin(wdir), wvel_z)
end

"Angle of attack and sideslip [deg] of the apparent wind in the frame `dcm`."
function airflow_angles(dcm, va)
    ey = dcm[:, 2]
    ez = dcm[:, 3]
    aoa = rad2deg(angle_between(ez, project_onto_plane(va, ey))) - 90
    sideslip = 90 - rad2deg(angle_between(ey, project_onto_plane(va, ez)))
    return aoa, sideslip
end

const ENU2NED = @SMatrix [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 -1.0]

"""
Roll, pitch, yaw [rad] of a rotation matrix as extrinsic x-y-z Euler angles,
R = Rz(yaw) Ry(pitch) Rx(roll) (scipy's `Rotation.as_euler("xyz")`).
"""
function euler_xyz(R)
    # scipy raises on a non-finite matrix, which makes the filter re-initialise
    # at this step; keep that behaviour
    all(isfinite, R) || error("Non-finite rotation matrix")
    pitch = -nasin(clamp(R[3, 1], -1.0, 1.0))
    roll = atan(R[3, 2], R[3, 3])
    yaw = atan(R[2, 1], R[1, 1])
    return roll, pitch, yaw
end

euler_ned(dcm_enu) = euler_xyz(ENU2NED * dcm_enu)

"""
    numpy_gradient(f, x)

`numpy.gradient(f, x)` with first-order edges: second-order central differences
inside, on a uniform or non-uniform grid, with numpy's exact formulas.
"""
function numpy_gradient(f::AbstractVector, x::AbstractVector)
    n = length(f)
    out = similar(f, Float64)
    dx = diff(x)
    if all(==(dx[1]), dx)
        h = dx[1]
        for i in 2:n-1
            out[i] = (f[i+1] - f[i-1]) / (2h)
        end
    else
        for i in 2:n-1
            dx1, dx2 = dx[i-1], dx[i]
            a = -dx2 / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            out[i] = a * f[i-1] + b * f[i] + c * f[i+1]
        end
    end
    out[1] = (f[2] - f[1]) / dx[1]
    out[n] = (f[n] - f[n-1]) / dx[end]
    return out
end

"`numpy.gradient(f)`: unit spacing."
function numpy_gradient(f::AbstractVector)
    n = length(f)
    out = similar(f, Float64)
    for i in 2:n-1
        out[i] = (f[i+1] - f[i-1]) / 2.0
    end
    out[1] = f[2] - f[1]
    out[n] = f[n] - f[n-1]
    return out
end
