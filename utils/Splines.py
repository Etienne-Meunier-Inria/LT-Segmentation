def cubic_spline(x) :
    ax = abs(x)
    r = (ax < 1) * (2/3 - ax**2 + (ax**3)/2)
    r += (ax >= 1) * (ax < 2) * ((2 - ax)**3)/6
    return r

def interpolate(x, kp, kv) :
    """
    Interpolation of points using cubic spline and values of control points
    Args :
        x (n) : tensor with the position of points to interpolate in 1D
        kp (k) : tensor with the position of the control points in 1D
        kv (k, ft) : tensor with the values of the control points
    Params :
        interpolations (n, *) : interpolated values
    """
    return (cubic_spline(x[None] - kp[:, None])[..., None] *kv[:,None]).sum(axis=0)
