import numpy as np
import xarray as xr

ph = xr.DataArray(
    np.array([[[0.7, 0.2], [0.1, 0.8]], [[0.3, 0.8], [0.9, 0.2]]]),
    coords=dict(s2=["b", "c"], s1=["b", "c"], a=["x", "y"]),
)

p = xr.DataArray(
    np.array([[[0.8, 0.1], [0.05, 0.9]], [[0.2, 0.9], [0.95, 0.1]]]),
    coords=dict(s2=["b", "c"], s1=["b", "c"], a=["x", "y"]),
)
