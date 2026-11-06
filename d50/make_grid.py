#!/usr/bin/env python
"""
Builds one XBeach case's grid, bathymetry and params.txt (varying D50).

  python make_grid.py <case_folder>     reads <case_folder>/grid.json and writes
                                        x.grd, y.grd, bathy.dep, params.txt into it
  python make_grid.py --dx 5 --out .    writes the files for given settings

grid.json holds each case's own settings, e.g.
  {"dx": 5.0, "output": "mean", "d50": 0.0002, "d90": 0.0003, ...}
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------- bathymetry (as in hy_setup)
Z0 = 20.0             # offshore depth (m)
THETA_1 = 0.5         # slope 1 (degrees)
THETA_2 = 0.5         # slope 2 (degrees)
X_FLAT_LEN = 750.0    # flat area before shore face (m)
LENGTH_Y = 2000.0     # alongshore domain length (m), cyclic
SMOOTH_M = 200.0      # smoothing length in metres (= sigma_smooth 40 points at 5 m)
W_SZONE = 600.0       # surfzone width (m)
W_BEACH = 300.0       # beach width (m)

# ---------------------------------------------------------------- model time / output
TSTOP = 3600
TINTM = 600           # mean-output interval (s); q_tot's averaging start must be a multiple
MEAN_VARS = ["Svsg", "Svbg", "Susg", "Subg"]


def wavelength(T, h, g=9.81):
    """Linear-theory wavelength (m) for period T (s) in depth h (m)."""
    w = 2 * np.pi / T
    k = w ** 2 / g
    for _ in range(200):
        k = w ** 2 / (g * np.tanh(k * h))
    return 2 * np.pi / k


def x_total():
    tan1 = np.tan(np.deg2rad(THETA_1))
    tan2 = np.tan(np.deg2rad(THETA_2))
    L1_width = (-(W_SZONE * tan2) + Z0) / tan1
    return X_FLAT_LEN + L1_width + W_SZONE + W_BEACH, L1_width


def grid_axes(dx):
    """Cross-shore and alongshore coordinates spanning the fixed domain at (about) spacing dx."""
    xt, _ = x_total()
    nx = max(1, int(round(xt / dx)))
    ny = max(1, int(round(LENGTH_Y / dx)))
    return np.linspace(0.0, xt, nx + 1), np.linspace(0.0, LENGTH_Y, ny + 1)


def grid_info(dx):
    x, y = grid_axes(dx)
    return {"nx": len(x) - 1, "ny": len(y) - 1, "dx_actual": float(x[1] - x[0]), "dy": float(y[1] - y[0])}


def bathymetry(dx):
    """X, Y (m) and elevation (m, positive up) for nominal spacing dx."""
    x, y = grid_axes(dx)
    _, L1_width = x_total()
    tan1 = np.tan(np.deg2rad(THETA_1))
    tan2 = np.tan(np.deg2rad(THETA_2))
    xr = x - X_FLAT_LEN
    z_at_end_l1 = tan1 * L1_width - Z0
    f_x = np.where(xr < 0, -Z0,
                   np.where(xr < L1_width, tan1 * xr - Z0, z_at_end_l1 + tan2 * (xr - L1_width)))
    elevation_1d = gaussian_filter1d(f_x, sigma=SMOOTH_M / (x[1] - x[0]), mode="nearest")
    X, Y = np.meshgrid(x, y)
    return X, Y, np.tile(elevation_1d, (len(y), 1))


D50_DEFAULT = 0.0002      # median grain size (m) of the main study
D90_DEFAULT = 0.0003


def params_text(nx, ny, output="mean", d50=D50_DEFAULT, d90=D90_DEFAULT):
    """params.txt contents: the shared setup, with this case's grid size, output type and grain sizes."""
    if output == "global":
        out_block = f"""tintm        = {TINTM}          
tintg        = 2           

nglobalvar   = 4 
Svsg
Susg
Svbg
Subg
"""
    elif output == "mean":
        out_block = f"""tintm        = {TINTM}
tintg        = {TSTOP}

nglobalvar   = 1
zb

nmeanvar     = {len(MEAN_VARS)}
""" + "".join(f"{v}\n" for v in MEAN_VARS)
    else:
        raise ValueError(f"output must be 'global' or 'mean', not {output!r}")

    return f"""
! --- Grid Parameters ---
nx           = {nx}
ny           = {ny}
coord        = cartesian   
vardx        = 1
xfile        = x.grd
yfile        = y.grd
depfile      = bathy.dep
posdwn       = 1    
posfree      = 1 
split        = 1
outputformat = netcdf

! --- Wave & Boundary Conditions ---
cyclic       = 1
wavemodel    = surfbeat
dirnorm      = 0
thetamin     = -90          
thetamax     = 90           
dtheta       = 10           
single_dir   = 0
wbctype      = parametric        
bcfile       = jonswap.txt  
front        = abs_2d
back         = wall       
zs0          = 0.0
lateralwave  = cyclic
reuse        = 0
CFL          = 0.5 

! --- Sediment Parameters (Uniform) ---
morphology   = 1
sedtrans     = 1
ngd          = 1            
D50          = {d50:.6f}
D90          = {d90:.6f}
por          = 0.4
rhos         = 2650.0     

! --- Model Time ---
tstart       = 0 
tstop        = {TSTOP}                            
""" + out_block


def write_inputs(out_dir, dx, output="mean", d50=D50_DEFAULT, d90=D90_DEFAULT):
    """Write x.grd, y.grd, bathy.dep and params.txt for nominal spacing dx into out_dir."""
    X, Y, elevation_2d = bathymetry(dx)
    np.savetxt(os.path.join(out_dir, "bathy.dep"), -elevation_2d, fmt="%.4f")
    np.savetxt(os.path.join(out_dir, "x.grd"), X, fmt="%.4f")
    np.savetxt(os.path.join(out_dir, "y.grd"), Y, fmt="%.4f")
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write(params_text(X.shape[1] - 1, X.shape[0] - 1, output, d50, d90))
    return grid_info(dx)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case_dir", nargs="?", help="case folder containing grid.json")
    p.add_argument("--dx", type=float, help="grid spacing (instead of a case folder)")
    p.add_argument("--output", default="mean", choices=["global", "mean"])
    p.add_argument("--out", default=".", help="folder to write into (with --dx)")
    p.add_argument("--d50", type=float, default=D50_DEFAULT, help="median grain size (m), with --dx")
    p.add_argument("--d90", type=float, default=D90_DEFAULT, help="90th percentile grain size (m), with --dx")
    a = p.parse_args()
    if a.case_dir:
        with open(os.path.join(a.case_dir, "grid.json")) as f:
            spec = json.load(f)
        info = write_inputs(a.case_dir, spec["dx"], spec.get("output", "mean"),
                            spec.get("d50", D50_DEFAULT), spec.get("d90", D90_DEFAULT))
    elif a.dx:
        info = write_inputs(a.out, a.dx, a.output, a.d50, a.d90)
    else:
        p.error("give a case folder or --dx")
    print(f"nx={info['nx']}, ny={info['ny']}, dx_actual={info['dx_actual']:.4f} m, dy={info['dy']:.4f} m")
