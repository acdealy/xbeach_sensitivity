#!/usr/bin/env python
"""
Post-processing for the bed-slope sensitivity study.

Modes
-----
One case (called by run_case.sh right after XBeach finishes):
    python post_process.py --case xbeach_cases/run_z0_15.00m

Collect every finished case into the summary CSV (safe to run any time, even mid-batch):
    python post_process.py --case_list case_list.txt --out q_tot_summary_slope.csv

Write the list of cases that still need running (feed this to the SLURM array);
--dx_min / --dx_max keep only a band of grid spacings, so each band gets its own time limit:
    python post_process.py --case_list case_list.txt --write_pending pending.txt
    python post_process.py --case_list case_list.txt --write_pending pending_dx1-2.txt --dx_min 1 --dx_max 2

(Re)compute q_tot.json for cases with complete output but no up-to-date result:
    python post_process.py --case_list case_list.txt --compute_missing --workers 8
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import glob
import json
import re
import sys
import traceback

import numpy as np

CASE_GLOB = "run_*"
NC_FILE = "xboutput.nc"
RESULT_FILE = "q_tot.json"
ERROR_FILE = "q_tot_error.json"
MEAN_FILE = "xb_timemean.nc"
LOCK_FILE = ".running"
MEAN_VARS = ["Svsg", "Svbg", "Susg", "Subg"]
# Output before this time (s) is left out of the time average (spin-up).
# Change it only here: results computed with a different value are redone automatically.
T_AVG_START = 600.0

COLUMNS = ["z0", "slope_deg", "q_tot", "tp", "dx", "hm0", "mainang", "s0", "d50", "d90",
           "dx_actual", "dy", "folder", "t_avg_start"]
EXIT_BAD_OUTPUT = 3  # xboutput.nc missing, truncated or unreadable -> XBeach should be re-run


class BadOutputError(RuntimeError):
    """xboutput.nc is missing, incomplete or unreadable (as opposed to a bug in this script)."""


# ----------------------------------------------------------------------------- helpers
def write_json_atomic(path, obj):
    """Write to a temp file then rename, so a killed job never leaves a half-written file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_param(params_path, key):
    """Read a numeric value like 'tstop = 3600' from an XBeach params.txt (None if absent)."""
    if not os.path.exists(params_path):
        return None
    with open(params_path) as f:
        for line in f:
            line = line.split("!")[0]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip().lower() == key.lower():
                try:
                    return float(v.split()[0])
                except (ValueError, IndexError):
                    return None
    return None


def load_current_result(case_dir):
    """q_tot.json contents if it exists AND was computed with the current T_AVG_START, else None."""
    path = os.path.join(case_dir, RESULT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            res = json.load(f)
    except (OSError, ValueError):
        return None
    return res if res.get("t_avg_start") == T_AVG_START else None


def grid_spec(case_dir):
    """The case's grid.json ({"dx": ..., "dx_actual": ..., "dy": ..., "output": ...})."""
    path = os.path.join(case_dir, "grid.json")
    if not os.path.exists(path):
        return {"dx": float("nan")}
    with open(path) as f:
        return json.load(f)


def list_case_dirs(run_dir, case_list=None):
    if case_list:
        with open(case_list) as f:
            return [line.strip().rstrip("/") for line in f if line.strip()]
    return sorted(d.rstrip("/") for d in glob.glob(os.path.join(run_dir, CASE_GLOB)) if os.path.isdir(d))


# ----------------------------------------------------------------------------- core computation
def compute_case(case_dir, write_mean=True):
    """
    Same q_tot as the main study: time-mean over t >= T_AVG_START of Svsg + Svbg. Also:
      * refuses to process an incomplete run (last output time < tstop), so a job that
        was killed mid-run is re-run instead of silently giving a wrong q_tot
      * reads each variable once and also saves the time-mean fields to xb_timemean.nc
        (a few MB) -- enough to recompute q_tot or redo the transport maps later even
        if the multi-GB xboutput.nc is deleted.
    mean(Svsg + Svbg) == mean(Svsg) + mean(Svbg), so the result is identical.
    """
    import xarray as xr

    case_dir = os.path.abspath(case_dir)
    folder = os.path.basename(case_dir)
    wave_path = os.path.join(case_dir, "jonswap.txt")
    hm0_val, tp_val, ang_val = (read_param(wave_path, k) for k in ("Hm0", "Tp", "mainang"))
    if None in (hm0_val, tp_val, ang_val):
        raise ValueError("could not read Hm0, Tp and mainang from jonswap.txt")
    s0_val = hm0_val / (1.56 * tp_val ** 2)
    grid = grid_spec(case_dir)

    nc_path = os.path.join(case_dir, NC_FILE)
    if not os.path.exists(nc_path):
        raise BadOutputError(f"{NC_FILE} not found")

    params_path = os.path.join(case_dir, "params.txt")
    tstop = read_param(params_path, "tstop")
    tintg = read_param(params_path, "tintg") or 1.0

    try:
        ds = xr.open_dataset(nc_path, chunks={"globaltime": 20})
        t = ds["globaltime"].values
        t_end = float(t[-1])
        mean_mode = f"{MEAN_VARS[0]}_mean" in ds
        if mean_mode:
            mt = ds["meantime"].values.astype(float)
            t_end = max(t_end, float(mt[-1]))
    except Exception as e:
        raise BadOutputError(f"could not read {NC_FILE}: {type(e).__name__}: {e}") from e
    with ds:
        if mean_mode:
            # XBeach's time-mean output: each record averages the interval ending at its meantime
            if tstop is not None and t_end < tstop - 1:
                raise BadOutputError(f"incomplete run: last output at t={t_end:g}s but tstop={tstop:g}s")
            if tstop is not None and mt[-1] < tstop - 1:
                raise ValueError(f"mean output ends at t={mt[-1]:g}s although the run reached tstop; "
                                 "meantime may label interval starts, not ends -- check before trusting q_tot")
            tstart = read_param(params_path, "tstart") or 0.0
            starts = np.concatenate([[tstart], mt[:-1]])
            keep = starts >= T_AVG_START - 1e-6
            if not keep.any() or not np.isclose(starts[keep][0], T_AVG_START):
                raise ValueError(f"T_AVG_START={T_AVG_START:g}s is not at a mean-output interval boundary "
                                 f"(intervals start at {starts.tolist()})")
            idx = np.flatnonzero(keep)
            weights = xr.DataArray((mt - starts)[idx] / (mt - starts)[idx].sum(), dims="meantime")
            present = [v for v in MEAN_VARS if f"{v}_mean" in ds]
            sel = ds[[f"{v}_mean" for v in present]].isel(meantime=idx)
            means = (sel * weights).sum(dim="meantime").compute()
            means = means.rename({f"{v}_mean": v for v in present})
            n_avg = int(len(idx))
        else:
            if tstop is not None and t_end < tstop - 1.5 * tintg:
                raise BadOutputError(f"incomplete run: last output at t={t_end:g}s but tstop={tstop:g}s")
            keep = t >= T_AVG_START
            if not keep.any():
                raise BadOutputError(f"no output at or after t={T_AVG_START:g}s")
            window = ds.isel(globaltime=slice(int(np.argmax(keep)), None))
            n_avg = int(window.sizes["globaltime"])
            present = [v for v in MEAN_VARS if v in ds]
            if "Svsg" not in present or "Svbg" not in present:
                raise ValueError("no Svsg/Svbg snapshots and no Svsg_mean/Svbg_mean in xboutput.nc; "
                                 f"variables found: {sorted(ds.data_vars)}")
            means = window[present].mean(dim="globaltime").compute()
        globalx = ds["globalx"].compute()
        globaly = ds["globaly"].compute()

    time_avg = means["Svsg"] + means["Svbg"]
    y_mask = (globaly >= 800) & (globaly <= 1200)
    transport_y_filtered = time_avg.where(y_mask, drop=True)
    # 20 evenly-spaced alongshore profiles inside the y-window
    n_profiles = transport_y_filtered.sizes["ny"]
    idx = np.unique(np.linspace(0, n_profiles - 1, 20).round().astype(int))
    profile_1d = transport_y_filtered.isel(ny=idx).mean(dim="ny")
    x_coord_1d = globalx.where(y_mask, drop=True).isel(ny=idx).mean(dim="ny")
    profile_1d = profile_1d.assign_coords(x_coord=("nx", x_coord_1d.data))
    q_tot_val = float(profile_1d.integrate(coord="x_coord").squeeze().item())

    if write_mean:
        out = means.astype("float32")
        out["globalx"] = globalx
        out["globaly"] = globaly
        out.attrs.update(
            description="time-mean of xboutput.nc over t >= t_avg_start",
            source="XBeach mean output" if mean_mode else "2-s snapshots (global output)",
            t_avg_start=T_AVG_START,
            t_end=t_end,
            n_times_averaged=n_avg,
        )
        tmp = os.path.join(case_dir, f"{MEAN_FILE}.tmp.{os.getpid()}")
        out.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in present})
        os.replace(tmp, os.path.join(case_dir, MEAN_FILE))

    return {
        "z0": float(grid.get("z0", np.nan)),
        "slope_deg": float(grid.get("slope_deg", np.nan)),
        "d50": float(grid.get("d50", 0.0002)),
        "d90": float(grid.get("d90", 0.0003)),
        "tp": tp_val,
        "dx": float(grid["dx"]),
        "q_tot": q_tot_val,
        "hm0": hm0_val,
        "mainang": ang_val,
        "s0": s0_val,
        "dx_actual": float(grid.get("dx_actual", np.nan)),
        "dy": float(grid.get("dy", np.nan)),
        "folder": folder,
        "t_avg_start": T_AVG_START,
        "output": "mean" if mean_mode else "global",
        "t_end": t_end,
        "n_times_averaged": n_avg,
    }


def process_and_record(case_dir, write_mean=True, delete_nc=False, dask_threads=None):
    """
    Compute one case and write q_tot.json (or q_tot_error.json).
    Returns (code, message): 0 = ok, EXIT_BAD_OUTPUT = output missing/incomplete, 1 = other error.
    """
    if dask_threads:
        import dask

        dask.config.set(scheduler="threads", num_workers=dask_threads)

    case_dir = os.path.abspath(case_dir)
    folder = os.path.basename(case_dir)
    err_path = os.path.join(case_dir, ERROR_FILE)
    try:
        res = compute_case(case_dir, write_mean=write_mean)
    except Exception as e:
        write_json_atomic(err_path, {
            "folder": folder,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        })
        code = EXIT_BAD_OUTPUT if isinstance(e, BadOutputError) else 1
        return code, f"✗ {folder}: {type(e).__name__}: {e}"

    write_json_atomic(os.path.join(case_dir, RESULT_FILE), res)
    if os.path.exists(err_path):
        os.remove(err_path)
    if delete_nc and write_mean and os.path.exists(os.path.join(case_dir, MEAN_FILE)):
        os.remove(os.path.join(case_dir, NC_FILE))
    return 0, f"✓ {folder}: q_tot = {res['q_tot']:.6g}"


# ----------------------------------------------------------------------------- batch modes
def compute_missing(dirs, workers, write_mean):
    todo = []
    for d in dirs:
        if load_current_result(d) is not None:
            continue
        if os.path.exists(os.path.join(d, LOCK_FILE)):
            print(f"  skipping {os.path.basename(d)} (a job is running it, or was killed: {LOCK_FILE} present)")
            continue
        if os.path.exists(os.path.join(d, NC_FILE)):
            todo.append(d)
    print(f"Computing {len(todo)} case(s) that have {NC_FILE} but no up-to-date {RESULT_FILE} "
          f"(t_avg_start={T_AVG_START:g}s), {workers} worker(s)")
    if not todo:
        return

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    threads = max(1, cpus // workers)

    if workers == 1:
        for i, d in enumerate(todo, 1):
            _, msg = process_and_record(d, write_mean, dask_threads=threads)
            print(f"[{i}/{len(todo)}] {msg}", flush=True)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(process_and_record, d, write_mean, False, threads) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _, msg = fut.result()
            except Exception as e:  # worker process crashed (e.g. out of memory)
                msg = f"✗ worker crashed: {type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {msg}", flush=True)


def collect(dirs, out):
    import pandas as pd

    results, errors, pending, stale = [], [], [], 0
    for d in dirs:
        err_path = os.path.join(d, ERROR_FILE)
        res = load_current_result(d)
        if res is not None:
            results.append(res)
            continue
        if os.path.exists(os.path.join(d, RESULT_FILE)):
            stale += 1
        if os.path.exists(err_path):
            with open(err_path) as f:
                errors.append(json.load(f))
        else:
            pending.append(os.path.basename(d))

    df = pd.DataFrame(results, columns=COLUMNS)
    tmp = f"{out}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)

    root, ext = os.path.splitext(out)
    err_out = f"{root}_errors{ext or '.csv'}"
    if errors:
        pd.DataFrame(errors).to_csv(err_out, index=False)
    elif os.path.exists(err_out):
        os.remove(err_out)

    print(f"{len(dirs)} cases: {len(results)} done, {len(errors)} with errors, {len(pending)} not run yet "
          f"(t_avg_start={T_AVG_START:g}s)")
    if stale:
        print(f"  {stale} case(s) have an old q_tot.json from a different averaging window (counted as not done)")
    print(f"Wrote {out}" + (f" and {err_out}" if errors else ""))


def write_pending(dirs, path, dx_min=None, dx_max=None):
    """Unfinished cases, optionally only those with dx_min <= dx < dx_max."""
    lo = -np.inf if dx_min is None else dx_min
    hi = np.inf if dx_max is None else dx_max
    dirs = [d for d in dirs if lo <= grid_spec(d)["dx"] < hi]
    pending = [d for d in dirs if load_current_result(d) is None]
    with open(path, "w") as f:
        f.write("".join(f"{d}\n" for d in pending))
    which = f" with {lo:g} <= dx < {hi:g} m" if (dx_min is not None or dx_max is not None) else ""
    print(f"{len(pending)} of {len(dirs)} cases{which} still need running -> {path}")
    if pending:
        print(f"Submit with:  sbatch --time=... --mem=... --array=0-{len(pending) - 1}%5 xbeach_array.sbatch {path}"
              "")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", help="process a single case folder -> q_tot.json")
    p.add_argument("--is_done", metavar="CASE",
                   help="exit 0 if CASE has an up-to-date q_tot.json, 1 if not (used by run_case.sh)")
    p.add_argument("--run_dir", default="xbeach_cases", help="folder holding the run_* case folders")
    p.add_argument("--case_list", help="text file of case folders (one per line) to use instead of globbing run_dir")
    p.add_argument("--out", default="q_tot_summary_slope.csv", help="summary CSV to write")
    p.add_argument("--write_pending", metavar="FILE", help="write unfinished case folders to FILE and exit")
    p.add_argument("--dx_min", type=float, help="with --write_pending: only cases with dx >= this")
    p.add_argument("--dx_max", type=float, help="with --write_pending: only cases with dx < this")
    p.add_argument("--compute_missing", action="store_true",
                   help="before collecting, compute q_tot.json for cases with xboutput.nc but no result")
    p.add_argument("--workers", type=int, default=4, help="parallel files for --compute_missing")
    p.add_argument("--no_mean_file", action="store_true", help=f"don't write {MEAN_FILE}")
    p.add_argument("--delete_nc", action="store_true",
                   help=f"--case only: delete {NC_FILE} after {RESULT_FILE} and {MEAN_FILE} are written")
    args = p.parse_args()

    if args.is_done:
        sys.exit(0 if load_current_result(os.path.abspath(args.is_done)) is not None else 1)

    if args.case:
        code, msg = process_and_record(args.case, write_mean=not args.no_mean_file, delete_nc=args.delete_nc)
        print(msg, flush=True)
        sys.exit(code)

    dirs = list_case_dirs(args.run_dir, args.case_list)
    if not dirs:
        sys.exit(f"No case folders found (run_dir={args.run_dir!r}, case_list={args.case_list!r})")

    if args.write_pending:
        write_pending(dirs, args.write_pending, args.dx_min, args.dx_max)
        return
    if args.compute_missing:
        compute_missing(dirs, max(1, args.workers), write_mean=not args.no_mean_file)
    collect(dirs, args.out)


if __name__ == "__main__":
    main()
