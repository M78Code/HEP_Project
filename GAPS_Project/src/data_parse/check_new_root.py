
import numpy as np
import uproot
from pathlib import Path

ROOT_DIR = Path("/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/dataset/tar_root")
C_MM_PER_NS = 299.792458
N_PER_FILE = 2000

def vec3_array(raw):
    if len(raw) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([
        np.asarray(raw["fX"], dtype=np.float32),
        np.asarray(raw["fY"], dtype=np.float32),
        np.asarray(raw["fZ"], dtype=np.float32),
    ], axis=1)

def get_tree(f, name):
    keys = [k for k in f.keys() if k.startswith(name + ";")]
    return f[sorted(keys)[-1]] if keys else f[name]

def summary(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"{name}: no finite values")
        return
    print(
        f"{name}: n={len(arr):,}, "
        f"min={arr.min():.4g}, "
        f"p1={np.percentile(arr,1):.4g}, "
        f"median={np.median(arr):.4g}, "
        f"p99={np.percentile(arr,99):.4g}, "
        f"max={arr.max():.4g}, "
        f"mean={arr.mean():.4g}"
    )

def analyze_particle(particle):
    files = sorted((ROOT_DIR / particle).glob("*.root"))
    print("\n" + "=" * 90)
    print(f"{particle}: {len(files)} files, first {N_PER_FILE} events/file")
    print("=" * 90)

    tof_time_vals = []
    si_time_vals = []
    dt_list = []
    beta_est_list = []
    beta_mc_list = []
    beta_rec_list = []

    no_outer = 0
    no_inner = 0
    no_time = 0
    neg_dt = 0
    total_events = 0

    for fp in files:
        print("reading", fp.name)
        with uproot.open(fp) as f:
            mc = get_tree(f, "TreeMc")
            rec = get_tree(f, "TreeRec")

            n = min(N_PER_FILE, mc.num_entries)

            beta_mc = mc["Mc/CEventBase/primaryBetaGenerated_"].array(entry_stop=n)
            vids = rec["Rec/hitseries_/hitseries_.volume_id_"].array(entry_stop=n)
            times = rec["Rec/hitseries_/hitseries_.hit_time_"].array(entry_stop=n)
            pos = rec["Rec/hitseries_/hitseries_.hit_position_"].array(entry_stop=n)

            rec_beta_raw = rec["Rec/primaryBeta_/primaryBeta_.second"].array(entry_stop=n)

        total_events += len(vids)

        for i in range(len(vids)):
            v = np.asarray(vids[i], dtype=np.int64)
            t = np.asarray(times[i], dtype=np.float64)
            p = vec3_array(pos[i])

            if len(v) == 0:
                continue

            system = v // 100000000
            is_tof = system == 1
            is_si = system == 2

            if is_tof.any():
                tof_time_vals.extend(t[is_tof][np.isfinite(t[is_tof])].tolist())
            if is_si.any():
                si_time_vals.extend(t[is_si][np.isfinite(t[is_si])].tolist())

            tof_subsystem = (v // 10000000) % 10
            is_outer = is_tof & (tof_subsystem == 0)
            is_inner = is_tof & (tof_subsystem == 1)

            if not is_outer.any():
                no_outer += 1
                continue
            if not is_inner.any():
                no_inner += 1
                continue

            outer_t = t[is_outer]
            inner_t = t[is_inner]
            outer_p = p[is_outer]
            inner_p = p[is_inner]

            outer_valid = np.isfinite(outer_t)
            inner_valid = np.isfinite(inner_t)

            if not outer_valid.any() or not inner_valid.any():
                no_time += 1
                continue

            outer_idx = np.where(outer_valid)[0][np.argmin(outer_t[outer_valid])]
            inner_idx = np.where(inner_valid)[0][np.argmin(inner_t[inner_valid])]

            t_outer = float(outer_t[outer_idx])
            t_inner = float(inner_t[inner_idx])
            dt = t_inner - t_outer

            if dt <= 0:
                neg_dt += 1
                continue

            path_mm = float(np.linalg.norm(inner_p[inner_idx] - outer_p[outer_idx]))
            beta_est = path_mm / (C_MM_PER_NS * dt)

            dt_list.append(dt)
            beta_est_list.append(beta_est)
            beta_mc_list.append(float(beta_mc[i]))

            if len(rec_beta_raw[i]) > 0:
                rb = np.asarray(rec_beta_raw[i], dtype=np.float64)
                rb = rb[np.isfinite(rb)]
                beta_rec_list.append(float(rb[0]) if len(rb) else np.nan)
            else:
                beta_rec_list.append(np.nan)

    print("\n--- hit_time validity ---")
    summary("TOF hit_time", tof_time_vals)
    summary("Si  hit_time", si_time_vals)

    print("\n--- TOF pair availability ---")
    print("events analyzed:", total_events)
    print("no outer:", no_outer)
    print("no inner:", no_inner)
    print("no valid outer/inner time:", no_time)
    print("negative dt skipped:", neg_dt)
    print("valid dt events:", len(dt_list))

    print("\n--- dt and beta check ---")
    summary("dt = inner_t - outer_t [ns]", dt_list)
    summary("beta_est = path/(c*dt)", beta_est_list)
    summary("beta_mc", beta_mc_list)

    be = np.asarray(beta_est_list)
    bm = np.asarray(beta_mc_list)
    mask = np.isfinite(be) & np.isfinite(bm) & (be > 0) & (be < 2)

    if mask.sum() > 10:
        print(f"corr(beta_est, beta_mc): {np.corrcoef(be[mask], bm[mask])[0,1]:.4f}")
        diff = be[mask] - bm[mask]
        print(f"beta_est - beta_mc: median={np.median(diff):.4g}, mean={diff.mean():.4g}, std={diff.std():.4g}")

    br = np.asarray(beta_rec_list)
    mask2 = np.isfinite(be) & np.isfinite(br) & (be > 0) & (be < 2)

    if mask2.sum() > 10:
        print(f"corr(beta_est, beta_rec): {np.corrcoef(be[mask2], br[mask2])[0,1]:.4f}")
        diff2 = be[mask2] - br[mask2]
        print(f"beta_est - beta_rec: median={np.median(diff2):.4g}, mean={diff2.mean():.4g}, std={diff2.std():.4g}")
    else:
        print("beta_rec comparison: not enough valid values")


if __name__ == "__main__":
    for particle in ["antiD", "antiP"]:
        analyze_particle(particle)