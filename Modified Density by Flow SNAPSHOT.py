import pandas as pd
import numpy as np
import os
import csv
import sys
import traceback

# Raise the csv module's field size limit -- a stray, unbalanced " character
# in the export makes the parser treat everything after it as one giant
# unterminated field, which trips the default 131072-byte limit.
csv.field_size_limit(10_000_000)

# ======================================
# FILE PATHS
# ======================================
file_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Flow\Raw_Flow_Density_5step.csv"
traj_1s_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\Trajectory\05_12_25_1315_1345.trajectory_1s_final.xlsx"

# ======================================
# READ CSV
# The file is genuinely comma-delimited, but the trajectory portion of
# each row is one large quoted field (quotechar='"') protecting internal
# commas -- and one stray/unbalanced quote somewhere in the export was
# desyncing pandas' own quote handling, which is what caused the earlier
# "Expected N fields, saw ~2N" and "field larger than field limit" errors.
# Parsing manually with Python's csv module sidesteps both problems: it
# respects quoting correctly and doesn't require every row to have the
# same number of fields (rows differ in length because vehicles have
# different numbers of trajectory points).
# ======================================
with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    rows = [r for r in reader if r]  # drop fully blank lines

header = [h.strip() for h in rows[0]]
data_rows = rows[1:]

max_len = max(len(r) for r in data_rows)
if len(header) < max_len:
    header = header + [f"traj_col_{i}" for i in range(len(header), max_len)]

# Pad short rows (fewer trajectory points) so the table is rectangular
data_rows = [r + [""] * (max_len - len(r)) for r in data_rows]

df = pd.DataFrame(data_rows, columns=header[:max_len])
df.columns = df.columns.str.strip()
df = df.replace("", np.nan)  # csv.reader gives "" for blanks/padding, not NaN

print("Detected Columns (first 10):")
print(df.columns[:10].tolist())

# ======================================
# DEFINE COLUMNS
# ======================================
entry_gate_col = "Entry Gate"
exit_gate_col  = "Exit Gate"
entry_time_col = "Entry Time [s]"
exit_time_col  = "Exit Time [s]"
type_col       = "Type"
dist_col       = "Traveled Dist. [m]"

# ======================================
# CLEAN DATA
# ======================================
df = df[df[type_col].notna()].copy()

df[entry_time_col] = pd.to_numeric(df[entry_time_col], errors="coerce")
df[exit_time_col]  = pd.to_numeric(df[exit_time_col], errors="coerce")
df[dist_col]       = pd.to_numeric(df[dist_col], errors="coerce")

df = df.dropna(subset=[entry_time_col, exit_time_col])

df[entry_gate_col] = df[entry_gate_col].astype(str).str.replace('"', '').str.strip()
df[exit_gate_col]  = df[exit_gate_col].astype(str).str.replace('"', '').str.strip()

# Keep a stable row index to use as a vehicle identifier when extracting trajectories
df = df.reset_index(drop=True)

# ======================================
# KEEP VALID TRAJECTORIES (Gate 16/14 -> Gate 15) -- kept for distance
# summary / reference only; not used for speed (boundary-snapshot method
# below uses the separate 1-second trajectory file instead)
# ======================================
traj = df[
    (df[exit_gate_col] == "Gate 15") &
    (df[entry_gate_col].isin(["Gate 16", "Gate 14"]))
].copy()

# ======================================
# AUTOMATICALLY CALCULATE AVERAGE SECTION DISTANCES FROM DATA
# ======================================
dist_summary = traj.groupby(entry_gate_col)[dist_col].agg(["mean", "std", "count"])
print("\n📏 Average measured distance per section (from drone trajectory data):")
print(dist_summary)

traj["Distance (m)"] = traj[dist_col]
traj["Travel Time (s)"] = traj[exit_time_col] - traj[entry_time_col]
traj = traj[(traj["Travel Time (s)"] > 0) & (traj["Travel Time (s)"] < 300)]

# ======================================
# MAX TIME
# ======================================
max_time = int(max(df[entry_time_col].max(), df[exit_time_col].max()))
print("\nMax time:", max_time)

# ======================================
# LOAD 1-SECOND TRAJECTORY DATA (for boundary-snapshot speed)
# ======================================
print("\nLoading 1-second trajectory file for boundary-snapshot speed estimation...")
traj_1s = pd.read_excel(traj_1s_path)
traj_1s.columns = traj_1s.columns.str.strip()
traj_1s = traj_1s.dropna(subset=["Time [s]", "Track ID", "Speed [km/h]"])
print(f"Loaded {len(traj_1s)} trajectory rows covering {traj_1s['Track ID'].nunique()} vehicles.")

# ======================================
# BOUNDARY-SNAPSHOT SPEED FUNCTION
#
# For an interval [t0, t1) (e.g. 0-30s), this no longer slices into
# sub-windows. Instead it takes exactly two snapshots -- one at t0 and one
# at t1 (e.g. at 0s and at 30s) -- averages every vehicle's instantaneous
# speed found at EACH of those two instants separately, and then takes the
# PLAIN (unweighted) average of those two snapshot averages to get the
# final speed for the whole interval.
# ======================================
SNAPSHOT_TOL = 0.5  # seconds, tolerance for matching a row's timestamp to the snapshot instant


def compute_boundary_snapshot_speed(t0, t1, traj_1s, tol=SNAPSHOT_TOL):
    snap_t0 = traj_1s[np.isclose(traj_1s["Time [s]"], t0, atol=tol)]
    snap_t1 = traj_1s[np.isclose(traj_1s["Time [s]"], t1, atol=tol)]

    avg_speed_t0 = snap_t0["Speed [km/h]"].mean() if len(snap_t0) > 0 else np.nan
    avg_speed_t1 = snap_t1["Speed [km/h]"].mean() if len(snap_t1) > 0 else np.nan

    n_t0 = int(snap_t0["Track ID"].nunique()) if len(snap_t0) > 0 else 0
    n_t1 = int(snap_t1["Track ID"].nunique()) if len(snap_t1) > 0 else 0

    valid = [s for s in (avg_speed_t0, avg_speed_t1) if not np.isnan(s)]

    # Plain average of the t0-snapshot average and the t1-snapshot average
    final_kmh = np.mean(valid) if len(valid) > 0 else np.nan

    return final_kmh, avg_speed_t0, avg_speed_t1, n_t0, n_t1


# ======================================
# INTERVAL SETTINGS
# ======================================
interval = 30  # matches the "30_sec" folder this run's flow file came from

print(f"\nBuilding {interval}-second flow/speed table using boundary-snapshot speed (t0 & t1 only)...")
sys.stdout.flush()

results = []
speed_detail_rows = []
start = 0

# ======================================
# MAIN LOOP
# ======================================
while start <= max_time:

    end = start + interval

    # -------- FLOW IN (Gate 16) -- uses Entry Gate label ----------
    flow_in = df[
        (df[entry_gate_col] == "Gate 16") &
        (df[entry_time_col] >= start) &
        (df[entry_time_col] < end)
    ].shape[0]

    # -------- FLOW OUT (Gate 15) -- uses Exit Gate label ----------
    flow_out = df[
        (df[exit_gate_col] == "Gate 15") &
        (df[exit_time_col] >= start) &
        (df[exit_time_col] < end)
    ].shape[0]

    # Gate 14 (mid-gate) removed -- Average Flow is now Entry/Exit only.
    avg_flow = (flow_in + flow_out) / 2
    avg_flow_hour = (avg_flow / interval) * 3600

    # ======================================
    # SPEED -- BOUNDARY-SNAPSHOT METHOD (replaces M4-15 sub-windows)
    # ======================================
    final_kmh, speed_t0, speed_t1, n_t0, n_t1 = compute_boundary_snapshot_speed(start, end, traj_1s)
    final_ms = final_kmh / 3.6 if not np.isnan(final_kmh) else np.nan

    results.append([
        f"{start}-{end}",
        flow_in,
        flow_out,
        avg_flow,
        avg_flow_hour,
        final_ms,
        final_kmh
    ])

    # -------- detail rows for the separate speed-detail sheet (one row per boundary snapshot) --------
    speed_detail_rows.append([f"{start}-{end}", start, speed_t0, n_t0])
    speed_detail_rows.append([f"{start}-{end}", end, speed_t1, n_t1])

    start += interval

# ======================================
# CREATE TABLES
# ======================================
result = pd.DataFrame(results, columns=[
    "Time (s)",
    "Flow (In) - Gate 16",
    "Flow (Out) - Gate 15",
    "Average Flow",
    "Average Flow (veh/hour)",
    "Space Mean Speed - Boundary Snapshot (m/s)",
    "Space Mean Speed - Boundary Snapshot (km/h)"
])

speed_detail_df = pd.DataFrame(speed_detail_rows, columns=[
    "Interval (s)",
    "Snapshot Time [s]",
    "Snapshot Avg Speed [km/h]",
    "Vehicle Count at Snapshot"
])

# ======================================
# SAFE SAVE
# ======================================
# Save into the parent folder of the input CSV's directory -- e.g. if the
# CSV lives in "...\05_12_25_1315_1345\30_sec\Flow\", the output goes into
# "...\05_12_25_1315_1345\30_sec\" (one level up from the Flow subfolder).
output_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

if not os.path.isdir(output_dir):
    print(f"⚠️  Parent folder not found at: {output_dir}")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Falling back to script folder: {output_dir}")

base_name = "05_12_25.Density_Flow_30sec_BoundarySnapshot.xlsx"
output_file = os.path.join(output_dir, base_name)

counter = 1
while os.path.exists(output_file):
    output_file = os.path.join(output_dir, f"05_12_25.Density_Flow_30sec_BoundarySnapshot_{counter}.xlsx")
    counter += 1

print(f"\nAttempting to save to: {output_file}")
sys.stdout.flush()

try:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="30sec_Flow_Speed", index=False)
        speed_detail_df.to_excel(writer, sheet_name="Speed_Detail", index=False)
        dist_summary.to_excel(writer, sheet_name="Gate_Distance_Summary")
except Exception:
    print("\n❌ Excel save FAILED. Full error below:")
    traceback.print_exc()
    raise

if os.path.exists(output_file):
    print("\n✅ Excel successfully created!")
    print("Saved at:", output_file)
else:
    print("\n❌ Write finished with no exception, but the file is missing from disk.")
    print("Expected at:", output_file)