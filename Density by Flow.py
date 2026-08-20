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
# FILE PATH
# ======================================
file_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Flow\Raw_Flow_Density_5step.csv"

# ======================================
# GATE 14 (MID GATE) DETECTOR LINE COORDINATES
# These are the two physical edge points of the Gate 14 detector line,
# NOT a point every vehicle is guaranteed to touch -- a vehicle "crosses
# Gate 14" if its path crosses the LINE SEGMENT between these two points.
# ======================================
GATE14_POINT_A = (665944.41, 1517656.45)
GATE14_POINT_B = (665947.58, 1517644.27)

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

TRAJECTORY_START_COL_INDEX = 8   # trajectory data starts at column index 8
POINT_BLOCK_SIZE = 6             # each trajectory point = x, y, speed, tan_acc, lat_acc, time

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
# KEEP VALID TRIPS (Gate 16/14 -> Gate 15)
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

# Each vehicle uses its OWN measured distance (more accurate than one fixed average)
traj["Distance (m)"] = traj[dist_col]

# ======================================
# TRAVEL TIME
# ======================================
traj["Travel Time (s)"] = traj[exit_time_col] - traj[entry_time_col]
traj = traj[(traj["Travel Time (s)"] > 0) & (traj["Travel Time (s)"] < 300)].reset_index(drop=True)

# Plain numpy arrays -- much faster than repeated pandas filtering inside the loop
entry_arr = traj[entry_time_col].to_numpy()
exit_arr  = traj[exit_time_col].to_numpy()
time_arr  = traj["Travel Time (s)"].to_numpy()
dist_arr  = traj["Distance (m)"].to_numpy()

# ======================================
# MAX TIME
# ======================================
max_time = int(max(df[entry_time_col].max(), df[exit_time_col].max()))
print("\nMax time:", max_time)

# ======================================
# GATE 14 LINE-CROSSING DETECTION
# (replaces the old "Entry Gate == Gate 14" check for flow_mid,
#  which missed vehicles that entered at Gate 16 but physically
#  passed through the Gate 14 location on their way to Gate 15)
# ======================================

def ccw(p1, p2, p3):
    """Standard counter-clockwise orientation test, used for segment intersection."""
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

def segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 crosses segment p3-p4."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def extract_trajectory_points(raw_row):
    """
    Pulls (x, y, time) triples out of a vehicle's trajectory row (already a
    plain numeric numpy array). Skips any incomplete/empty trailing blocks.
    """
    n_points = len(raw_row) // POINT_BLOCK_SIZE
    points = []
    for i in range(n_points):
        x = raw_row[i * POINT_BLOCK_SIZE + 0]
        y = raw_row[i * POINT_BLOCK_SIZE + 1]
        t = raw_row[i * POINT_BLOCK_SIZE + 5]
        if not np.isnan(x) and not np.isnan(y) and not np.isnan(t):
            points.append((x, y, t))
    return points

def find_gate14_crossing_time(points, point_a, point_b):
    """
    Walks through consecutive trajectory points for one vehicle and checks whether
    the short path segment between them crosses the Gate 14 detector line (A-B).
    Returns the interpolated crossing time, or None if the vehicle never crosses it.
    Only the FIRST crossing is returned (a vehicle should only cross the gate once).
    """
    for i in range(len(points) - 1):
        p1 = (points[i][0], points[i][1])
        p2 = (points[i + 1][0], points[i + 1][1])
        if segments_intersect(p1, p2, point_a, point_b):
            # Interpolate crossing time as the midpoint between the two bracketing timestamps
            crossing_time = (points[i][2] + points[i + 1][2]) / 2
            return crossing_time
    return None

print("\nScanning all vehicle trajectories for Gate 14 line crossings...")

# Convert the whole trajectory block to numeric ONCE, up front, as a plain
# numpy array. Iterating with df.iterrows() over a dataframe with 2000+
# mixed-dtype columns is very slow -- it rebuilds a full object Series on
# every row. Working from a numeric numpy array instead is dramatically
# faster for a file this wide.
trajectory_block = df.iloc[:, TRAJECTORY_START_COL_INDEX:].apply(
    pd.to_numeric, errors="coerce"
).to_numpy()

gate14_crossing_times = []
n_rows = trajectory_block.shape[0]

for i in range(n_rows):
    pts = extract_trajectory_points(trajectory_block[i])
    if len(pts) < 2:
        continue
    crossing_time = find_gate14_crossing_time(pts, GATE14_POINT_A, GATE14_POINT_B)
    if crossing_time is not None:
        gate14_crossing_times.append(crossing_time)
    if (i + 1) % 500 == 0:
        print(f"  ...scanned {i + 1}/{n_rows} rows")
        sys.stdout.flush()

gate14_crossings = pd.Series(gate14_crossing_times, name="Gate14_Crossing_Time")
print(f"Total vehicles detected crossing the Gate 14 line: {len(gate14_crossings)}")
print(f"(For comparison, vehicles labeled Entry Gate == 'Gate 14': {(df[entry_gate_col]=='Gate 14').sum()})")

# ======================================
# IMPROVED SPACE MEAN SPEED -- PROPORTIONAL TIME-SLICING
#
# This is Edie's generalized definition of space mean speed: total distance
# covered by all vehicles PRESENT in a space-time window, divided by the
# total vehicle-time spent in that window.
#
# For every trip, work out how much of THIS interval [t0, t1) overlaps the
# vehicle's own [entry, exit) window. Credit the vehicle only for the slice
# of distance/time that actually happened during this interval (assuming
# constant speed across its own trip). A trip that spans several intervals
# now contributes a fair share to EVERY interval it was actually on the
# road for -- not just the one interval it happened to exit in, which was
# the flaw in the original exit-time-only method.
# ======================================
def compute_improved_sms(t0, t1):
    overlap_start = np.maximum(entry_arr, t0)
    overlap_end = np.minimum(exit_arr, t1)
    overlap = overlap_end - overlap_start          # seconds this vehicle was in the window
    mask = overlap > 0                              # only vehicles actually present

    if not mask.any():
        return np.nan, 0

    frac = overlap[mask] / time_arr[mask]             # fraction of the FULL trip that fell in this window
    distance_in_window = dist_arr[mask] * frac         # that same fraction of the trip's distance

    total_distance = distance_in_window.sum()
    total_time = overlap[mask].sum()
    n = int(mask.sum())

    sms = total_distance / total_time if total_time > 0 else np.nan
    return sms, n


# ======================================
# INTERVAL SETTINGS
# ======================================
interval = 30

print(f"\nBuilding {interval}-second flow/speed table (improved SMS method)...")
sys.stdout.flush()

results = []
start = 0

# ======================================
# MAIN LOOP
# ======================================
while start <= max_time:

    end = start + interval

    # -------- FLOW IN (Gate 16) -- unchanged, uses Entry Gate label ----------
    flow_in = df[
        (df[entry_gate_col] == "Gate 16") &
        (df[entry_time_col] >= start) &
        (df[entry_time_col] < end)
    ].shape[0]

    # -------- FLOW MID (Gate 14) -- uses actual line-crossing time ----------
    flow_mid = gate14_crossings[
        (gate14_crossings >= start) &
        (gate14_crossings < end)
    ].shape[0]

    # -------- FLOW OUT (Gate 15) -- unchanged, uses Exit Gate label ----------
    flow_out = df[
        (df[exit_gate_col] == "Gate 15") &
        (df[exit_time_col] >= start) &
        (df[exit_time_col] < end)
    ].shape[0]

    avg_flow = (flow_in + flow_mid + flow_out) / 3
    avg_flow_hour = (avg_flow / interval) * 3600

    # ======================================
    # SPACE MEAN SPEED -- improved proportional time-slicing method
    # ======================================
    sms, n_vehicles = compute_improved_sms(start, end)

    results.append([
        f"{start}-{end}",
        flow_in,
        flow_mid,
        flow_out,
        avg_flow,
        avg_flow_hour,
        sms,
        n_vehicles
    ])

    start += interval

# ======================================
# CREATE TABLE
# ======================================
result = pd.DataFrame(results, columns=[
    "Time (s)",
    "Flow (In) - Gate 16",
    "Flow (Mid) - Gate 14 [line-crossing]",
    "Flow (Out) - Gate 15",
    "Average Flow",
    "Average Flow (veh/hour)",
    "Space Mean Speed (m/s)",
    "Vehicle Count (Improved SMS)"
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

base_name = "05_12_25.Density_Flow_30sec_ImprovedSMS.xlsx"
output_file = os.path.join(output_dir, base_name)

counter = 1
while os.path.exists(output_file):
    output_file = os.path.join(output_dir, f"05_12_25.Density_Flow_30sec_ImprovedSMS_{counter}.xlsx")
    counter += 1

print(f"\nAttempting to save to: {output_file}")
sys.stdout.flush()

try:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="30sec_Flow_Speed", index=False)
        dist_summary.to_excel(writer, sheet_name="Gate_Distance_Summary")
        gate14_crossings.to_frame().to_excel(writer, sheet_name="Gate14_Crossings_Raw", index=False)
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