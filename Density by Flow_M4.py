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
# KEEP VALID TRAJECTORIES (Gate 16/14 -> Gate 15) -- kept for distance
# summary / reference only; no longer used for speed (M4-15 replaces it)
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
# LOAD 1-SECOND TRAJECTORY DATA (for M4-15 speed snapshots)
# ======================================
print("\nLoading 1-second trajectory file for M4-15 speed estimation...")
traj_1s = pd.read_excel(traj_1s_path)
traj_1s.columns = traj_1s.columns.str.strip()
traj_1s = traj_1s.dropna(subset=["Time [s]", "Track ID", "Speed [km/h]"])
print(f"Loaded {len(traj_1s)} trajectory rows covering {traj_1s['Track ID'].nunique()} vehicles.")

# ======================================
# M4-15 SNAPSHOT SPEED FUNCTION
#
# For a given interval (t0 to t1), this cuts it into 15-second sub-windows.
# For each sub-window it takes a single "snapshot" at the sub-window's
# midpoint (e.g. 0-15s -> snapshot at 7.5s, 15-30s -> snapshot at 22.5s),
# averages every vehicle's instantaneous speed found at that snapshot
# instant, and then takes the PLAIN (unweighted) average of those
# sub-window snapshot averages to get one M4-15 speed for the whole
# interval -- exactly as specified: average the 0-15s snapshot average
# and the 15-30s snapshot average together.
# ======================================
M4_SUBWINDOW = 15  # seconds -> snapshot taken at the midpoint (7.5s into each sub-window)


def compute_m4_15(t0, t1, traj_1s, sub_window=M4_SUBWINDOW):
    n_sub = int(round((t1 - t0) / sub_window))

    sub_avg_speeds = []   # one average speed per sub-window snapshot
    sub_veh_counts = []   # how many vehicles were in each snapshot

    for i in range(n_sub):
        s0 = t0 + i * sub_window
        ts = s0 + sub_window / 2  # snapshot instant (midpoint of the sub-window)

        snap = traj_1s[np.isclose(traj_1s["Time [s]"], ts, atol=0.5)]

        if len(snap) > 0:
            sub_avg_speeds.append(snap["Speed [km/h]"].mean())
            sub_veh_counts.append(snap["Track ID"].nunique())
        else:
            sub_avg_speeds.append(np.nan)
            sub_veh_counts.append(0)

    valid_speeds = [s for s in sub_avg_speeds if not np.isnan(s)]

    # Plain average of the sub-window snapshot averages (not vehicle-count weighted)
    m4_15_kmh = np.mean(valid_speeds) if len(valid_speeds) > 0 else np.nan

    return m4_15_kmh, sub_avg_speeds, sub_veh_counts


# ======================================
# INTERVAL SETTINGS
# ======================================
interval = 30  # matches the "30_sec" folder this run's flow file came from

print(f"\nBuilding {interval}-second flow/speed table using M4-15 snapshot speed...")
sys.stdout.flush()

results = []
m4_detail_rows = []
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
    # SPEED -- M4-15 SNAPSHOT METHOD (replaces the old distance/time SMS)
    # ======================================
    m4_15_kmh, sub_speeds, sub_counts = compute_m4_15(start, end, traj_1s, M4_SUBWINDOW)
    m4_15_ms = m4_15_kmh / 3.6 if not np.isnan(m4_15_kmh) else np.nan

    results.append([
        f"{start}-{end}",
        flow_in,
        flow_mid,
        flow_out,
        avg_flow,
        avg_flow_hour,
        m4_15_ms,
        m4_15_kmh
    ])

    # -------- detail rows for the separate M4-15 sheet --------
    n_sub = int(round(interval / M4_SUBWINDOW))
    for i in range(n_sub):
        s0 = start + i * M4_SUBWINDOW
        s1 = s0 + M4_SUBWINDOW
        ts = s0 + M4_SUBWINDOW / 2
        m4_detail_rows.append([
            f"{start}-{end}",
            f"{s0}-{s1}",
            ts,
            sub_speeds[i] if i < len(sub_speeds) else np.nan,
            sub_counts[i] if i < len(sub_counts) else 0
        ])

    start += interval

# ======================================
# CREATE TABLES
# ======================================
result = pd.DataFrame(results, columns=[
    "Time (s)",
    "Flow (In) - Gate 16",
    "Flow (Mid) - Gate 14 [line-crossing]",
    "Flow (Out) - Gate 15",
    "Average Flow",
    "Average Flow (veh/hour)",
    "Space Mean Speed - M4-15 (m/s)",
    "Space Mean Speed - M4-15 (km/h)"
])

m4_detail_df = pd.DataFrame(m4_detail_rows, columns=[
    "Interval (s)",
    "Sub-window (s)",
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

base_name = "05_12_25.Density_Flow_30sec_M4_15.xlsx"
output_file = os.path.join(output_dir, base_name)

counter = 1
while os.path.exists(output_file):
    output_file = os.path.join(output_dir, f"05_12_25.Density_Flow_30sec_M4_15_{counter}.xlsx")
    counter += 1

print(f"\nAttempting to save to: {output_file}")
sys.stdout.flush()

try:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="30sec_Flow_Speed", index=False)
        m4_detail_df.to_excel(writer, sheet_name="M4_15_Speed_Detail", index=False)
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