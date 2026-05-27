import pandas as pd
import numpy as np

# ==========================================================
# STEP 1 — PIVOT: Wide CSV  ->  Long Excel
# Filters each vehicle's frames to [Entry Time, min(Exit Time, last frame time)]
# Vehicles whose recording ends before the exit gate are flagged Truncated=True
# ==========================================================

input_file  = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\T.Raw_data.csv"
output_file = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\T.cross_test2_pivoted.xlsx"

BLOCK_SIZE         = 6
META_COLS          = 8
MAX_ROWS_PER_SHEET = 1_000_000

# ----------------------------------------------------------
print("Loading raw CSV ...")
df_raw = pd.read_csv(input_file, low_memory=False)
df_raw.columns = df_raw.columns.str.strip()
print(f"  {len(df_raw):,} vehicles  |  {df_raw.shape[1]:,} columns")
print(f"  First 8 columns: {df_raw.columns[:8].tolist()}")

# ----------------------------------------------------------
print("Pivoting wide -> long (filtering to entry-exit window) ...")

all_rows = []
n_trunc  = 0

for rec_idx, row in df_raw.iterrows():
    tid        = row["Track ID"]
    vtype      = row["Type"].strip().strip('"') if isinstance(row["Type"], str) else row["Type"]
    entry_gate = str(row["Entry Gate"]).strip().strip('"')
    exit_gate  = str(row["Exit Gate"]).strip().strip('"')
    entry_t    = float(row["Entry Time [s]"])
    exit_t     = float(row["Exit Time [s]"])
    trav_dist  = float(row["Traveled Dist. [m]"])
    avg_speed  = float(row["Avg. Speed [km/h]"])

    frame_vals = row.iloc[META_COLS:].dropna().values
    n_frames   = len(frame_vals) // BLOCK_SIZE
    if n_frames < 2:
        continue

    # Actual last frame time (absolute clock)
    try:
        t_last = float(frame_vals[(n_frames - 1) * BLOCK_SIZE + 5])
    except (ValueError, TypeError):
        continue

    exit_t_eff   = min(exit_t, t_last)
    is_truncated = exit_t > t_last
    if is_truncated:
        n_trunc += 1

    for i in range(0, n_frames * BLOCK_SIZE, BLOCK_SIZE):
        try:
            x     = float(frame_vals[i])
            y     = float(frame_vals[i + 1])
            speed = float(frame_vals[i + 2])
            tan_a = float(frame_vals[i + 3])
            lat_a = float(frame_vals[i + 4])
            t     = float(frame_vals[i + 5])
        except (ValueError, TypeError):
            continue

        if t < entry_t or t > exit_t_eff:
            continue

        all_rows.append({
            "Track ID"          : tid,
            "Type"              : vtype,
            "Entry Gate"        : entry_gate,
            "Entry Time [s]"    : entry_t,
            "Exit Gate"         : exit_gate,
            "Exit Time [s]"     : exit_t,
            "Traveled Dist. [m]": trav_dist,
            "Avg. Speed [km/h]" : avg_speed,
            "Truncated"         : is_truncated,
            "x [m]"             : x,
            "y [m]"             : y,
            "Speed [km/h]"      : speed,
            "Tan. Acc. [m/s2]"  : tan_a,
            "Lat. Acc. [m/s2]"  : lat_a,
            "Time [s]"          : t,
        })

df_long = pd.DataFrame(all_rows)
print(f"  Frame rows after filter : {len(df_long):,}")
print(f"  Truncated vehicles      : {n_trunc:,}")

df_long = (df_long
           .sort_values(["Track ID", "Time [s]"])
           .drop_duplicates(subset=["Track ID", "Time [s]"], keep="first")
           .reset_index(drop=True))
print(f"  After dedup             : {len(df_long):,} rows")

print("Saving pivoted Excel ...")
n_sheets = max(1, int(np.ceil(len(df_long) / MAX_ROWS_PER_SHEET)))
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    for s in range(n_sheets):
        chunk      = df_long.iloc[s * MAX_ROWS_PER_SHEET : (s + 1) * MAX_ROWS_PER_SHEET]
        sheet_name = f"Sheet{s+1}" if n_sheets > 1 else "Sheet1"
        chunk.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  Sheet '{sheet_name}': {len(chunk):,} rows")

print(f"\n  Pivot complete -> {output_file}")
print(f"   Vehicles  : {df_long['Track ID'].nunique():,}")
print(f"   Truncated : {n_trunc:,}  (cumulative distance corrected in Step 2)")