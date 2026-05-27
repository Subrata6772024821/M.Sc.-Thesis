import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# STEP 7 — SATURATION FLOW & DEGREE OF SATURATION
#
# Inputs:
#   1. GreenTime_{DATASET_ID}.xlsx   — your green time template
#   2. S.trajectory_1s_final.xlsx   — trajectory data (Step 2 output)
#
# Outputs (saved to same folder):
#   S.saturation_dos.xlsx  — full results with:
#       Sheet 1: Per-phase results (flow, headways, saturation flow)
#       Sheet 2: Dataset summary (q, s, capacity, DoS, LOS)
# ==========================================================

data_folder     = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH"
green_file      = str(Path(data_folder) / "GreenTime_S_Dec03_1315.xlsx")
# Output saved to the SAME folder as the green time template
output_file     = str(Path(green_file).parent /
                  (Path(green_file).stem.replace("GreenTime_", "")
                   + "_saturation_dos.xlsx"))

OBSERVATION_DURATION_S = 1800   # 30 minutes in seconds
STOP_LINE_POS          = 96.0   # metres — exit gate position
SKIP_VEHICLES          = 2      # skip first N vehicles per phase (startup delay)
MIN_HEADWAY_S          = 1.0    # discard headways shorter than this (same-time exits)
MAX_HEADWAY_S          = 10.0   # discard headways longer than this (gap in discharge)

# ----------------------------------------------------------
# LOAD GREEN TIME TEMPLATE
# ----------------------------------------------------------
print("Loading green time template ...")
df_green = pd.read_excel(green_file, sheet_name="Green Phases", header=4)
df_green.columns = df_green.columns.str.strip()

# Read metadata from template
dataset_id   = df_green["Dataset_ID"].iloc[0]
location     = df_green["Location"].iloc[0]
date_str     = str(df_green["Date"].iloc[0])
session_start= df_green["Session_Start"].iloc[0]
session_end  = df_green["Session_End"].iloc[0]
traj_file    = df_green["Trajectory_File"].iloc[0]

# Drop summary/empty rows (Phase_No must be a valid number)
df_green = df_green.dropna(subset=["Phase_No"]).copy()
df_green["Phase_No"] = df_green["Phase_No"].astype(int)
df_green["Green_Start_s"]    = pd.to_numeric(df_green["Green_Start_s"],    errors="coerce")
df_green["Green_End_s"]      = pd.to_numeric(df_green["Green_End_s"],      errors="coerce")
df_green["Green_Duration_s"] = pd.to_numeric(df_green["Green_Duration_s"], errors="coerce")
df_green = df_green.dropna(subset=["Green_Start_s", "Green_End_s"]).reset_index(drop=True)

print(f"  Dataset  : {dataset_id}")
print(f"  Location : {location}  |  Date : {date_str}")
print(f"  Session  : {session_start} – {session_end}")
print(f"  Phases   : {len(df_green)}  "
      f"({(df_green['Use_For_SatFlow']=='Yes').sum()} complete)")

# ----------------------------------------------------------
# LOAD TRAJECTORY DATA
# ----------------------------------------------------------
traj_path = str(Path(data_folder) / traj_file)
print(f"\nLoading trajectory data: {traj_file} ...")
df_traj = pd.read_excel(traj_path, sheet_name="Final_Data")
df_traj.columns = df_traj.columns.str.strip()

# Exclude reconstructed vehicles
if "Reconstructed" in df_traj.columns:
    df_traj = df_traj[df_traj["Reconstructed"] == False].copy()

df_traj = df_traj.sort_values(["Track ID", "Time [s]"]).reset_index(drop=True)
print(f"  {df_traj['Track ID'].nunique():,} vehicles  |  {len(df_traj):,} rows")

# ----------------------------------------------------------
# HELPER: find when each vehicle crosses the stop line
# (first time cumulative_distance >= STOP_LINE_POS * 0.95)
# ----------------------------------------------------------
crossing_threshold = STOP_LINE_POS * 0.95  # allow 5% tolerance

vehicle_crossings = {}   # track_id -> crossing time (seconds)
for track_id, grp in df_traj.groupby("Track ID"):
    grp = grp.sort_values("Time [s]")
    reached = grp[grp["cumulative_distance"] >= crossing_threshold]
    if len(reached) > 0:
        vehicle_crossings[track_id] = reached["Time [s]"].iloc[0]

print(f"  Vehicles that reached stop line: {len(vehicle_crossings):,}")

# ----------------------------------------------------------
# PER-PHASE ANALYSIS
# ----------------------------------------------------------
print("\nAnalysing green phases ...")

phase_results  = []   # detailed per-phase results
all_headways   = []   # all valid headways (for saturation flow)

for _, phase in df_green.iterrows():
    phase_no    = int(phase["Phase_No"])
    g_start     = float(phase["Green_Start_s"])
    g_end       = float(phase["Green_End_s"])
    g_dur       = float(phase["Green_Duration_s"])
    use_for_sat = str(phase["Use_For_SatFlow"]).strip() == "Yes"
    is_partial  = str(phase["Partial_Phase"]).strip() == "Yes"

    # Vehicles that crossed stop line DURING this green phase
    phase_crossings = {
        tid: t for tid, t in vehicle_crossings.items()
        if g_start <= t <= g_end
    }

    crossing_times = sorted(phase_crossings.values())
    n_vehicles     = len(crossing_times)

    # Headways: skip first SKIP_VEHICLES (startup delay)
    headways = []
    if use_for_sat and n_vehicles > SKIP_VEHICLES + 1:
        discharge_times = crossing_times[SKIP_VEHICLES:]
        for i in range(1, len(discharge_times)):
            h = discharge_times[i] - discharge_times[i-1]
            if MIN_HEADWAY_S <= h <= MAX_HEADWAY_S:
                headways.append(h)
        all_headways.extend(headways)

    phase_results.append({
        "Phase No"          : phase_no,
        "Green Start [s]"   : g_start,
        "Green End [s]"     : g_end,
        "Green Duration [s]": g_dur,
        "Partial Phase"     : "Yes" if is_partial else "No",
        "Use For Sat Flow"  : "Yes" if use_for_sat else "No",
        "Vehicles Crossed"  : n_vehicles,
        "Valid Headways"    : len(headways),
        "Mean Headway [s]"  : round(np.mean(headways), 3) if headways else np.nan,
        "Min Headway [s]"   : round(np.min(headways),  3) if headways else np.nan,
        "Max Headway [s]"   : round(np.max(headways),  3) if headways else np.nan,
        "Phase Sat Flow"    : round(3600 / np.mean(headways), 1) if headways else np.nan,
    })

df_phase = pd.DataFrame(phase_results)

# ----------------------------------------------------------
# OVERALL SATURATION FLOW
# ----------------------------------------------------------
if len(all_headways) == 0:
    print("  WARNING: No valid headways found. Check crossing threshold or green phases.")
    s = np.nan
else:
    mean_h = np.mean(all_headways)
    s      = 3600 / mean_h
    print(f"  Total valid headways : {len(all_headways)}")
    print(f"  Mean headway         : {mean_h:.3f}s")
    print(f"  Saturation flow (s)  : {s:.1f} veh/hr")

# ----------------------------------------------------------
# ACTUAL FLOW (q)
# ----------------------------------------------------------
# Count ALL vehicles that crossed the stop line during ANY green phase
# (vehicles crossing during red are stragglers — not counted as served)
green_intervals = list(zip(
    df_green["Green_Start_s"].astype(float),
    df_green["Green_End_s"].astype(float)
))

def in_green(t):
    return any(gs <= t <= ge for gs, ge in green_intervals)

vehicles_served = sum(1 for t in vehicle_crossings.values() if in_green(t))
q = vehicles_served * (3600 / OBSERVATION_DURATION_S)  # convert to veh/hr

print(f"\n  Vehicles served (crossed during green) : {vehicles_served}")
print(f"  Actual flow q                          : {q:.1f} veh/hr")

# ----------------------------------------------------------
# DEGREE OF SATURATION
# ----------------------------------------------------------
total_green_s = df_green["Green_Duration_s"].sum()
green_ratio   = total_green_s / OBSERVATION_DURATION_S

capacity      = s * green_ratio if not np.isnan(s) else np.nan
dos           = q / capacity     if not np.isnan(capacity) and capacity > 0 else np.nan

# LOS classification
def get_los(d):
    if np.isnan(d):  return "N/A"
    if d < 0.60:     return "A-B (Free flow)"
    if d < 0.70:     return "C (Stable)"
    if d < 0.85:     return "D (Near stable)"
    if d < 1.00:     return "E (Near capacity)"
    return               "F (Oversaturated)"

los = get_los(dos)

print(f"\n  Total green time G     : {total_green_s}s ({total_green_s/60:.2f} min)")
print(f"  Green ratio (G/T)      : {green_ratio:.4f} ({green_ratio*100:.1f}%)")
print(f"  Saturation flow (s)    : {s:.1f} veh/hr" if not np.isnan(s) else "  Saturation flow: N/A")
print(f"  Capacity (s × G/T)     : {capacity:.1f} veh/hr" if not np.isnan(capacity) else "  Capacity: N/A")
print(f"  Actual flow (q)        : {q:.1f} veh/hr")
print(f"  Degree of Saturation   : {dos:.4f}" if not np.isnan(dos) else "  DoS: N/A")
print(f"  Level of Service       : {los}")

# ----------------------------------------------------------
# BUILD SUMMARY DATAFRAME
# ----------------------------------------------------------
summary = {
    "Parameter": [
        "Dataset ID",
        "Location",
        "Date",
        "Session",
        "Observation duration (s)",
        "Observation duration (min)",
        "────────────── SIGNAL ──────────────",
        "Total green time G (s)",
        "Total green time G (min)",
        "Green ratio G/T",
        "Total green phases",
        "Complete phases (used for s)",
        "Average green duration (s)",
        "Average cycle time (s)",
        "────────────── FLOW ──────────────",
        "Total vehicles in scene",
        "Vehicles that reached stop line",
        "Vehicles served during green",
        "Actual flow q (veh/hr)",
        "────────────── SATURATION FLOW ──────────────",
        "Valid headways used",
        "Mean discharge headway (s)",
        "Saturation flow s (veh/hr)",
        "────────────── DEGREE OF SATURATION ──────────────",
        "Capacity = s × G/T (veh/hr)",
        "Degree of Saturation (DoS = q/capacity)",
        "Level of Service (LOS)",
    ],
    "Value": [
        dataset_id,
        location,
        date_str,
        f"{session_start} – {session_end}",
        OBSERVATION_DURATION_S,
        round(OBSERVATION_DURATION_S/60, 1),
        "",
        total_green_s,
        round(total_green_s/60, 2),
        round(green_ratio, 4),
        len(df_green),
        int((df_green["Use_For_SatFlow"]=="Yes").sum()),
        round(df_green["Green_Duration_s"].mean(), 1),
        round(df_green["Green_Start_s"].diff().dropna().mean(), 1),
        "",
        df_traj["Track ID"].nunique(),
        len(vehicle_crossings),
        vehicles_served,
        round(q, 1),
        "",
        len(all_headways),
        round(mean_h, 3) if all_headways else "N/A",
        round(s, 1) if not np.isnan(s) else "N/A",
        "",
        round(capacity, 1) if not np.isnan(capacity) else "N/A",
        round(dos, 4)      if not np.isnan(dos)      else "N/A",
        los,
    ]
}
df_summary = pd.DataFrame(summary)

# ----------------------------------------------------------
# SAVE EXCEL
# ----------------------------------------------------------
print(f"\nSaving -> {output_file}")
# Replace NaN with empty string to avoid xlsxwriter crash
df_phase   = df_phase.fillna("")
df_summary = df_summary.fillna("")

with pd.ExcelWriter(output_file, engine="xlsxwriter",
                    engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
    df_summary.to_excel(writer, sheet_name="DoS Summary",    index=False)
    df_phase.to_excel(  writer, sheet_name="Per-Phase Detail", index=False)

    wb = writer.book

    # ── DoS Summary sheet ──
    ws1   = writer.sheets["DoS Summary"]
    hdr   = wb.add_format({"bold": True, "bg_color": "#D9E1F2",
                            "border": 1, "align": "center"})
    sec   = wb.add_format({"bold": True, "bg_color": "#1F3864",
                            "font_color": "#FFFFFF", "border": 1})
    val   = wb.add_format({"align": "center", "border": 1})
    hi    = wb.add_format({"bold": True, "bg_color": "#E2EFDA",
                            "border": 1, "align": "center", "font_size": 12})

    ws1.set_column(0, 0, 45)
    ws1.set_column(1, 1, 25)
    ws1.write(0, 0, "Parameter", hdr)
    ws1.write(0, 1, "Value",     hdr)

    for ri, row_data in df_summary.iterrows():
        param = str(row_data["Parameter"])
        value = row_data["Value"]
        excel_row = ri + 1
        if "──" in param:
            ws1.write(excel_row, 0, param.replace("─","").strip(), sec)
            ws1.write(excel_row, 1, "", sec)
        elif param in ["Degree of Saturation (DoS = q/capacity)",
                       "Level of Service (LOS)"]:
            ws1.write(excel_row, 0, param, hi)
            ws1.write(excel_row, 1, value, hi)
        else:
            ws1.write(excel_row, 0, param, val)
            ws1.write(excel_row, 1, value, val)

    # ── Per-Phase Detail sheet ──
    ws2 = writer.sheets["Per-Phase Detail"]
    for ci, col in enumerate(df_phase.columns):
        max_w = max(len(str(col)),
                    df_phase[col].astype(str).str.len().max()) + 2
        ws2.set_column(ci, ci, min(max_w, 22))
        ws2.write(0, ci, col, hdr)

    partial_fmt  = wb.add_format({"bg_color": "#FFF2CC",
                                   "border": 1, "align": "center"})
    complete_fmt = wb.add_format({"bg_color": "#E2EFDA",
                                   "border": 1, "align": "center"})
    for ri, row_data in df_phase.iterrows():
        fmt = partial_fmt if row_data["Partial Phase"] == "Yes" else complete_fmt
        for ci, val_cell in enumerate(row_data):
            ws2.write(ri + 1, ci, val_cell, fmt)

    ws2.freeze_panes(1, 0)

print("  Saved successfully.")
print(f"\n{'='*50}")
print(f"  RESULT SUMMARY")
print(f"{'='*50}")
print(f"  Dataset          : {dataset_id}")
print(f"  Actual flow q    : {q:.1f} veh/hr")
print(f"  Saturation flow s: {s:.1f} veh/hr" if not np.isnan(s) else "  Saturation flow : N/A")
print(f"  Capacity         : {capacity:.1f} veh/hr" if not np.isnan(capacity) else "  Capacity        : N/A")
print(f"  DoS              : {dos:.4f}" if not np.isnan(dos) else "  DoS             : N/A")
print(f"  LOS              : {los}")
print(f"{'='*50}")