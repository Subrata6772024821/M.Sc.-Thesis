import pandas as pd
from pathlib import Path

# ==========================================================
# GREEN TIME TEMPLATE CREATOR
# Run once per dataset to create an input template.
# Fill in the green phase intervals, then run step7.
# ==========================================================

# ----------------------------------------------------------
# CONFIGURE FOR YOUR DATASET
# ----------------------------------------------------------
DATASET_ID      = "S_Dec03_1315"
LOCATION        = "Sathorn-Convent"
DATE            = "2025-12-03"
SESSION_START   = "13:15"
SESSION_END     = "13:45"
TRAJECTORY_FILE = "S.trajectory_1s_final.xlsx"

output_folder   = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH"
output_file     = str(Path(output_folder) / f"GreenTime_{DATASET_ID}.xlsx")

# ----------------------------------------------------------
# PRE-FILLED EXAMPLE — replace with your actual green times
# Format: "MM:SS" relative to session start (0:00 = recording start)
# One row per green phase
# ----------------------------------------------------------
green_phases = [
    # ("Green_Start", "Green_End")  ← MM:SS format
    ("0:00",  "0:11" ),   # Phase 1  — partial (session started mid-green)
    ("1:27",  "2:18" ),   # Phase 2
    ("3:23",  "4:14" ),   # Phase 3
    ("5:20",  "6:11" ),   # Phase 4
    ("7:13",  "8:20" ),   # Phase 5
    ("9:24",  "10:32"),   # Phase 6
    ("11:27", "12:24"),   # Phase 7
    ("13:14", "14:14"),   # Phase 8
    ("15:11", "16:15"),   # Phase 9
    ("17:04", "17:56"),   # Phase 10
    ("19:05", "20:04"),   # Phase 11
    ("21:05", "21:50"),   # Phase 12
    ("23:00", "24:01"),   # Phase 13
    ("25:16", "26:10"),   # Phase 14
    ("27:23", "28:15"),   # Phase 15
    ("29:25", "30:00"),   # Phase 16 — partial (session ended mid-green)
]

# ----------------------------------------------------------
# BUILD DATAFRAME
# ----------------------------------------------------------
def to_seconds(mmss):
    mm, ss = mmss.strip().split(":")
    return int(mm)*60 + int(ss)

rows = []
for i, (gs, ge) in enumerate(green_phases, start=1):
    g_start_s = to_seconds(gs)
    g_end_s   = to_seconds(ge)
    duration  = g_end_s - g_start_s

    is_partial = (i == 1 and g_start_s == 0) or \
                 (i == len(green_phases) and g_end_s == 1800)

    rows.append({
        "Dataset_ID"       : DATASET_ID,
        "Location"         : LOCATION,
        "Date"             : DATE,
        "Session_Start"    : SESSION_START,
        "Session_End"      : SESSION_END,
        "Trajectory_File"  : TRAJECTORY_FILE,
        "Phase_No"         : i,
        "Green_Start_MMSS" : gs,
        "Green_End_MMSS"   : ge,
        "Green_Start_s"    : g_start_s,
        "Green_End_s"      : g_end_s,
        "Green_Duration_s" : duration,
        "Partial_Phase"    : "Yes" if is_partial else "No",
        "Use_For_SatFlow"  : "No"  if is_partial else "Yes",
    })

df = pd.DataFrame(rows)

# Summary row
total_green = df["Green_Duration_s"].sum()
n_complete  = (df["Use_For_SatFlow"] == "Yes").sum()

# ----------------------------------------------------------
# SAVE EXCEL WITH FORMATTING
# ----------------------------------------------------------
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Green Phases", index=False,
                startrow=4)

    wb = writer.book
    ws = writer.sheets["Green Phases"]

    # Formats
    title_fmt   = wb.add_format({"bold": True, "font_size": 13,
                                  "font_color": "#1F3864"})
    info_fmt    = wb.add_format({"font_size": 10,
                                  "font_color": "#444444"})
    hdr_fmt     = wb.add_format({"bold": True, "bg_color": "#D9E1F2",
                                  "border": 1, "align": "center",
                                  "valign": "vcenter", "text_wrap": True})
    data_fmt    = wb.add_format({"align": "center", "border": 1})
    partial_fmt = wb.add_format({"align": "center", "border": 1,
                                  "bg_color": "#FFF2CC"})
    complete_fmt= wb.add_format({"align": "center", "border": 1,
                                  "bg_color": "#E2EFDA"})
    summary_fmt = wb.add_format({"bold": True, "bg_color": "#F2F2F2",
                                  "border": 1, "align": "center"})

    # Title block (rows 0-3)
    ws.write(0, 0, "GREEN TIME INPUT TEMPLATE", title_fmt)
    ws.write(1, 0, f"Dataset  : {DATASET_ID}   |   "
                   f"Location : {LOCATION}   |   "
                   f"Date : {DATE}   |   "
                   f"Session : {SESSION_START} – {SESSION_END}", info_fmt)
    ws.write(2, 0, f"Total green time : {total_green}s  "
                   f"({total_green/60:.2f} min)   |   "
                   f"Green ratio : {total_green/1800*100:.1f}%   |   "
                   f"Complete phases for saturation flow : {n_complete}",
             info_fmt)
    ws.write(3, 0, "Yellow = partial phase (exclude from saturation flow)   |   "
                   "Green = complete phase (include in saturation flow)",
             wb.add_format({"italic": True, "font_size": 9,
                             "font_color": "#888888"}))

    # Column widths
    col_widths = [16, 18, 14, 14, 12, 34, 10, 18, 16, 15, 13, 18, 14, 18]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    # Header row (row 4)
    for ci, col in enumerate(df.columns):
        ws.write(4, ci, col, hdr_fmt)

    # Data rows (row 5 onwards)
    for ri, row_data in df.iterrows():
        excel_row = ri + 5
        is_partial = row_data["Partial_Phase"] == "Yes"
        row_fmt = partial_fmt if is_partial else complete_fmt
        for ci, val in enumerate(row_data):
            ws.write(excel_row, ci, val, row_fmt)

    # Summary block below data
    sum_row = len(df) + 6
    ws.write(sum_row,     0, "SUMMARY",                     summary_fmt)
    ws.write(sum_row + 1, 0, "Total green time (s)",        summary_fmt)
    ws.write(sum_row + 1, 1, total_green,                   summary_fmt)
    ws.write(sum_row + 2, 0, "Total green time (min)",      summary_fmt)
    ws.write(sum_row + 2, 1, round(total_green/60, 2),      summary_fmt)
    ws.write(sum_row + 3, 0, "Green ratio (G/T)",           summary_fmt)
    ws.write(sum_row + 3, 1, round(total_green/1800, 4),    summary_fmt)
    ws.write(sum_row + 4, 0, "Total phases",                summary_fmt)
    ws.write(sum_row + 4, 1, len(df),                       summary_fmt)
    ws.write(sum_row + 5, 0, "Complete phases (for s)",     summary_fmt)
    ws.write(sum_row + 5, 1, n_complete,                    summary_fmt)
    ws.write(sum_row + 6, 0, "Partial phases (excluded)",   summary_fmt)
    ws.write(sum_row + 6, 1, len(df) - n_complete,          summary_fmt)

    ws.freeze_panes(5, 0)

print(f"Template created -> {output_file}")
print(f"  Total green : {total_green}s ({total_green/60:.2f} min)")
print(f"  Green ratio : {total_green/1800*100:.1f}%")
print(f"  Complete phases for saturation flow: {n_complete}")
print()
print("Next step: run step7_saturation_flow.py with this template file as input.")