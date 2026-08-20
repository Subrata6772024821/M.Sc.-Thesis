import pandas as pd
import numpy as np
import os
import csv

# ======================================
# FILE PATH
# ======================================
file_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Accumulation\Raw_Accumulation_Density_5step.csv"

# ======================================
# PEEK AT RAW FILE FOR DIAGNOSTICS
# ======================================
with open(file_path, encoding="utf-8-sig", errors="replace") as f:
    first_lines = [f.readline() for _ in range(3)]

print("===== RAW FILE PREVIEW =====")
for i, line in enumerate(first_lines, start=1):
    print(f"--- line {i} (len={len(line)}) ---")
    print(line[:300])
print("=============================")

# ======================================
# MANUAL ROW-BY-ROW PARSE (ragged CSV)
# ======================================
# This file has a FIXED set of 8 summary columns per vehicle, followed by a
# VARIABLE-length run of trajectory samples (x, y, Speed, Tan. Acc., Lat. Acc.,
# Time), one group per GPS sample. Row length therefore differs per vehicle,
# which is why pandas.read_csv fails with "Expected N fields, saw M".
# We only need the first 8 fields, so we parse with csv.reader (which also
# correctly handles the quoted "Gate 14" style fields) and discard the rest.

FIXED_COLUMNS = [
    "Track ID", "Type", "Entry Gate", "Entry Time [s]",
    "Exit Gate", "Exit Time [s]", "Traveled Dist. [m]", "Avg. Speed [km/h]",
]

rows = []
with open(file_path, encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f, skipinitialspace=True)
    header = next(reader)  # skip header line
    for raw_row in reader:
        if not raw_row:
            continue
        rows.append(raw_row[:len(FIXED_COLUMNS)])

df = pd.DataFrame(rows, columns=FIXED_COLUMNS)

# Strip stray whitespace from string fields
for col in ("Type", "Entry Gate", "Exit Gate"):
    df[col] = df[col].astype(str).str.strip()

print("Columns after parsing:")
print(df.columns.tolist())
print(f"Shape: {df.shape}")
print(df.head())

# ======================================
# DEFINE REQUIRED COLUMNS
# ======================================
entry_col = "Entry Time [s]"
exit_col  = "Exit Time [s]"
type_col  = "Type"

# ======================================
# KEEP ONLY VEHICLE ROWS
# ======================================
df = df[df[type_col].notna() & (df[type_col] != "")]

# Convert times to numeric
df[entry_col] = pd.to_numeric(df[entry_col], errors="coerce")
df[exit_col]  = pd.to_numeric(df[exit_col], errors="coerce")

# Remove rows without time
df = df.dropna(subset=[entry_col, exit_col], how="all")

# ======================================
# FIND MAX TIME
# ======================================
max_time = int(max(df[entry_col].max(), df[exit_col].max()))
print("Max time:", max_time)

# ======================================
# CREATE 1-SECOND INTERVAL FLOWS
# ======================================
interval = 1
results = []

start = 0

while start <= max_time:

    end = start + interval

    # -------- FLOW IN (ENTRY) ----------
    flow_in = df[
        (df[entry_col] >= start) &
        (df[entry_col] < end)
    ].shape[0]

    # -------- FLOW OUT (EXIT) ----------
    flow_out = df[
        (df[exit_col] >= start) &
        (df[exit_col] < end)
    ].shape[0]

    # -------- NET FLOW ----------
    net_flow = flow_in - flow_out

    results.append([
        f"{start}-{end}",
        flow_in,
        flow_out,
        net_flow
    ])

    start += interval

# ======================================
# CREATE DATAFRAME
# ======================================
result = pd.DataFrame(results, columns=[
    "Time (s)",
    "Flow (In)",
    "Flow (Out)",
    "Net flow"
])

# ======================================
# ACCUMULATION (CUMULATIVE NET FLOW)
# ======================================
result["Accumulation"] = result["Net flow"].cumsum()

# ======================================
# SAVE (back into the source folder, next to the input file)
# ======================================
output_file = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Accumulation\Accumulation_Density_1sec.xlsx"

result.to_excel(output_file, index=False)

print("\n✅ Correct Excel created!")
print("Saved at:", output_file)