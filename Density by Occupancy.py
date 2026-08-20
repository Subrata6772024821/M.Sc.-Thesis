import pandas as pd
import os

# ======================================
# FILE PATH
# ======================================
file_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\120_sec\Occupancy\Raw_Occupancy_Density_Exit_5step.csv"

# ======================================
# READ CSV — parse raw lines, take only first 8 fields
# ======================================
rows = []
with open(file_path, encoding="utf-8", errors="replace") as f:
    next(f)  # skip header
    for line in f:
        fields = line.split(",")
        if len(fields) >= 6:
            rows.append({
                "Track ID":          fields[0].strip(),
                "Type":              fields[1].strip(),
                "Entry Gate":        fields[2].strip(),
                "Entry Time [s]":    fields[3].strip(),
                "Exit Gate":         fields[4].strip(),
                "Exit Time [s]":     fields[5].strip(),
            })

df = pd.DataFrame(rows)

# ======================================
# CLEAN
# ======================================
entry_col = "Entry Time [s]"
exit_col  = "Exit Time [s]"
type_col  = "Type"

df = df[df[type_col].notna()].copy()
df[entry_col] = pd.to_numeric(df[entry_col], errors="coerce")
df[exit_col]  = pd.to_numeric(df[exit_col],  errors="coerce")
df = df.dropna(subset=[entry_col, exit_col])

print(f"✅ Loaded {len(df)} vehicles")
print(df[[type_col, entry_col, exit_col]].head(10))

max_time = int(max(df[entry_col].max(), df[exit_col].max()))

# ======================================
# 120-SECOND INTERVAL OCCUPANCY
# ======================================
WINDOW = 120
results = []

for start in range(0, max_time + 1, WINDOW):
    end = start + WINDOW

    mask = (df[exit_col] > start) & (df[entry_col] < end)
    sub = df[mask]

    clip_start = sub[entry_col].clip(lower=start)
    clip_end   = sub[exit_col].clip(upper=end)

    intervals = sorted(zip(clip_start, clip_end))

    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    t_occupied = sum(e - s for s, e in merged)

    results.append({
        "Time (s)":          f"{start}-{end}",
        "Time occupied (s)": round(t_occupied, 4)
    })

# ======================================
# SAVE OUTPUT TO DESKTOP
# ======================================
result_df = pd.DataFrame(results)

desktop     = os.path.join(os.path.expanduser("~"), "Desktop")
output_file = os.path.join(desktop, "05_12_25_1315_1345.Occupancy_Density_Exit_120sec.xlsx")
result_df.to_excel(output_file, index=False)

print(f"\n✅ Done! Saved to: {output_file}")