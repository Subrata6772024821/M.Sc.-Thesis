import pandas as pd

# ======================================
# FILE (edit in place - adds a 2nd sheet)
# ======================================
file_path = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Accumulation\Accumulation_Density_1sec.xlsx"

# ======================================
# READ EXISTING 1-SEC SHEET
# ======================================
df = pd.read_excel(file_path, sheet_name=0)

# ======================================
# GROUP EVERY 30 ROWS AND AVERAGE ACCUMULATION
# ======================================
group_size = 30

df["Group"] = df.index // group_size

averaged = (
    df.groupby("Group")["Accumulation"]
    .mean()
    .reset_index(drop=True)
)

n_groups = len(averaged)
time_labels = [f"{i*group_size}-{(i+1)*group_size}" for i in range(n_groups)]

result = pd.DataFrame({
    "Time (s)": time_labels,
    "Avg Accumulation (30s)": averaged.values
})

# ======================================
# WRITE AS A NEW SHEET IN THE SAME WORKBOOK
# ======================================
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    result.to_excel(writer, sheet_name="Avg Accumulation 120s", index=False)

print("✅ 2nd sheet 'Avg Accumulation 120s' added to:")
print(file_path)
print(result.head())