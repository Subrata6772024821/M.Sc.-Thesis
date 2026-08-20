"""
Vehicle-wise Composition Percentage Calculator
------------------------------------------------
Reads a trajectory CSV where each row = one vehicle track:
    Track ID, Type, Entry Gate, Entry Time, Exit Gate, Exit Time,
    Traveled Dist., Avg. Speed, <trajectory points...>

Since the trajectory columns make each row a different length ("ragged" CSV),
this script reads the file manually with Python's csv module instead of
pandas.read_csv(), which would otherwise raise a "tokenizing" error on
rows of unequal length.

Output:
    1. Printed summary table of counts + percentages
    2. A CSV file: vehicle_composition_summary.csv
    3. A bar chart and pie chart (PNG files) saved next to the script
"""

import csv
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. SET YOUR FILE PATH HERE
# ------------------------------------------------------------------
FILE_PATH = r"D:\Thesis_Final Data Analysis\05_12_25_1315_1345\30_sec\Accumulation\Raw_Accumulation_Density_5step.csv"

# Where to save the outputs (same folder as the input file by default)
OUTPUT_DIR = os.path.dirname(FILE_PATH)


def compute_vehicle_composition(file_path):
    """Reads the CSV and counts vehicles by Type (column index 1)."""
    type_counts = {}
    total_vehicles = 0

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row

        for row in reader:
            if len(row) < 2:
                continue  # skip empty/malformed lines
            vehicle_type = row[1].strip()
            if vehicle_type == "":
                continue
            type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1
            total_vehicles += 1

    return type_counts, total_vehicles


def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found:\n{FILE_PATH}")

    type_counts, total_vehicles = compute_vehicle_composition(FILE_PATH)

    if total_vehicles == 0:
        raise ValueError("No vehicle records found. Check the file format/path.")

    # Sort by count, descending
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\nTotal vehicles counted: {total_vehicles}\n")
    print(f"{'Vehicle Type':<18}{'Count':>8}{'Percentage':>14}")
    print("-" * 42)
    for vtype, count in sorted_types:
        pct = count / total_vehicles * 100
        print(f"{vtype:<18}{count:>8}{pct:>13.2f}%")

    # ------------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "vehicle_composition_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle Type", "Count", "Percentage (%)"])
        for vtype, count in sorted_types:
            pct = count / total_vehicles * 100
            writer.writerow([vtype, count, round(pct, 2)])
    print(f"\nSummary CSV saved to: {summary_path}")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    labels = [t for t, _ in sorted_types]
    counts = [c for _, c in sorted_types]
    percentages = [c / total_vehicles * 100 for c in counts]

    # Bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, percentages, color="steelblue")
    plt.ylabel("Percentage (%)")
    plt.title("Vehicle-wise Composition Percentage")
    plt.xticks(rotation=30, ha="right")
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "vehicle_composition_bar.png")
    plt.savefig(bar_path, dpi=200)
    print(f"Bar chart saved to: {bar_path}")

    # Pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Vehicle-wise Composition")
    plt.tight_layout()
    pie_path = os.path.join(OUTPUT_DIR, "vehicle_composition_pie.png")
    plt.savefig(pie_path, dpi=200)
    print(f"Pie chart saved to: {pie_path}")

    plt.show()


if __name__ == "__main__":
    main()