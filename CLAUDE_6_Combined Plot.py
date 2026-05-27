import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================================
# STEP 6 — COMBINED FOUR-PANEL FIGURE
#   Panel 1 (top, large) : Time-Space diagram (speed colormap)
#   Panel 2              : Queue length over time
#   Panel 3              : Multiple-stop events (Gantt bars per vehicle)
#   Panel 4 (bottom)     : Density (30-second intervals)
# All panels share the same X axis (Time [s])
# ==========================================================

trajectory_file = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\SATHORN_03 DEC_17.00_17.30\T.trajectory_1s_final.xlsx"
queue_file      = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\SATHORN_03 DEC_17.00_17.30\T.queue_length.xlsx"
stop_file       = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\SATHORN_03 DEC_17.00_17.30\T.stop_analysis.xlsx"
density_file    = r"D:\THESIS\SATHORN_DENSITY_QUEUE LENGTH\SATHORN_03 DEC_17.00_17.30\Density_Accumulation.xlsx"
output_png      = str(Path(trajectory_file).parent / "T.combined_4panel.png")

STOP_LINE_POS = 96.0   # metres
FIGURE_DPI    = 200
FIGURE_SIZE   = (22, 18)

# Stop event colors per stop number
STOP_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#4CAF50", 4: "#9C27B0", 5: "#F44336"}

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
print("Loading trajectory data ...")
df_traj = pd.read_excel(trajectory_file, sheet_name="Final_Data")
df_traj.columns = df_traj.columns.str.strip()
if "Reconstructed" in df_traj.columns:
    df_traj = df_traj[df_traj["Reconstructed"] == False].copy()
df_traj = df_traj.sort_values(["Track ID", "Time [s]"]).reset_index(drop=True)
print(f"  {df_traj['Track ID'].nunique():,} vehicles")

print("Loading queue length data ...")
df_queue = pd.read_excel(queue_file, sheet_name="Queue Per Second")
df_queue.columns = df_queue.columns.str.strip()

print("Loading stop analysis data ...")
df_stops_wide = pd.read_excel(stop_file, sheet_name="Multi-Stop Vehicles")
df_stops_wide.columns = df_stops_wide.columns.str.strip()

# Parse wide-format stop columns -> long format
# Columns: Track ID | Type | Total Stops | Stop 1 - Start [s] | Stop 1 - End [s] | ...
stop_events = []
for _, row in df_stops_wide.iterrows():
    tid   = row["Track ID"]
    vtype = row["Type"] if "Type" in row else "Unknown"
    n_stops = int(row["Total Stops"])
    for n in range(1, n_stops + 1):
        start_col = f"Stop {n} - Start [s]"
        end_col   = f"Stop {n} - End [s]"
        dur_col   = f"Stop {n} - Duration [s]"
        if start_col in row and not pd.isna(row[start_col]):
            stop_events.append({
                "Track ID"    : tid,
                "Type"        : vtype,
                "Stop #"      : n,
                "Start [s]"   : float(row[start_col]),
                "End [s]"     : float(row[end_col]),
                "Duration [s]": float(row[dur_col]),
            })

df_stops_long = pd.DataFrame(stop_events)
multi_ids = sorted(df_stops_wide["Track ID"].tolist())
print(f"  {len(multi_ids)} multi-stop vehicles  |  {len(df_stops_long)} stop events")

print("Loading density data ...")
df_dens = pd.read_excel(density_file)
df_dens.columns = df_dens.columns.str.strip()
df_dens["t_start"] = df_dens["Time (s)"].str.split("-").str[0].astype(float)
df_dens["t_end"]   = df_dens["Time (s)"].str.split("-").str[1].astype(float)
df_dens["t_mid"]   = (df_dens["t_start"] + df_dens["t_end"]) / 2

# Common time axis
t_min = int(df_traj["Time [s]"].min())
t_max = int(df_traj["Time [s]"].max())

# ----------------------------------------------------------
# FIGURE LAYOUT
# ----------------------------------------------------------
fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    4, 1,
    height_ratios=[3.0, 1.0, 1.2, 1.0],
    hspace=0.07,
)
ax1 = fig.add_subplot(gs[0])                      # Time-Space
ax2 = fig.add_subplot(gs[1], sharex=ax1)          # Queue length
ax3 = fig.add_subplot(gs[2], sharex=ax1)          # Multi-stop Gantt
ax4 = fig.add_subplot(gs[3], sharex=ax1)          # Density

# Vertical grid lines helper
def add_vgrid(ax):
    for xt in np.arange(250, t_max, 250):
        ax.axvline(xt, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0)

# ----------------------------------------------------------
# PANEL 1 — TIME-SPACE DIAGRAM
# ----------------------------------------------------------
ax1.set_facecolor("#f8f8f8")

vmin = df_traj["Speed [km/h]"].quantile(0.02)
vmax = df_traj["Speed [km/h]"].quantile(0.98)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.plasma

for track_id, grp in df_traj.groupby("Track ID", sort=False):
    grp    = grp.sort_values("Time [s]")
    t      = grp["Time [s]"].values
    pos    = grp["cumulative_distance"].values
    spd    = grp["Speed [km/h]"].values
    is_multi = track_id in multi_ids
    lw     = 1.4 if is_multi else 0.6
    alpha  = 0.85 if is_multi else 0.50
    for i in range(len(t) - 1):
        ax1.plot([t[i], t[i+1]], [pos[i], pos[i+1]],
                 color=cmap(norm(spd[i])),
                 linewidth=lw, alpha=alpha, solid_capstyle="round")

# Queue zone shading
ax1.fill_between(
    df_queue["Time [s]"],
    STOP_LINE_POS - df_queue["Queue Length [m]"],
    STOP_LINE_POS,
    where=df_queue["Queue Length [m]"] > 0,
    color="#CC000018", zorder=0, label="Queue zone"
)
ax1.axhline(STOP_LINE_POS, color="#CC0000", linewidth=1.0,
            linestyle="--", alpha=0.7, label=f"Stop line ({STOP_LINE_POS:.0f} m)")

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, fraction=0.010, pad=0.01)
cbar.set_label("Speed (km/h)", fontsize=8)
cbar.outline.set_linewidth(0.4)
cbar.ax.tick_params(labelsize=7)

ax1.set_ylabel("Space [m]", fontsize=11)
ax1.set_ylim(-2, STOP_LINE_POS * 1.1)
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="#ccc")
ax1.grid(axis="y", color="#dddddd", linewidth=0.4, linestyle="--", zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(labelbottom=False, labelsize=9)
ax1.set_title(
    "Space-Time Trajectories  |  Queue Length  |  Multiple Stops  |  Density\n"
    "Location: Sathon-Convent, Approach: Sathon Road (All lanes combined)  |  "
    "Dec 03, 2025   13:15 – 13:45",
    fontsize=12, fontweight="bold", pad=8
)
add_vgrid(ax1)

# ----------------------------------------------------------
# PANEL 2 — QUEUE LENGTH
# ----------------------------------------------------------
ax2.set_facecolor("white")

ax2.fill_between(
    df_queue["Time [s]"],
    df_queue["Queue Length [m]"],
    color="#CC000028", step="post", zorder=1
)
ax2.step(
    df_queue["Time [s]"],
    df_queue["Queue Length [m]"],
    color="#CC0000", linewidth=1.2, where="post",
    label="Queue length [m]", zorder=2
)

q_active = df_queue[df_queue["Queue Length [m]"] > 0]["Queue Length [m]"]
if len(q_active) > 0:
    mean_q = q_active.mean()
    ax2.axhline(mean_q, color="#FF6600", linewidth=1.0, linestyle=":",
                alpha=0.85, label=f"Mean: {mean_q:.1f} m", zorder=3)
    idx_max = df_queue["Queue Length [m]"].idxmax()
    t_peak  = df_queue.loc[idx_max, "Time [s]"]
    q_peak  = df_queue.loc[idx_max, "Queue Length [m]"]
    ax2.annotate(
        f" Max:{q_peak:.1f}m",
        xy=(t_peak, q_peak), xytext=(t_peak + 30, q_peak * 0.85),
        fontsize=7, color="#990000",
        arrowprops=dict(arrowstyle="->", color="#990000", lw=0.7)
    )

ax2.set_ylabel("Queue\n[m]", fontsize=10)
ax2.set_ylim(0, STOP_LINE_POS * 1.1)
ax2.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#ccc")
ax2.grid(color="#eeeeee", linewidth=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(labelbottom=False, labelsize=9)
add_vgrid(ax2)

# ----------------------------------------------------------
# PANEL 3 — MULTIPLE STOPS (Gantt chart)
# ----------------------------------------------------------
ax3.set_facecolor("#fafafa")

# Y axis: one row per multi-stop vehicle (sorted by Track ID)
y_ticks   = list(range(len(multi_ids)))
y_labels  = [str(tid) for tid in multi_ids]
y_map     = {tid: i for i, tid in enumerate(multi_ids)}

for _, ev in df_stops_long.iterrows():
    tid      = ev["Track ID"]
    y_pos    = y_map[tid]
    t_start  = ev["Start [s]"]
    t_end    = ev["End [s]"]
    stop_n   = int(ev["Stop #"])
    color    = STOP_COLORS.get(stop_n, "#888888")

    ax3.barh(
        y_pos,
        width  = max(t_end - t_start, 2.0),   # min 2s width so short stops are visible
        left   = t_start,
        height = 0.55,
        color  = color,
        edgecolor = "white",
        linewidth = 0.4,
        zorder = 2,
    )
    # Label stop number inside bar if wide enough
    if (t_end - t_start) > 15:
        ax3.text(
            t_start + (t_end - t_start) / 2, y_pos,
            f"S{stop_n}", ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=3
        )

ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels, fontsize=7.5)
ax3.set_ylabel("Track ID\n(multi-stop)", fontsize=10)
ax3.set_ylim(-0.7, len(multi_ids) - 0.3)
ax3.tick_params(labelbottom=False, labelsize=9)
ax3.grid(axis="x", color="#eeeeee", linewidth=0.4, zorder=0)
ax3.set_axisbelow(True)
for spine in ax3.spines.values():
    spine.set_linewidth(0.5)
add_vgrid(ax3)

# Legend for stop numbers
legend_handles = [
    mpatches.Patch(color=STOP_COLORS[n], label=f"Stop {n}")
    for n in sorted(STOP_COLORS.keys())
    if n in df_stops_long["Stop #"].values
]
ax3.legend(handles=legend_handles, loc="upper right",
           fontsize=8, framealpha=0.9, edgecolor="#ccc",
           title="Stop event", title_fontsize=8)

# ----------------------------------------------------------
# PANEL 4 — DENSITY
# ----------------------------------------------------------
ax4.set_facecolor("white")

d_max  = df_dens["Density (veh/km/lane)"].max()
d_norm = mcolors.Normalize(vmin=0, vmax=d_max)
d_cmap = cm.YlOrRd
bar_colors = [d_cmap(d_norm(v)) for v in df_dens["Density (veh/km/lane)"]]

ax4.bar(
    df_dens["t_start"],
    df_dens["Density (veh/km/lane)"],
    width     = 30,
    align     = "edge",
    color     = bar_colors,
    edgecolor = "white",
    linewidth = 0.3,
    zorder    = 2,
)

mean_d = df_dens["Density (veh/km/lane)"].mean()
ax4.axhline(mean_d, color="#333333", linewidth=0.9, linestyle=":",
            alpha=0.7, label=f"Mean: {mean_d:.1f} veh/km/lane", zorder=3)

sm_d  = cm.ScalarMappable(cmap=d_cmap, norm=d_norm)
sm_d.set_array([])
cbar_d = plt.colorbar(sm_d, ax=ax4, fraction=0.010, pad=0.01)
cbar_d.set_label("Density\n[veh/km/lane]", fontsize=7.5)
cbar_d.outline.set_linewidth(0.4)
cbar_d.ax.tick_params(labelsize=7)

ax4.set_ylabel("Density\n[veh/km/lane]", fontsize=10)
ax4.set_ylim(0, d_max * 1.2)
ax4.set_xlabel("Time [s]", fontsize=11)
ax4.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#ccc")
ax4.grid(axis="y", color="#eeeeee", linewidth=0.4, zorder=0)
ax4.set_axisbelow(True)
ax4.tick_params(labelsize=9)
add_vgrid(ax4)

# ----------------------------------------------------------
# SHARED X AXIS LIMITS
# ----------------------------------------------------------
ax1.set_xlim(t_min, t_max)

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
plt.savefig(output_png, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  Saved -> {output_png}")