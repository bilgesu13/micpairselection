#%%

PAPER_STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
}

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(PAPER_STYLE)


#%%


#### accuracy vs. the number of microphone pairs

## in the paper

# ==========================
# Load data
# ==========================
threshold = 0.25
pair_counts = [2, 4, 6, 8, 12]

measure1 = [np.load(f'measure1200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
measure2 = [np.load(f'measure2200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
broadside = [np.load(f'broadside200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
random = [np.load(f'random200newsource_realistic6db_{p}pairs.npy') for p in pair_counts]

methods = {
    'Norm-based': measure1,
    'Kurtosis-based': measure2,
    'Zero-lag CC-based': broadside,
    'Random': random
}

colors = {
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

# ==========================
# Plot accuracy curves
# ==========================
fig, ax = plt.subplots()

for label, data_list in methods.items():
    accuracies = []
    for data in data_list:
        if np.isscalar(data):
            acc = 100.0 if data <= threshold else 0.0
        else:
            acc = 100.0 * np.sum(data <= threshold) / len(data)
        accuracies.append(acc)

    ax.plot(
        pair_counts,
        accuracies,
        marker='o',
        markersize=9,        # larger markers
        label=label,
        color=colors[label]
    )

# Labels and axes
ax.set_xlabel("# of Microphone Pairs")
ax.set_ylabel("Localization Accuracy (%)")

# Reviewer request: rescale y-axis to remove empty zero region
ax.set_ylim(20, 105)

ax.set_xticks(pair_counts)

# No grid (clean look)
ax.grid(False)

# Legend
#ax.legend(title="Algorithm")
ax.legend()


fig.tight_layout()

# Save if needed:
fig.savefig("accuracy_vs_pairs.png", dpi=300)
# fig.savefig("accuracy_vs_pairs.pdf", bbox_inches="tight")

plt.show()
# %%
### accuracy vs. number of pairs with conventional srp

#### accuracy vs. the number of microphone pairs

## in the paper

# ==========================
# Load data
# ==========================
threshold = 0.25
pair_counts = [2, 4, 6, 8, 12]

measure1 = [np.load(f'measure1200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
measure2 = [np.load(f'measure2200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
broadside = [np.load(f'broadside200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
random = [np.load(f'random200newsource_realistic6db_{p}pairs.npy') for p in pair_counts]

# Conventional SRP using all microphone pairs
conventional = np.load('conv6dB.npy')
conv_acc = 100.0 * np.sum(conventional <= threshold) / len(conventional)

methods = {
    'Norm-based': measure1,
    'Kurtosis-based': measure2,
    'Zero-lag CC-based': broadside,
    'Random': random
}

colors = {
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

# ==========================
# Plot accuracy curves
# ==========================
fig, ax = plt.subplots()

for label, data_list in methods.items():
    accuracies = []

    for data in data_list:
        if np.isscalar(data):
            acc = 100.0 if data <= threshold else 0.0
        else:
            acc = 100.0 * np.sum(data <= threshold) / len(data)

        accuracies.append(acc)

    ax.plot(
        pair_counts,
        accuracies,
        marker='o',
        markersize=9,
        label=label,
        color=colors[label]
    )

# Conventional SRP reference line
ax.axhline(
    y=conv_acc,
    color='tab:purple',
    linestyle='--',
    linewidth=2,
    label='Conventional SRP'
)

# Labels and axes
ax.set_xlabel("# of Microphone Pairs")
ax.set_ylabel("Localization Accuracy (%)")

# Reviewer request: rescale y-axis to remove empty zero region
ax.set_ylim(20, 105)

ax.set_xticks(pair_counts)

# No grid (clean look)
ax.grid(False)

# Legend
ax.legend()

fig.tight_layout()

# Save if needed:
fig.savefig("accuracy_vs_pairs_with_conventional.png", dpi=300)
# fig.savefig("accuracy_vs_pairs.pdf", bbox_inches="tight")

plt.show()




#%%

### box plot of errors for different pair counts

# ==============================
# Load data
# ==============================
pair_counts = [2, 4, 6, 8, 12]

measure1 = [np.load(f'measure1200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
measure2 = [np.load(f'measure2200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
broadside = [np.load(f'broadside200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
random = [np.load(f'random200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]

all_data = [measure1, measure2, broadside, random]
methods = ['Norm-based', 'Kurtosis-based', 'Zero-lag CC-based', 'Random']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ==============================
# Plot grouped boxplots
# ==============================
positions = np.arange(len(pair_counts))
width = 0.18

fig, ax = plt.subplots()   # uses the rcParams figsize

for i, method_data in enumerate(all_data):
    box_positions = positions + (i - 1.5)*width

    bp = ax.boxplot(method_data,
                    positions=box_positions,
                    widths=0.16,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(linewidth=1.3),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    # Color fill
    for patch in bp['boxes']:
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.45)

# Labels
ax.set_xticks(positions)
ax.set_xticklabels(pair_counts)
ax.set_xlabel("# of Microphone Pairs")
ax.set_ylabel("Localization Error (m)")

ax.grid(False)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.45)
           for i in range(4)]
#ax.legend(handles, methods, title="Algorithm")
ax.legend(handles, methods)


fig.tight_layout()

# Save for LaTeX (optional)
fig.savefig("november_boxplots_micpairs.png", dpi=300, bbox_inches="tight")
# fig.savefig("november_boxplots.pdf", bbox_inches="tight")

plt.show()

#%%

### box plot of errors for different pair counts with conventional SRP reference

### box plot of errors for different pair counts

# ==============================
# Load data
# ==============================
pair_counts = [2, 4, 6, 8, 12]

measure1 = [np.load(f'measure1200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
measure2 = [np.load(f'measure2200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
broadside = [np.load(f'broadside200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]
random = [np.load(f'random200newsource_realistic6dB_{p}pairs.npy') for p in pair_counts]

# Conventional SRP
conventional = np.load('conv6dB.npy')
conv_median = np.median(conventional)

all_data = [measure1, measure2, broadside, random]
methods = ['Norm-based', 'Kurtosis-based', 'Zero-lag CC-based', 'Random']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ==============================
# Plot grouped boxplots
# ==============================
positions = np.arange(len(pair_counts))
width = 0.18

fig, ax = plt.subplots()

for i, method_data in enumerate(all_data):
    box_positions = positions + (i - 1.5) * width

    bp = ax.boxplot(
        method_data,
        positions=box_positions,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2)
    )

    for patch in bp['boxes']:
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.45)

# ==============================
# Conventional SRP reference
# ==============================
ax.axhline(
    y=conv_median,
    color='tab:purple',
    linestyle='--',
    linewidth=2
)

# Labels
ax.set_xticks(positions)
ax.set_xticklabels(pair_counts)
ax.set_xlabel("# of Microphone Pairs")
ax.set_ylabel("Localization Error (m)")

ax.grid(False)

# ==============================
# Legend
# ==============================
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.45)
    for i in range(4)
]

handles.append(
    plt.Line2D(
        [0], [0],
        color='tab:purple',
        linestyle='--',
        linewidth=2
    )
)

labels = methods + ['Conventional SRP']

ax.legend(handles, labels)

fig.tight_layout()

# Save for LaTeX
fig.savefig("november_boxplots_micpairs_with_conventional.png",
            dpi=300,
            bbox_inches="tight")

# fig.savefig("november_boxplots.pdf",
#             bbox_inches="tight")

plt.show()



# %%
### in the paper
### accuracy vs. SNR curves


# ==========================
# Data
# ==========================
snr = [0, 6, 12]
threshold = 0.25

conventional = [np.load(f'conv{s}dB.npy') for s in snr] 
measure1     = [np.load(f'measure1200newsource_realistic_snr{s}.npy') for s in snr]
measure2     = [np.load(f'measure2200newsource_realistic_snr{s}.npy') for s in snr]
broadside    = [np.load(f'broadside200newsource_realistic_snr{s}.npy') for s in snr]
random       = [np.load(f'random200newsource_realistic_snr{s}.npy') for s in snr]

methods = {
    'Conventional': conventional,
    'Norm-based': measure1,
    'Kurtosis-based': measure2,
    'Zero-lag CC-based': broadside,
    'Random': random
}

colors = {
    'Conventional': 'tab:purple',
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

# ==========================
# Accuracy curves
# ==========================
fig, ax = plt.subplots()

for label, data_list in methods.items():
    accuracies = []
    for data in data_list:
        if np.isscalar(data):
            acc = 100.0 if data <= threshold else 0.0
        else:
            acc = 100.0 * np.sum(data <= threshold) / len(data)
        accuracies.append(acc)

    ax.plot(
        snr,
        accuracies,
        marker='o',
        markersize=9,
        linewidth=2,
        color=colors[label],
        label=label
    )

# ==========================
# Formatting
# ==========================
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Localization Accuracy (%)")

# Start Y-axis at 50% (your request)
ax.set_ylim(50, 105)

ax.set_xticks(snr)
ax.grid(False)

#ax.legend(title="Algorithm")
ax.legend()


fig.tight_layout()

# Optional save lines
fig.savefig("accuracy_vs_snr.png", dpi=300)
# fig.savefig("accuracy_vs_snr.pdf", dpi=300, bbox_inches='tight')

plt.show()
# %%
### in the paper

### box plot of errors for different SNRs


# ==========================
# Load data
# ==========================
snr = [0, 6, 12]

conventional = [np.load(f'conv{s}dB.npy') for s in snr]
measure1     = [np.load(f'measure1200newsource_realistic_snr{s}.npy') for s in snr]
measure2     = [np.load(f'measure2200newsource_realistic_snr{s}.npy') for s in snr]
broadside    = [np.load(f'broadside200newsource_realistic_snr{s}.npy') for s in snr]
random       = [np.load(f'random200newsource_realistic_snr{s}.npy') for s in snr]

labels = ['Conventional','Norm-based', 'Kurtosis-based', 'Zero-lag CC-based', 'Random']
all_methods = [conventional, measure1, measure2, broadside, random]

colors = ['tab:purple','tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ==========================
# Plot grouped boxplots
# ==========================
fig, ax = plt.subplots()

positions = np.arange(len(snr))   # x-centers: 0, 1, 2
width = 0.18                      # spacing between methods

for i, method in enumerate(all_methods):

    box_positions = positions + (i - 2) * width

    bp = ax.boxplot(
        method,
        positions=box_positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(color='black', linewidth=2)
    )

    for box in bp['boxes']:
        box.set(facecolor=colors[i], alpha=0.45)

# ==========================
# Labels & styling
# ==========================
ax.set_xticks(positions)
ax.set_xticklabels(snr)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Localization Error (m)")

ax.grid(False)

# ==========================
# Legend (default size, same as others)
# ==========================
handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.45)
           for i in range(len(labels))]
#ax.legend(handles, labels, title="Algorithm")

ax.legend(handles, labels)


fig.tight_layout()

# ==========================
# SAVE LIKE THE OTHER FIGURES
# ==========================
fig.savefig("snr_boxplot.png", dpi=300, bbox_inches='tight')
#fig.savefig("snr_boxplot.pdf", dpi=300, bbox_inches='tight')

plt.show()
# %%

### in the paper

### box plot of errors for different reverberation levels (T60s)

# ==========================
# Data
# ==========================
reverb_levels = [0.3, 0.5, 0.74, 1.0, 1.2]
reverbs = ['rev03', 'rev05', 'rev074', 'rev1', 'rev1_2']

conventional_files = [
    "simulations_conv_RT60_0.30s.npy",
    "simulations_conv_RT60_0.50s.npy",
    "simulations_conv_RT60_0.74s.npy",
    "simulations_conv_RT60_1.00s.npy",
    "simulations_conv_RT60_1.20s.npy"
]
conventional = [np.load(fname) for fname in conventional_files]

measure1  = [np.load(f'measure1200newsource_realistic_{r}.npy') for r in reverbs]
measure2  = [np.load(f'measure2200newsource_realistic_{r}.npy') for r in reverbs]
broadside = [np.load(f'broadside200newsource_realistic_{r}.npy') for r in reverbs]
random    = [np.load(f'random200newsource_realistic_{r}.npy') for r in reverbs]

methods = ['Conventional', 'Norm-based', 'Kurtosis-based', 'Zero-lag CC-based', 'Random']
all_data = [conventional, measure1, measure2, broadside, random]

colors = ['tab:purple','tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ==========================
# Plot boxplots
# ==========================
fig, ax = plt.subplots()

positions = np.arange(len(reverb_levels))   # x-centers for groups
width = 0.18                                 # spacing for each method

for i, method_data in enumerate(all_data):

    # Each method has 5 arrays (for 5 reverbs)
    box_positions = positions + (i - 2)*width  # shift left/right of center

    bp = ax.boxplot(
        method_data,
        positions=box_positions,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(color='black', linewidth=2)
    )

    for box in bp['boxes']:
        box.set(facecolor=colors[i], alpha=0.45)

# ==========================
# Formatting
# ==========================
ax.set_xticks(positions)
ax.set_xticklabels([str(t) for t in reverb_levels])
ax.set_xlabel("T60 (s)")
ax.set_ylabel("Localization Error (m)")

ax.grid(False)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.45)
           for i in range(len(methods))]
#ax.legend(handles, methods, title="Algorithm")
#ax.legend(handles, methods, title="Algorithm", fontsize=12)

ax.legend(handles, methods)


fig.tight_layout()

# Save optionally:
fig.savefig("boxplot_reverb_localization_error.png", dpi=300, bbox_inches='tight')
# fig.savefig("boxplot_reverb_localization_error.pdf", dpi=300, bbox_inches='tight')

plt.show()

# %%

### accuracy vs. reverberation level (T60) curves

# ==========================
# Data
# ==========================
reverb_levels = [0.3, 0.5, 0.74, 1.0, 1.2]
reverbs = ['rev03', 'rev05', 'rev074', 'rev1', 'rev1_2']
threshold = 0.25

# Conventional (different names)
conventional_files = [
    "simulations_conv_RT60_0.30s.npy",
    "simulations_conv_RT60_0.50s.npy",
    "simulations_conv_RT60_0.74s.npy",
    "simulations_conv_RT60_1.00s.npy",
    "simulations_conv_RT60_1.20s.npy"
]
conventional = [np.load(f) for f in conventional_files]

# Selection-based methods
measure1  = [np.load(f'measure1200newsource_realistic_{r}.npy') for r in reverbs]
measure2  = [np.load(f'measure2200newsource_realistic_{r}.npy') for r in reverbs]
broadside = [np.load(f'broadside200newsource_realistic_{r}.npy') for r in reverbs]
random    = [np.load(f'random200newsource_realistic_{r}.npy') for r in reverbs]

# ==========================
# Organized methods
# ==========================
methods = {
    'Conventional': conventional,
    'Norm-based': measure1,
    'Kurtosis-based': measure2,
    'Zero-lag CC-based': broadside,
    'Random': random
}

colors = {
    'Conventional': 'tab:purple',
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

# ==========================
# Plot accuracy curves
# ==========================
fig, ax = plt.subplots()

for label, data_list in methods.items():
    accuracies = [
        100.0 * np.sum(data <= threshold) / len(data)
        for data in data_list
    ]

    ax.plot(reverb_levels, accuracies,
            marker='o', markersize=9,
            linewidth=2.2,
            label=label, color=colors[label])

# ==========================
# Formatting
# ==========================
ax.set_xlabel("T60 (s)")
ax.set_ylabel("Localization Accuracy (%)")

# Zooming in as you requested
ax.set_ylim(50, 105)

ax.set_xticks(reverb_levels)
ax.grid(False)

#ax.legend(title="Algorithm")
ax.legend()


fig.tight_layout()

# Optional save
fig.savefig("accuracy_vs_reverb.png", dpi=300, bbox_inches='tight')
# fig.savefig("accuracy_vs_reverb.pdf", bbox_inches='tight')

plt.show()
# %%

### larger for the paper 

### homula

# ==============================
# Load all Homula FINAL errors
# ==============================

def load_err(method, snr):
    fname = f"loc_errors_FINAL__homula_{method}_{snr}dB.npy"
    return np.load(fname)

methods = ["conventional","sparsity1", "sparsity2", "broadside","random"]
method_labels = ["Conventional","Norm-based", "Kurtosis-based",
                 "Zero-lag CC-based", "Random"]

snrs = [0, 6, 12]

# Load all error matrices: shape → [method][snr]
errors = {m: {s: load_err(m, s) for s in snrs} for m in methods}

# ==============================
# Compute medians for plotting
# ==============================

med_0  = [np.median(errors[m][0])  for m in methods]
med_6  = [np.median(errors[m][6])  for m in methods]
med_12 = [np.median(errors[m][12]) for m in methods]

print("Medians at 0 dB: ", med_0)
print("Medians at 6 dB: ", med_6)
print("Medians at 12 dB:", med_12)

medians = np.array([med_0, med_6, med_12])  # shape = 3 × 5

# ==============================
# Plot grouped bar chart (Homula)
# ==============================

snr_labels = ["0 dB", "6 dB", "12 dB"]
x = np.arange(len(method_labels))
width = 0.22

fig, ax = plt.subplots()  # uses the figsize from rcParams

for i in range(3):
    ax.bar(
        x + i*width,
        medians[i],
        width=width,
        label=f"SNR {snr_labels[i]}",
        edgecolor="black",    # improves separation when printed
        linewidth=0.7
    )

ax.set_xticks(x + width)
ax.set_xticklabels(method_labels, rotation=15, ha="right")
ax.set_ylabel("Median Localization Error (m)")
#ax.set_xlabel("Microphone pair selection method")
ax.legend()

# grid: optional / very light
#ax.grid(axis="y", linestyle="--", alpha=0.2)
ax.grid(False)

fig.tight_layout()

# For the paper:
fig.savefig("homula_median_errors.png", bbox_inches="tight")      # PNG, 300 dpi
# fig.savefig("homula_median_errors.pdf", bbox_inches="tight")      # vector PDF

plt.show()

#%%
# ==============================
# Plot median error vs SNR (Homula)
# ==============================

snrs = [0, 6, 12]
snr_labels = ["0 dB", "6 dB", "12 dB"]

methods = ["conventional", "sparsity1", "sparsity2", "broadside", "random"]
method_labels = [
    "Conventional",
    "Norm-based",
    "Kurtosis-based",
    "Zero-lag CC-based",
    "Random"
]

colors = {
    'Conventional': 'tab:purple',
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

# Load errors
errors = {m: {s: load_err(m, s) for s in snrs} for m in methods}

# Median localization error for each method and SNR
medians = {
    m: [np.median(errors[m][s]) for s in snrs]
    for m in methods
}

print("Medians:")
for m, label in zip(methods, method_labels):
    print(f"{label}: {medians[m]}")

# ==============================
# Plot
# ==============================

fig, ax = plt.subplots()

for m, label in zip(methods, method_labels):
    ax.plot(
        snrs,
        medians[m],
        marker="o",
        linewidth=1.8,
        markersize=5,
        label=label,
        color=colors[label]
    )

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Median Localization Error (m)")
ax.set_xticks(snrs)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig("homula_median_errors_2.png", dpi=300, bbox_inches="tight")
# fig.savefig("homula_median_errors.pdf", bbox_inches="tight")

plt.show()



# %%

### make it larger for the paper

### televic

# ==============================
# Load all TELEVIC FINAL errors
# ==============================

def load_err(method, snr):
    fname = f"loc_errors_FINAL__televic_{method}_{snr}dB.npy"
    # fname = f"loc_errors_FINAL20__televic_{method}_{snr}dB.npy"
    return np.load(fname)

methods = ["conventional", "sparsity1", "sparsity2", "broadside", "random"]
method_labels = ["Conventional", "Norm-based", "Kurtosis-based",
                 "Zero-lag CC-based", "Random"]


snrs = [0, 6, 12]

# Load all error matrices: shape → [method][snr]
errors = {m: {s: load_err(m, s) for s in snrs} for m in methods}

# ==============================
# Compute medians for plotting
# ==============================

med_0  = [np.median(errors[m][0])  for m in methods]
med_6  = [np.median(errors[m][6])  for m in methods]
med_12 = [np.median(errors[m][12]) for m in methods]

print("Medians at 0 dB: ", med_0)
print("Medians at 6 dB: ", med_6)
print("Medians at 12 dB:", med_12)

medians = np.array([med_0, med_6, med_12])  # shape = 3 × 5



snr_labels = ["0 dB", "6 dB", "12 dB"]
x = np.arange(len(method_labels))
width = 0.22

fig, ax = plt.subplots() 

for i in range(3):
    ax.bar(
        x + i*width,
        medians[i],
        width=width,
        label=f"SNR {snr_labels[i]}",
        edgecolor="black",      # improves separation of bars when printed
        linewidth=0.7
    )

ax.set_xticks(x + width)
ax.set_xticklabels(method_labels, rotation=15, ha="right")  # slight rotation helps
ax.set_ylabel("Median Localization Error (m)")
#ax.set_xlabel("Microphone pair selection method")
ax.legend()
#ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(False)


fig.tight_layout()

# For the paper: save with high DPI
# fig.savefig("televic_median_errors.pdf", bbox_inches="tight")
fig.savefig("televic_median_errors.png", dpi=400, bbox_inches="tight")

plt.show()
# %%
### Median Error vs SNR (TELEVIC)

# ==============================
# Load all TELEVIC FINAL errors
# ==============================

def load_err(method, snr):
    fname = f"loc_errors_FINAL__televic_{method}_{snr}dB.npy"
    return np.load(fname)

methods = ["conventional", "sparsity1", "sparsity2", "broadside", "random"]
method_labels = [
    "Conventional",
    "Norm-based",
    "Kurtosis-based",
    "Zero-lag CC-based",
    "Random"
]
colors = {
    'Conventional': 'tab:purple',
    'Norm-based': 'tab:blue',
    'Kurtosis-based': 'tab:orange',
    'Zero-lag CC-based': 'tab:green',
    'Random': 'tab:red'
}

snrs = [0, 6, 12]

# Load all error matrices
errors = {m: {s: load_err(m, s) for s in snrs} for m in methods}

# ==============================
# Compute medians
# ==============================

medians = {
    m: [np.median(errors[m][s]) for s in snrs]
    for m in methods
}

print("Medians:")
for m, label in zip(methods, method_labels):
    print(f"{label}: {medians[m]}")

# ==============================
# Plot
# ==============================

fig, ax = plt.subplots()

for m, label in zip(methods, method_labels):
    ax.plot(
        snrs,
        medians[m],
        marker="o",
        linewidth=1.8,
        markersize=5,
        label=label,
        color=colors[label]
    )

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Median Localization Error (m)")
ax.set_xticks(snrs)

ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()

# For the paper
fig.savefig("televic_median_errors_2.png", dpi=400, bbox_inches="tight")
# fig.savefig("televic_median_errors.pdf", bbox_inches="tight")

plt.show()





