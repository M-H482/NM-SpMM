import matplotlib.pyplot as plt
import numpy as np

# Function to read data from a file
def read_data(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        sparsity = None
        for line in lines:
            line = line.strip()
            if line.startswith('sparsity'):
                sparsity = line.split('=')[1].strip()
                data[sparsity] = []
            elif sparsity:
                data[sparsity].append(float(line))
    return data

# File name containing the data
filename = 'data.txt'

# Reading data from the file
data = read_data(filename)
sparsity_levels = list(data.keys())

# Setting up the plot
x = np.arange(len(sparsity_levels))  # The label locations
width = 0.15  # The width of the bars

# Define colors for each version
# colors = ['#9CD1CB', '#C6B3D3', '#ED9F9B', '#80BA8A', '#00B0F0']
colors = ['#F4B183', '#FFD966', '#B69DC7', '#7CC2BA', '#00B0F0']

# Define font sizes
label_fontsize = 28
title_fontsize = 28
legend_fontsize = 24
tick_fontsize = 28

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting bars for each version with custom colors and labels
bar_labels = ["V0 naive", "V1 block tiling", "V2 warp&thread tiling", "V3 sparsity aware", "V4 double buffering"]
for i, (label, color) in enumerate(zip(bar_labels, colors)):
    values = [data[sp][i] for sp in sparsity_levels]
    ax.bar(x + i * width, values, width, label=label, color=color)

# Adding labels and title with custom font sizes
ax.set_xlabel('Sparsity Ratios', fontsize=label_fontsize)
ax.set_ylabel('Effective TFLOPS', fontsize=label_fontsize)
# ax.set_title('Step-wise Performance Analysis by Sparsity and Version', fontsize=title_fontsize)
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(sparsity_levels, fontsize=tick_fontsize)
ax.legend(title="Versions", fontsize=legend_fontsize, title_fontsize=legend_fontsize)

# Set y-axis tick font size
ax.tick_params(axis='y', labelsize=tick_fontsize)

# Adding a grid for better readability
ax.grid(True, linestyle='--', alpha=0.6)

# Save the plot as a PDF file
plt.tight_layout()
plt.savefig('step-wise_performance_analysis.pdf', format='pdf')

# Optionally, you can show the plot
# plt.show()
