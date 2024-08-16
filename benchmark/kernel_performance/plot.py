import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据文件
def read_data(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f.readlines()]

# 设置命令行参数
parser = argparse.ArgumentParser(description="Plot speedup comparison")
parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity value for the data files')
parser.add_argument('--yticks', type=str, default='0,1,2', help='Comma-separated list of y-axis ticks for the plots. Example: "0,1,2"')
parser.add_argument('--no-legend', action='store_true', help='Disable the display of the legend')

args = parser.parse_args()
sparsity = args.sparsity

# 解析 y 轴刻度
yticks = list(map(float, args.yticks.split(',')))

# 准备数据和文件名
machines = ['a100', '3090', '4090']
sparsity_percentage = sparsity * 100

file_template = './result_data/{algorithm}_{machine}_{sparsity}.txt'   

# 算法名称和文件字典
algorithms = ['nmspmm', 'nmsparse', 'sputnik', 'cublas']

files = {
    'cublas': {machine: f'./result_data/cublas_{machine}.txt' for machine in machines},
    'sputnik': {machine: file_template.format(machine=machine, algorithm='sputnik', sparsity=sparsity) for machine in machines},
    'nmsparse': {machine: file_template.format(machine=machine, algorithm='nmsparse', sparsity=sparsity) for machine in machines},
    'nmspmm': {machine:  file_template.format(machine=machine, algorithm='nmspmm', sparsity=sparsity) for machine in machines}
}

# 创建包含三个子图的画布，增加宽度
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6.5), sharey=True)

# 读取数据并绘制每个子图
for i, machine in enumerate(machines):
    # 读取数据
    data = {algorithm: read_data(files[algorithm][machine]) for algorithm in algorithms}

    # 计算加速比
    speedups = {
        'nmspmm': [nmspmm / data['cublas'][j] for j, nmspmm in enumerate(data['nmspmm'])],
        'nmsparse': [nmsparse / data['cublas'][j] for j, nmsparse in enumerate(data['nmsparse'])],
        'sputnik': [sputnik / data['cublas'][j] for j, sputnik in enumerate(data['sputnik'])],                
    }

    # 绘制折线图
    x = range(len(data['cublas']))  # 设置横轴从0开始
    colors = {'nmspmm': '#262626', 'nmsparse': '#FF0000', 'sputnik': '#00B0F0', 'cublas': '#7030A0'}
    for algo, speedup in speedups.items():
        axes[i].plot(x, speedup, marker='', label=algo.replace('nmsparse', 'nmSPARSE').replace('nmspmm', 'NM-SpMM').replace('sputnik', 'Sputnik'), color=colors[algo])

    # 添加等于1的横线
    axes[i].axhline(y=1, color=colors['cublas'], linestyle='--', label='cuBLAS')

    # 添加理想情况下的加速比横线
    ideal_speedup = 1 / (1 - sparsity)
    axes[i].axhline(y=ideal_speedup, color='#00B050', linestyle='--', label='Ideal Speedup')

    # 设置横轴标签
    axes[i].set_xlabel('Data Point', fontsize=30)

    # 仅在第一个子图显示纵轴标签
    if i == 0:
        axes[i].set_ylabel('Speedup vs. cuBLAS', fontsize=30)
        # 调整纵轴刻度的数字大小
        axes[i].tick_params(axis='y', labelsize=30, labelcolor='black')
    else:
        axes[i].tick_params(axis='y', labelsize=30, labelcolor='black', labelleft=False)  # 不显示纵轴标签

    # 调整横轴刻度的数字大小
    axes[i].tick_params(axis='x', labelsize=30, labelcolor='black')

    # 设置虚线网格
    axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置 y 轴刻度
    axes[i].set_yticks(yticks)

    # 格式化 y 轴刻度为两位小数
    axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # 在每个子图的右下角添加小方块显示机器名称
    machine_label = machine.upper() if machine == 'a100' else machine
    axes[i].annotate(machine_label, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', fontsize=30, bbox=dict(facecolor='white', edgecolor='black'))

# 确保所有子图的纵轴从0开始
for ax in axes:
    ax.set_ylim(bottom=0)

# 调整布局，给图例留出空间
fig.subplots_adjust(top=0.75, bottom=0.2, left=0.05, right=0.95)

# 添加图例到顶部，如果 --no-legend 参数未设置
if not args.no_legend:
    handles, labels = axes[0].get_legend_handles_labels()

    # 将实线和虚线的图例分开
    line_handles = [handles[i] for i in range(len(handles)) if 'Ideal Speedup' not in labels[i] and 'cuBLAS' not in labels[i]]
    line_labels = [label for label in labels if 'Ideal Speedup' not in label and 'cuBLAS' not in label]

    dashed_handles = [handles[i] for i in range(len(handles)) if 'Ideal Speedup' in labels[i] or 'cuBLAS' in labels[i]]
    dashed_labels = [label for label in labels if 'Ideal Speedup' in label or 'cuBLAS' in label]

    # 合并所有图例项到一行
    all_handles = line_handles + dashed_handles
    all_labels = line_labels + dashed_labels

    fig.legend(all_handles, all_labels, loc='upper center', ncol=len(all_labels), fontsize=30, bbox_to_anchor=(0.5, 0.92),
               handletextpad=0.5, columnspacing=1.0)  # Adjust these values to make labels more compact

# 保存图表为PDF矢量图，裁掉白边
plt.savefig(f'kernel_performance_sparsity_{sparsity_percentage:.1f}.pdf', format='pdf', bbox_inches='tight')

# 显示图表（根据需要可以取消注释）
# plt.show()
