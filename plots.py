import matplotlib.pyplot as plt

grid_sizes = [512, 1024, 2048, 4096, 8192]
g_CUDA2_CPU_times = [10.8, 24, 67.3, 215.3, 1018.2]
g_CUDA2_GPU_times = [0.1, 0.3, 1.2, 5.1, 18.8]
g_CUDA2_speedup = []

for i in range(len(g_CUDA2_CPU_times)):
    g_CUDA2_speedup.append(g_CUDA2_CPU_times[i]/g_CUDA2_GPU_times[i])

g_CUDA3_CPU_times = [11.4, 25.3, 68.8, 220.2, 775.8]
g_CUDA3_GPU_times = [0.1, 0.16, 0.39, 1.2, 4.6]
g_CUDA3_1GPU_times = [0.1,0.17, 0.58, 2.2, 9.0]
g_CUDA3_speedup = []

for i in range(len(g_CUDA3_CPU_times)):
    g_CUDA3_speedup.append(g_CUDA3_CPU_times[i]/g_CUDA3_GPU_times[i])

iteration_sizes = [1000, 10000, 50000, 100000, 200000]
i_CUDA2_CPU_times = [1.5, 24.0, 172.0, 214.3, 306.4]
i_CUDA2_GPU_times = [0.04, 0.3, 1.6, 3.2, 11.3]
i_CUDA2_speedup = []

for i in range(len(i_CUDA2_CPU_times)):
    i_CUDA2_speedup.append(i_CUDA2_CPU_times[i]/i_CUDA2_GPU_times[i])

i_CUDA3_CPU_times = [1.6, 25.3, 177.2, 224.6, 320.4]
i_CUDA3_GPU_times = [0.016, 0.16, 0.85, 1.52, 2.9]
i_CUDA3_speedup = []

for i in range(len(i_CUDA3_CPU_times)):
    i_CUDA3_speedup.append(i_CUDA3_CPU_times[i]/i_CUDA3_GPU_times[i])


b_CUDA2_CPU_times = [10.8, 24, 67.3, 215.3, 1018.2]
b_CUDA2_GPU_8x8   = [0.1, 0.3, 1.2, 5.1, 20.1]
b_CUDA2_GPU_16x16 = [0.1, 0.3, 1.2, 5.1, 20.1]
b_CUDA2_GPU_32x32 = [0.1, 0.35, 1.3, 5.2, 20.3]





def plot_speedup_vs_blocksize():
    import numpy as np

    # Speedup = CPU time / GPU time
    b_CUDA2_speedup_8x8 = [cpu / gpu for cpu, gpu in zip(b_CUDA2_CPU_times, b_CUDA2_GPU_8x8)]
    b_CUDA2_speedup_16x16 = [cpu / gpu for cpu, gpu in zip(b_CUDA2_CPU_times, b_CUDA2_GPU_16x16)]
    b_CUDA2_speedup_32x32 = [cpu / gpu for cpu, gpu in zip(b_CUDA2_CPU_times, b_CUDA2_GPU_32x32)]

    bar_width = 0.25
    index = np.arange(len(grid_sizes))

    plt.figure()
    plt.bar(index, b_CUDA2_speedup_8x8, bar_width, label='Block 8x8')
    plt.bar(index + bar_width, b_CUDA2_speedup_16x16, bar_width, label='Block 16x16')
    plt.bar(index + 2 * bar_width, b_CUDA2_speedup_32x32, bar_width, label='Block 32x32')

    plt.xlabel('Grid Size', fontsize=20)
    plt.ylabel('Speedup (CPU time / GPU time)', fontsize=20)
    plt.title('Cuda2 Speedup vs Grid Size for Different Block Sizes', fontsize=20)
    plt.xticks(index + bar_width, grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()





def plot_speedup_vs_gridsize():
    import numpy as np
    bar_width = 0.35
    index = np.arange(len(grid_sizes))

    plt.figure()
    plt.bar(index, g_CUDA2_speedup, bar_width, label='CUDA2 Speedup')
    plt.bar(index + bar_width, g_CUDA3_speedup, bar_width, label='CUDA3 Speedup')
    plt.xlabel('Grid Size', fontsize=20)
    plt.ylabel('Speedup (CPU time / GPU time)', fontsize=20)
    plt.title('Speedup vs Grid Size', fontsize=20)
    plt.xticks(index + bar_width / 2, grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_speedup_vs_iterations():
    import numpy as np
    bar_width = 0.35
    index = np.arange(len(iteration_sizes))

    plt.figure()
    plt.bar(index, i_CUDA2_speedup, bar_width, label='CUDA2 Speedup')
    plt.bar(index + bar_width, i_CUDA3_speedup, bar_width, label='CUDA3 Speedup')
    plt.xlabel('Number of Iterations', fontsize=20)
    plt.ylabel('Speedup (CPU time / GPU time)', fontsize=20)
    plt.title('Speedup vs Number of Iterations', fontsize=20)
    plt.xticks(index + bar_width / 2, iteration_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_CUDA3_speedup_1GPU_vs_2GPU():
    import numpy as np

    # Speedup = CPU time / GPU time
    speedup_1GPU = [cpu / gpu for cpu, gpu in zip(g_CUDA3_CPU_times, g_CUDA3_1GPU_times)]
    speedup_2GPU = [cpu / gpu for cpu, gpu in zip(g_CUDA3_CPU_times, g_CUDA3_GPU_times)]

    bar_width = 0.35
    index = np.arange(len(grid_sizes))

    plt.figure()
    plt.bar(index, speedup_1GPU, bar_width, label='CUDA3 - 1 GPU')
    plt.bar(index + bar_width, speedup_2GPU, bar_width, label='CUDA3 - 2 GPUs')

    plt.xlabel('Grid Size', fontsize=20)
    plt.ylabel('Speedup (CPU time / GPU time)', fontsize=20)
    plt.title('CUDA3 Speedup: 1 GPU vs 2 GPUs', fontsize=20)
    plt.xticks(index + bar_width / 2, grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

def timeplot():
    import numpy as np

    plt.figure()
    plt.plot(grid_sizes, g_CUDA2_CPU_times, marker='o', linewidth=2, label='CUDA2 CPU Time', color="C2")
    plt.xscale('log', base=2)  # Log-skala base 2 for grid sizes
    plt.xlabel('Grid Size (log_2 scale)', fontsize=20)
    plt.ylabel('CPU Time [s]', fontsize=20)
    plt.xticks(grid_sizes, labels=grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', which='both')
    plt.tight_layout()
    plt.show()

def timeplot2():
    import numpy as np

    plt.figure()
    plt.plot(grid_sizes, g_CUDA3_CPU_times, marker='o', linewidth=2, label='CUDA3 CPU Time', color="C3")
    plt.xscale('log', base=2)  # Log-skala base 2 for grid sizes
    plt.xlabel('Grid Size (log_2 scale)', fontsize=20)
    plt.ylabel('CPU Time [s]', fontsize=20)
    plt.xticks(grid_sizes, labels=grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', which='both')
    plt.tight_layout()
    plt.show()

def plot_relative_speedup_2GPU_vs_1GPU():
    import numpy as np

    # CUDA3 speedups
    speedup_1GPU = [cpu / gpu for cpu, gpu in zip(g_CUDA3_CPU_times, g_CUDA3_1GPU_times)]
    speedup_2GPU = [cpu / gpu for cpu, gpu in zip(g_CUDA3_CPU_times, g_CUDA3_GPU_times)]

    # Relativ speedup = speedup med 2 GPU / speedup med 1 GPU
    relative_speedup = [s2 / s1 for s2, s1 in zip(speedup_2GPU, speedup_1GPU)]

    plt.figure()
    plt.plot(grid_sizes, relative_speedup, marker='o', linewidth=2, label='2 GPUs vs 1 GPU')

    plt.xscale('log', base=2)
    plt.xlabel('Grid Size (log scale)', fontsize=20)
    plt.ylabel('Relative Speedup (2 GPU / 1 GPU)', fontsize=20)
    plt.xticks(grid_sizes, labels=grid_sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', which='both')
    plt.tight_layout()
    plt.show()



    





# plot_speedup_vs_gridsize()
# plot_speedup_vs_iterations()
plot_speedup_vs_blocksize()
# plot_CUDA3_speedup_1GPU_vs_2GPU()
# timeplot()
# timeplot2()
# plot_relative_speedup_2GPU_vs_1GPU()
