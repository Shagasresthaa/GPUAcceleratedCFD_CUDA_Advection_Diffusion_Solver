import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Timing data from the run logs (in seconds)
seq_2d_dirichlet = [0.861, 6.063, 16.470]
seq_2d_neumann = [0.853, 6.095, 16.846]
seq_2d_periodic = [0.803, 5.943, 16.825]

seq_3d_dirichlet = [38.311, 1250.919, 5778.289]
seq_3d_neumann = [39.103, 1345.500, 5734.996]
seq_3d_periodic = [38.717, 1274.598, 5675.441]

cuda_2d_dirichlet = [0.848, 2.261, 5.184]
cuda_2d_neumann = [0.833, 2.308, 5.128]
cuda_2d_periodic = [0.850, 2.240, 5.139]

cuda_3d_dirichlet = [10.277, 227.651, 595.165]
cuda_3d_neumann = [10.768, 233.716, 597.419]
cuda_3d_periodic = [10.617, 232.228, 602.196]

grid_sizes_2d = [50, 150, 250]
grid_sizes_3d = [50, 150, 250]

# GPU configuration from benchmark results
threads_per_block = 1024
cuda_cores = 6144  # RTX 3070 Ti Mobile physical CUDA cores

# 2D: blocks = (bx, by, bz)
blocks_2d = [
    (4, 4, 1),  # 50x50: 16 blocks
    (10, 10, 1),  # 150x150: 100 blocks
    (16, 16, 1),  # 250x250: 256 blocks
]

# 3D: blocks = (bx, by, bz)
blocks_3d = [
    (4, 4, 13),  # 50x50x50: 208 blocks
    (10, 10, 38),  # 150x150x150: 3800 blocks
    (16, 16, 63),  # 250x250x250: 16128 blocks
]

# Calculate speedups
speedup_2d_dirichlet = [seq_2d_dirichlet[i] / cuda_2d_dirichlet[i] for i in range(3)]
speedup_2d_neumann = [seq_2d_neumann[i] / cuda_2d_neumann[i] for i in range(3)]
speedup_2d_periodic = [seq_2d_periodic[i] / cuda_2d_periodic[i] for i in range(3)]

speedup_3d_dirichlet = [seq_3d_dirichlet[i] / cuda_3d_dirichlet[i] for i in range(3)]
speedup_3d_neumann = [seq_3d_neumann[i] / cuda_3d_neumann[i] for i in range(3)]
speedup_3d_periodic = [seq_3d_periodic[i] / cuda_3d_periodic[i] for i in range(3)]

# Calculate efficiency: Speedup / CUDA_cores * 100
efficiency_2d_dirichlet = [speedup_2d_dirichlet[i] / cuda_cores * 100 for i in range(3)]
efficiency_2d_neumann = [speedup_2d_neumann[i] / cuda_cores * 100 for i in range(3)]
efficiency_2d_periodic = [speedup_2d_periodic[i] / cuda_cores * 100 for i in range(3)]

efficiency_3d_dirichlet = [speedup_3d_dirichlet[i] / cuda_cores * 100 for i in range(3)]
efficiency_3d_neumann = [speedup_3d_neumann[i] / cuda_cores * 100 for i in range(3)]
efficiency_3d_periodic = [speedup_3d_periodic[i] / cuda_cores * 100 for i in range(3)]

# 2d Sequential Execution Time
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_2d,
    seq_2d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    seq_2d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    seq_2d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x 1)", fontsize=14)
plt.ylabel("Execution Time (seconds)", fontsize=14)
plt.title("2D Sequential Execution Time", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_2d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/sequential_time_2d.png", dpi=300, bbox_inches="tight")
plt.close()

# 3d Sequential Execution Time
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_3d,
    seq_3d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    seq_3d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    seq_3d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x N)", fontsize=14)
plt.ylabel("Execution Time (seconds)", fontsize=14)
plt.title("3D Sequential Execution Time", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_3d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/sequential_time_3d.png", dpi=300, bbox_inches="tight")
plt.close()

# 2d CUDA Execution Time
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_2d,
    cuda_2d_dirichlet,
    "o-",
    color="#d62728",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    cuda_2d_neumann,
    "s-",
    color="#9467bd",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    cuda_2d_periodic,
    "D-",
    color="#8c564b",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x 1)", fontsize=14)
plt.ylabel("Execution Time (seconds)", fontsize=14)
plt.title("2D CUDA Execution Time", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_2d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/cuda_time_2d.png", dpi=300, bbox_inches="tight")
plt.close()

# 3d CUDA Execution Time
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_3d,
    cuda_3d_dirichlet,
    "o-",
    color="#d62728",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    cuda_3d_neumann,
    "s-",
    color="#9467bd",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    cuda_3d_periodic,
    "D-",
    color="#8c564b",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x N)", fontsize=14)
plt.ylabel("Execution Time (seconds)", fontsize=14)
plt.title("3D CUDA Execution Time", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_3d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/cuda_time_3d.png", dpi=300, bbox_inches="tight")
plt.close()

# 2d Speedup
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_2d,
    speedup_2d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    speedup_2d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    speedup_2d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x 1)", fontsize=14)
plt.ylabel("Speedup (Sequential Time / CUDA Time)", fontsize=14)
plt.title("2D CUDA Speedup", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_2d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/speedup_2d.png", dpi=300, bbox_inches="tight")
plt.close()

# 3d Speedup
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_3d,
    speedup_3d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    speedup_3d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    speedup_3d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x N)", fontsize=14)
plt.ylabel("Speedup (Sequential Time / CUDA Time)", fontsize=14)
plt.title("3D CUDA Speedup", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_3d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/speedup_3d.png", dpi=300, bbox_inches="tight")
plt.close()

# 2d Efficiency
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_2d,
    efficiency_2d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    efficiency_2d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_2d,
    efficiency_2d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x 1)", fontsize=14)
plt.ylabel("Parallel Efficiency (%)", fontsize=14)
plt.title("2D Parallel Efficiency", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_2d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/efficiency_2d.png", dpi=300, bbox_inches="tight")
plt.close()

# 3d Efficiency
plt.figure(figsize=(12, 7))
plt.plot(
    grid_sizes_3d,
    efficiency_3d_dirichlet,
    "o-",
    color="#1f77b4",
    label="Dirichlet",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    efficiency_3d_neumann,
    "s-",
    color="#ff7f0e",
    label="Neumann",
    linewidth=3,
    markersize=12,
)
plt.plot(
    grid_sizes_3d,
    efficiency_3d_periodic,
    "D-",
    color="#2ca02c",
    label="Periodic",
    linewidth=3,
    markersize=12,
)
plt.xlabel("Grid Size (N x N x N)", fontsize=14)
plt.ylabel("Parallel Efficiency (%)", fontsize=14)
plt.title("3D Parallel Efficiency", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.xticks(grid_sizes_3d, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plots/efficiency_3d.png", dpi=300, bbox_inches="tight")
plt.close()
