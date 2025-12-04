/*
 * CUDA implementation of the sequential scene from main.c
 * This code initializes a Gaussian concentration field and evolves it over time
 * using GPU acceleration. The simulation parameters are read from an XML file,
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "parameters.h"
#include "io.h"
#include "grid.h"
}

#include "solver_cuda.h"

#define IDX(i,j,k,nx,ny) ((k)*(nx)*(ny) + (j)*(nx) + (i))

// Initialize Gaussian on host
void initializeGaussian(double *h_u, int nx, int ny, int nz,
                       double Lx, double Ly, double Lz) {
    double dx = Lx / (nx - 1);
    double dy = Ly / (ny - 1);
    double dz = (nz > 1) ? (Lz / (nz - 1)) : 0.0;  // Avoid divide by zero for 2D case or suffer NAN train

    // Initialize to zero
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                h_u[IDX(i, j, k, nx, ny)] = 0.0;
            }
        }
    }

    // Gaussian parameters
    double x0 = Lx * 0.25;
    double y0 = Ly * 0.25;
    double z0 = (nz > 1) ? Lz * 0.25 : 0.0;
    double sigma = Lx * 0.1;
    double amplitude = (nz > 1) ? 50000.0 : 10000.0;

    // Add Gaussian blob
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;

                double r2 = (x - x0)*(x - x0) + (y - y0)*(y - y0) + (z - z0)*(z - z0);
                h_u[IDX(i, j, k, nx, ny)] = amplitude * exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse parameters from the xml config file
    Parameters *params = paramsReadFromFile(argv[1]);
    if (params == NULL) {
        fprintf(stderr, "Failed to read parameters from %s\n", argv[1]);
        return 1;
    }

    int nx = params->nx;
    int ny = params->ny;
    int nz = params->nz;
    size_t size = nx * ny * nz * sizeof(double);

    int n_steps = (int)(params->t_final / params->dt);
    printf("Running CUDA simulation: %d steps\n", n_steps);
    printf("Grid: %dx%dx%d (%zu points)\n", nx, ny, nz, (size_t)(nx*ny*nz));

    // Create temporary grid for I/O (reuse existing Grid structure)
    Grid *grid_io = gridCreate(nx, ny, nz, params->Lx, params->Ly, params->Lz);

    // Allocate host memory for computation
    double *h_u = (double*)malloc(size);

    // Initialize
    initializeGaussian(h_u, nx, ny, nz, params->Lx, params->Ly, params->Lz);

    // Allocate device memory
    double *d_u, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);

    // Copy initial data to device
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemset(d_u_new, 0, size); // Initialize d_u_new to zero or face the nan or inf apocaplypse

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during initialization: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Setup CUDA grid/block dimensions
    // This ensures fair performance comparison between dimensionalities
    dim3 blockDim(16, 16, 4);  // Setting to (16 x 16 x 4 threads) 1024 threads/block which apparantly is the max thread limit

    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);

    printf("Using CUDA block size: (%d, %d, %d) = %d threads/block\n",
           blockDim.x, blockDim.y, blockDim.z,
           blockDim.x * blockDim.y * blockDim.z);

    printf("CUDA grid: (%d, %d, %d) blocks, (%d, %d, %d) threads/block\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);

    // Create output directory as specified in input params
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", params->output_dir);
    system(mkdir_cmd);

    // Write initial condition
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
    memcpy(grid_io->data, h_u, size);
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/output_0000.vtk", params->output_dir);
    ioWriteVTK(grid_io, filename, "concentration");

    // Time the simulation
    clock_t start = clock();

    // Main time-stepping loop
    int output_count = 1;
    for (int step = 1; step <= n_steps; step++) {
        // Solve on GPU (result ends up in d_u)
        solverStepCUDA(d_u, d_u_new, params, gridDim, blockDim);

        // Write sim output at intervals specified in xml params
        if (step % params->output_interval == 0) {
            cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
            memcpy(grid_io->data, h_u, size);

            // Write VTK sim out file
            snprintf(filename, sizeof(filename), "%s/output_%04d.vtk",
                    params->output_dir, output_count);
            ioWriteVTK(grid_io, filename, "concentration");
            output_count++;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nSimulation time: %.3f seconds\n", elapsed);
    printf("Output: %s/ (%d files)\n", params->output_dir, output_count);

    // Cleanup
    free(h_u);
    cudaFree(d_u);
    cudaFree(d_u_new);
    gridDestroy(grid_io);
    paramsDestroy(params);

    return 0;
}
