/*
 * Cuda header file for GPU CFD solver
 * Declarations of CUDA kernels and host functions
 */

#ifndef SOLVER_CUDA_H
#define SOLVER_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "parameters.h"

#ifdef __cplusplus
}
#endif

// CUDA kernel declarations
__global__ void advectionKernel(double *u, double *u_new, int nx, int ny, int nz,
                                double dx, double dy, double dz, double dt,
                                double vx, double vy, double vz);

__global__ void diffusionKernel(double *u, double *u_new, int nx, int ny, int nz,
                                double dx, double dy, double dz, double dt, double D);

__global__ void applyBoundaryConditions(double *u, int nx, int ny, int nz,
                                       BoundaryType bc_type);

// Host functions
void solverStepCUDA(double *d_u, double *d_u_new, Parameters *params,
                    dim3 gridDim, dim3 blockDim);

#endif
