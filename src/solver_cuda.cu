/*
 * CUDA solver implementation for advection-diffusion equation
 * Advection kernel uses upwind scheme
 * Diffusion kernel uses central differences
 * Boundary conditions handled in separate kernel
 */

 #include<stdio.h>

extern "C" {
#include "parameters.h"
}

#include "solver_cuda.h"

// Macro for 3D indexing
#define IDX(i,j,k,nx,ny) ((k)*(nx)*(ny) + (j)*(nx) + (i))

// Advection kernel using upwind scheme
__global__ void advectionKernel(double *u, double *u_new, int nx, int ny, int nz,
                                double dx, double dy, double dz, double dt,
                                double vx, double vy, double vz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = IDX(i, j, k, nx, ny);

    // Interior points only - boundaries handled separately (handle 2D case where nz=1)
    bool interior_x = (i > 0 && i < nx-1);
    bool interior_y = (j > 0 && j < ny-1);
    bool interior_z = (nz > 1) ? (k > 0 && k < nz-1) : true;  // For 2D, k=0 is interior

    if (interior_x && interior_y && interior_z) {
        double u_here = u[idx];

        // Upwind scheme for x-direction
        double du_dx;
        if (vx > 0) {
            du_dx = (u_here - u[IDX(i-1, j, k, nx, ny)]) / dx;
        } else {
            du_dx = (u[IDX(i+1, j, k, nx, ny)] - u_here) / dx;
        }

        // Upwind scheme for y-direction
        double du_dy;
        if (vy > 0) {
            du_dy = (u_here - u[IDX(i, j-1, k, nx, ny)]) / dy;
        } else {
            du_dy = (u[IDX(i, j+1, k, nx, ny)] - u_here) / dy;
        }

        // Upwind scheme for z-direction
        double du_dz;
        if (nz > 1) {
            if (vz > 0) {
                du_dz = (u_here - u[IDX(i, j, k-1, nx, ny)]) / dz;
            } else {
                du_dz = (u[IDX(i, j, k+1, nx, ny)] - u_here) / dz;
            }
        } else {
            du_dz = 0.0;
        }

        // Update: u_new = u - dt * (vx*du/dx + vy*du/dy + vz*du/dz)
        u_new[idx] = u_here - dt * (vx * du_dx + vy * du_dy + vz * du_dz);
    } else {
        // Copy boundary values as-is
        u_new[idx] = u[idx];
    }
}

// Diffusion kernel using central differences
__global__ void diffusionKernel(double *u, double *u_new, int nx, int ny, int nz,
                                double dx, double dy, double dz, double dt, double D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = IDX(i, j, k, nx, ny);

    // Interior points only (handle 2D case where nz=1)
    bool interior_x = (i > 0 && i < nx-1);
    bool interior_y = (j > 0 && j < ny-1);
    bool interior_z = (nz > 1) ? (k > 0 && k < nz-1) : true;  // For 2D, k=0 is interior

    if (interior_x && interior_y && interior_z) {
        double u_here = u[idx];

        // Second derivatives using central differences
        double d2u_dx2 = (u[IDX(i+1, j, k, nx, ny)] - 2.0*u_here + u[IDX(i-1, j, k, nx, ny)]) / (dx*dx);
        double d2u_dy2 = (u[IDX(i, j+1, k, nx, ny)] - 2.0*u_here + u[IDX(i, j-1, k, nx, ny)]) / (dy*dy);

        double d2u_dz2 = 0.0;
        if (nz > 1) {
            d2u_dz2 = (u[IDX(i, j, k+1, nx, ny)] - 2.0*u_here + u[IDX(i, j, k-1, nx, ny)]) / (dz*dz);
        }

        // Update: u_new = u + dt * D * (d2u/dx2 + d2u/dy2 + d2u/dz2)
        u_new[idx] = u_here + dt * D * (d2u_dx2 + d2u_dy2 + d2u_dz2);
    } else {
        // Copy boundary values as-is
        u_new[idx] = u[idx];
    }
}

// Boundary condition kernel (periodic boundaries)
__global__ void applyBoundaryConditions(double *u, int nx, int ny, int nz,
                                       BoundaryType bc_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    if (bc_type == BC_PERIODIC) {
        // Periodic: wrap around boundaries
        // X boundaries
        if (i == 0) {
            u[IDX(0, j, k, nx, ny)] = u[IDX(nx-2, j, k, nx, ny)];
        }
        if (i == nx-1) {
            u[IDX(nx-1, j, k, nx, ny)] = u[IDX(1, j, k, nx, ny)];
        }

        // Y boundaries
        if (j == 0) {
            u[IDX(i, 0, k, nx, ny)] = u[IDX(i, ny-2, k, nx, ny)];
        }
        if (j == ny-1) {
            u[IDX(i, ny-1, k, nx, ny)] = u[IDX(i, 1, k, nx, ny)];
        }

        // Z boundaries (if 3D)
        if (nz > 1) {
            if (k == 0) {
                u[IDX(i, j, 0, nx, ny)] = u[IDX(i, j, nz-2, nx, ny)];
            }
            if (k == nz-1) {
                u[IDX(i, j, nz-1, nx, ny)] = u[IDX(i, j, 1, nx, ny)];
            }
        }
    }
    else if (bc_type == BC_DIRICHLET) {
        // Dirichlet: fixed value (zero) at boundaries
        bool is_boundary = false;

        // Check if this point is on any boundary
        if (i == 0 || i == nx-1) is_boundary = true;
        if (j == 0 || j == ny-1) is_boundary = true;
        if (nz > 1 && (k == 0 || k == nz-1)) is_boundary = true;

        if (is_boundary) {
            u[IDX(i, j, k, nx, ny)] = 0.0;  // Fixed value at boundary
        }
    }
    else if (bc_type == BC_NEUMANN) {
        // Neumann: zero gradient (∂u/∂n = 0) - copy from nearest interior
        // X boundaries
        if (i == 0) {
            u[IDX(0, j, k, nx, ny)] = u[IDX(1, j, k, nx, ny)];
        }
        if (i == nx-1) {
            u[IDX(nx-1, j, k, nx, ny)] = u[IDX(nx-2, j, k, nx, ny)];
        }

        // Y boundaries
        if (j == 0) {
            u[IDX(i, 0, k, nx, ny)] = u[IDX(i, 1, k, nx, ny)];
        }
        if (j == ny-1) {
            u[IDX(i, ny-1, k, nx, ny)] = u[IDX(i, ny-2, k, nx, ny)];
        }

        // Z boundaries
        if (nz > 1) {
            if (k == 0) {
                u[IDX(i, j, 0, nx, ny)] = u[IDX(i, j, 1, nx, ny)];
            }
            if (k == nz-1) {
                u[IDX(i, j, nz-1, nx, ny)] = u[IDX(i, j, nz-2, nx, ny)];
            }
        }
    }
}

// Host function to facilitate the solver step
void solverStepCUDA(double *d_u, double *d_u_new, Parameters *params,
                    dim3 gridDim, dim3 blockDim) {
    int nx = params->nx;
    int ny = params->ny;
    int nz = params->nz;
    double dx = params->Lx / (nx - 1);
    double dy = params->Ly / (ny - 1);
    double dz = params->Lz / (nz - 1);

    // Apply boundary conditions to input
    applyBoundaryConditions<<<gridDim, blockDim>>>(d_u, nx, ny, nz, params->bc_type);
    cudaDeviceSynchronize();

    // Advection (d_u -> d_u_new)
    advectionKernel<<<gridDim, blockDim>>>(d_u, d_u_new, nx, ny, nz,
                                           dx, dy, dz, params->dt,
                                           params->vx, params->vy, params->vz);
    cudaDeviceSynchronize();

    // Apply boundary conditions after advection
    applyBoundaryConditions<<<gridDim, blockDim>>>(d_u_new, nx, ny, nz, params->bc_type);
    cudaDeviceSynchronize();

    // Diffusion (d_u_new -> d_u)
    diffusionKernel<<<gridDim, blockDim>>>(d_u_new, d_u, nx, ny, nz,
                                           dx, dy, dz, params->dt, params->D);
    cudaDeviceSynchronize();

    // Apply boundary conditions to final result
    applyBoundaryConditions<<<gridDim, blockDim>>>(d_u, nx, ny, nz, params->bc_type);
    cudaDeviceSynchronize();

    // Note: Result is now in d_u, caller handles pointer swap
}
