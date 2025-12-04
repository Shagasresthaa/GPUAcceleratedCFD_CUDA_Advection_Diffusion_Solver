/*
 * Solver implementation for 3D advection-diffusion equation
 * Uses upwind scheme for advection and central differences for diffusion
 * Applies any of the three specified boundary conditions
 */

#include "../include/solver.h"
#include <math.h>

// Apply boundary conditions to a grid
void solverApplyBoundary(Grid *grid, BoundaryType bc_type) {
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;

    switch(bc_type) {
        case BC_PERIODIC:
            // X-direction periodic
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < ny; j++) {
                    gridSet(grid, 0, j, k, gridGet(grid, nx-2, j, k));
                    gridSet(grid, nx-1, j, k, gridGet(grid, 1, j, k));
                }
            }
            // Y-direction periodic
            for (int k = 0; k < nz; k++) {
                for (int i = 0; i < nx; i++) {
                    gridSet(grid, i, 0, k, gridGet(grid, i, ny-2, k));
                    gridSet(grid, i, ny-1, k, gridGet(grid, i, 1, k));
                }
            }
            // Z-direction periodic
            if (nz > 1) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        gridSet(grid, i, j, 0, gridGet(grid, i, j, nz-2));
                        gridSet(grid, i, j, nz-1, gridGet(grid, i, j, 1));
                    }
                }
            }
            break;

        case BC_DIRICHLET:
            // Set boundaries to zero
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < ny; j++) {
                    gridSet(grid, 0, j, k, 0.0);
                    gridSet(grid, nx-1, j, k, 0.0);
                }
            }
            for (int k = 0; k < nz; k++) {
                for (int i = 0; i < nx; i++) {
                    gridSet(grid, i, 0, k, 0.0);
                    gridSet(grid, i, ny-1, k, 0.0);
                }
            }
            if (nz > 1) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        gridSet(grid, i, j, 0, 0.0);
                        gridSet(grid, i, j, nz-1, 0.0);
                    }
                }
            }
            break;

        case BC_NEUMANN:
            // Zero gradient at boundaries
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < ny; j++) {
                    gridSet(grid, 0, j, k, gridGet(grid, 1, j, k));
                    gridSet(grid, nx-1, j, k, gridGet(grid, nx-2, j, k));
                }
            }
            for (int k = 0; k < nz; k++) {
                for (int i = 0; i < nx; i++) {
                    gridSet(grid, i, 0, k, gridGet(grid, i, 1, k));
                    gridSet(grid, i, ny-1, k, gridGet(grid, i, ny-2, k));
                }
            }
            if (nz > 1) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        gridSet(grid, i, j, 0, gridGet(grid, i, j, 1));
                        gridSet(grid, i, j, nz-1, gridGet(grid, i, j, nz-2));
                    }
                }
            }
            break;
    }
}

// Advection step using upwind scheme
void solverAdvectionStep(const Grid *u_old, Grid *u_new, const Parameters *params) {
    int nx = u_old->nx;
    int ny = u_old->ny;
    int nz = u_old->nz;
    double dx = u_old->dx;
    double dy = u_old->dy;
    double dz = u_old->dz;
    double dt = params->dt;
    double vx = params->vx;
    double vy = params->vy;
    double vz = params->vz;

    // Copy boundary points first
    gridCopy(u_old, u_new);

    // Determine loop bounds
    int k_start = (nz > 1) ? 1 : 0;
    int k_end = (nz > 1) ? nz-1 : 1;

    // Interior points using upwind scheme
    for (int k = k_start; k < k_end; k++) {
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                double u_ijk = gridGet(u_old, i, j, k);

                // Upwind differences for advection
                double dudx, dudy, dudz;

                // X-direction
                if (vx >= 0.0) {
                    dudx = (u_ijk - gridGet(u_old, i-1, j, k)) / dx;
                } else {
                    dudx = (gridGet(u_old, i+1, j, k) - u_ijk) / dx;
                }

                // Y-direction
                if (vy >= 0.0) {
                    dudy = (u_ijk - gridGet(u_old, i, j-1, k)) / dy;
                } else {
                    dudy = (gridGet(u_old, i, j+1, k) - u_ijk) / dy;
                }

                // Z-direction
                if (nz > 1) {
                    if (vz >= 0.0) {
                        dudz = (u_ijk - gridGet(u_old, i, j, k-1)) / dz;
                    } else {
                        dudz = (gridGet(u_old, i, j, k+1) - u_ijk) / dz;
                    }
                } else {
                    dudz = 0.0;
                }

                // Update: u_new = u_old - dt*(vx*du/dx + vy*du/dy + vz*du/dz)
                double u_new_val = u_ijk - dt * (vx * dudx + vy * dudy + vz * dudz);
                gridSet(u_new, i, j, k, u_new_val);
            }
        }
    }
}

// Diffusion step using explicit central differences
void solverDiffusionStep(const Grid *u_old, Grid *u_new, const Parameters *params) {
    int nx = u_old->nx;
    int ny = u_old->ny;
    int nz = u_old->nz;
    double dx = u_old->dx;
    double dy = u_old->dy;
    double dz = u_old->dz;
    double dt = params->dt;
    double D = params->D;

    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double dz2 = dz * dz;

    // Copy boundary points first
    gridCopy(u_old, u_new);

    // Determine loop bounds
    int k_start = (nz > 1) ? 1 : 0;
    int k_end = (nz > 1) ? nz-1 : 1;

    // Interior points using central differences for Laplacian
    for (int k = k_start; k < k_end; k++) {
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                double u_ijk = gridGet(u_old, i, j, k);

                // Second derivatives (Laplacian)
                double d2udx2 = (gridGet(u_old, i+1, j, k) - 2.0*u_ijk + gridGet(u_old, i-1, j, k)) / dx2;
                double d2udy2 = (gridGet(u_old, i, j+1, k) - 2.0*u_ijk + gridGet(u_old, i, j-1, k)) / dy2;

                double d2udz2 = 0.0;
                if (nz > 1) {
                    d2udz2 = (gridGet(u_old, i, j, k+1) - 2.0*u_ijk + gridGet(u_old, i, j, k-1)) / dz2;
                }

                // Update: u_new = u_old + dt*D*(d2u/dx2 + d2u/dy2 + d2u/dz2)
                double u_new_val = u_ijk + dt * D * (d2udx2 + d2udy2 + d2udz2);
                gridSet(u_new, i, j, k, u_new_val);
            }
        }
    }
}

// Combined advection-diffusion step
void solverStep(const Grid *u_old, Grid *u_new, const Parameters *params) {
    // Create temporary grid for intermediate step
    Grid *u_temp = gridCreate(u_old->nx, u_old->ny, u_old->nz,
                              u_old->Lx, u_old->Ly, u_old->Lz);

    // Copy old values to temp
    gridCopy(u_old, u_temp);

    // Apply boundary conditions to input
    solverApplyBoundary(u_temp, params->bc_type);

    // Advection (u_temp -> u_new)
    solverAdvectionStep(u_temp, u_new, params);
    solverApplyBoundary(u_new, params->bc_type);

    // Diffusion (u_new -> u_temp)
    gridCopy(u_new, u_temp);
    solverDiffusionStep(u_temp, u_new, params);
    solverApplyBoundary(u_new, params->bc_type);

    gridDestroy(u_temp);
}

// Calculate L2 norm of a grid
double solverCalcL2Norm(const Grid *grid) {
    double sum = 0.0;
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double val = gridGet(grid, i, j, k);
                sum += val * val;
            }
        }
    }

    return sqrt(sum / (nx * ny * nz));
}

// Calculate maximum value in grid
double solverCalcMax(const Grid *grid) {
    double max_val = gridGet(grid, 0, 0, 0);
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double val = gridGet(grid, i, j, k);
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    return max_val;
}

// Calculate minimum value in grid
double solverCalcMin(const Grid *grid) {
    double min_val = gridGet(grid, 0, 0, 0);
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double val = gridGet(grid, i, j, k);
                if (val < min_val) {
                    min_val = val;
                }
            }
        }
    }

    return min_val;
}
