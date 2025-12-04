/*
 * This is the grid helper utility
 * This defines all the function implementations we need to define a 3D grid (set z to 1 for 2D)
 *
 */

#include "../include/grid.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Create and allocate memory for a 3D grid
Grid* gridCreate(int nx, int ny, int nz, double Lx, double Ly, double Lz) {
    Grid *grid = (Grid*)malloc(sizeof(Grid));

    grid->nx = nx;
    grid->ny = ny;
    grid->nz = nz;
    grid->Lx = Lx;
    grid->Ly = Ly;
    grid->Lz = Lz;
    grid->dx = Lx / (nx - 1);
    grid->dy = Ly / (ny - 1);
    grid->dz = (nz > 1) ? Lz / (nz - 1) : 1.0;

    // Allocate memory for grid data
    grid->data = (double*)malloc(nx * ny * nz * sizeof(double));

    // Initialize to zero
    gridZero(grid);

    return grid;
}

// Free memory allocated for a grid
void gridDestroy(Grid *grid) {
    free(grid->data);
    free(grid);
}

// Initialize grid with zeros
void gridZero(Grid *grid) {
    memset(grid->data, 0, grid->nx * grid->ny * grid->nz * sizeof(double));
}

// Copy data from one grid to another
void gridCopy(const Grid *src, Grid *dst) {
    memcpy(dst->data, src->data, src->nx * src->ny * src->nz * sizeof(double));
}

// Set a specific grid point value
void gridSet(Grid *grid, int i, int j, int k, double value) {
    grid->data[k * grid->nx * grid->ny + j * grid->nx + i] = value;
}

// Get a specific grid point value
double gridGet(const Grid *grid, int i, int j, int k) {
    return grid->data[k * grid->nx * grid->ny + j * grid->nx + i];
}

// Initialize grid with a Gaussian distribution
void gridInitGaussian(Grid *grid, double x0, double y0, double z0, double sigma, double amplitude) {
    for (int k = 0; k < grid->nz; k++) {
        for (int j = 0; j < grid->ny; j++) {
            for (int i = 0; i < grid->nx; i++) {
                double x = i * grid->dx;
                double y = j * grid->dy;
                double z = k * grid->dz;

                double dx = x - x0;
                double dy = y - y0;
                double dz = z - z0;
                double r2 = dx * dx + dy * dy + dz * dz;

                double value = amplitude * exp(-r2 / (2.0 * sigma * sigma));
                gridSet(grid, i, j, k, value);
            }
        }
    }
}

// Debug print grid information
void gridPrintInfo(const Grid *grid) {
    printf("Grid Information:\n");
    printf("  Dimensions: %d x %d x %d\n", grid->nx, grid->ny, grid->nz);
    printf("  Domain size: %.2f x %.2f x %.2f\n", grid->Lx, grid->Ly, grid->Lz);
    printf("  Grid spacing: dx=%.4f, dy=%.4f, dz=%.4f\n", grid->dx, grid->dy, grid->dz);
    printf("  Total points: %d\n", grid->nx * grid->ny * grid->nz);
}
