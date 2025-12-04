/*
 * This is the grid helper utility header
 * This defines all the function implementations we need to define a 3D grid (set z to 1 for 2D)
 */

#ifndef GRID_H
#define GRID_H

// Structure to represent a 3D computational grid (set nz=1 for 2D)
typedef struct {
    int nx;           // Number of grid points in x-direction
    int ny;           // Number of grid points in y-direction
    int nz;           // Number of grid points in z-direction
    double dx;        // Grid spacing in x-direction
    double dy;        // Grid spacing in y-direction
    double dz;        // Grid spacing in z-direction
    double Lx;        // Domain length in x-direction
    double Ly;        // Domain length in y-direction
    double Lz;        // Domain length in z-direction
    double *data;     // 1D array storing grid values, index: k*nx*ny + j*nx + i
} Grid;

// Create and allocate memory for a 3D grid (set nz=1 for 2D)
Grid* gridCreate(int nx, int ny, int nz, double Lx, double Ly, double Lz);

// Free memory allocated for a grid
void gridDestroy(Grid *grid);

// Initialize grid with zeros
void gridZero(Grid *grid);

// Copy data from one grid to another
void gridCopy(const Grid *src, Grid *dst);

// Set a specific grid point value
void gridSet(Grid *grid, int i, int j, int k, double value);

// Get a specific grid point value
double gridGet(const Grid *grid, int i, int j, int k);

// Initialize grid with a Gaussian distribution
void gridInitGaussian(Grid *grid, double x0, double y0, double z0, double sigma, double amplitude);

// Print grid information
void gridPrintInfo(const Grid *grid);

#endif // GRID_H
