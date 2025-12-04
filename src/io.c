/*
 * I/O implementation for VTK file writing
 * Uses VTK ASCII format for STRUCTURED_POINTS dataset
 */

#include "../include/io.h"
#include <stdio.h>

// Write grid to VTK STRUCTURED_POINTS format
int ioWriteVTK(const Grid *grid, const char *filename, const char *scalar_name) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return -1;
    }

    // VTK file header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Advection-Diffusion Simulation Output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");

    // Grid dimensions
    fprintf(fp, "DIMENSIONS %d %d %d\n", grid->nx, grid->ny, grid->nz);

    // Origin (lower-left corner)
    fprintf(fp, "ORIGIN 0.0 0.0 0.0\n");

    // Spacing
    fprintf(fp, "SPACING %.8e %.8e %.8e\n", grid->dx, grid->dy, grid->dz);

    // Point data header
    int total_points = grid->nx * grid->ny * grid->nz;
    fprintf(fp, "POINT_DATA %d\n", total_points);
    fprintf(fp, "SCALARS %s float 1\n", scalar_name);
    fprintf(fp, "LOOKUP_TABLE default\n");

    // Write data in VTK order (k varies fastest in STRUCTURED_POINTS... actually i j k)
    // VTK STRUCTURED_POINTS expects data in i-j-k order (x varies fastest)
    for (int k = 0; k < grid->nz; k++) {
        for (int j = 0; j < grid->ny; j++) {
            for (int i = 0; i < grid->nx; i++) {
                double value = gridGet(grid, i, j, k);
                fprintf(fp, "%.8e\n", value);
            }
        }
    }

    fclose(fp);
    return 0;
}
