/*
 * Core Sequential Simulation Setup for Advection-Diffusion Equation
 * Initializes a Gaussian pulse and runs the simulation, saving output in VTK format.
 */

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include "../include/grid.h"
#include "../include/parameters.h"
#include "../include/solver.h"
#include "../include/io.h"

int main(int argc, char *argv[]) {
    // Parse parameters from the xml config file
    Parameters *params = paramsReadFromFile(argv[1]);
    if (params == NULL) {
        printf("Error: Failed to read parameters\n");
        return 1;
    }

    // Create output directory
    mkdir(params->output_dir, 0755);

    // Create grids
    Grid *u = gridCreate(params->nx, params->ny, params->nz,
                         params->Lx, params->Ly, params->Lz);
    Grid *u_new = gridCreate(params->nx, params->ny, params->nz,
                             params->Lx, params->Ly, params->Lz);

    // Initialize with Gaussian pulse in corner
    double x0 = params->Lx * 0.25;
    double y0 = params->Ly * 0.25;
    double z0, sigma, amplitude;

    // 3d case
    if (params->nz > 1) {
        z0 = params->Lz * 0.25;
        sigma = params->Lx * 0.1;
        amplitude = 50000.0;
    }
    // 2d case
    else {
        z0 = 0.0;
        sigma = params->Lx * 0.1;
        amplitude = 10000.0;
    }

    gridInitGaussian(u, x0, y0, z0, sigma, amplitude);

    // Save initial condition
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/output_0000.vtk", params->output_dir);
    ioWriteVTK(u, filename, "concentration");

    // Time stepping
    int n_steps = (int)(params->t_final / params->dt);
    printf("Running simulation: %d steps\n", n_steps);

    double t = 0.0;
    int output_count = 1;

    // Start timing
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);

    for (int step = 1; step <= n_steps; step++) {
        solverStep(u, u_new, params);

        // Swap grids
        Grid *temp = u;
        u = u_new;
        u_new = temp;

        t += params->dt;

        // Save output
        if (step % params->output_interval == 0) {
            snprintf(filename, sizeof(filename), "%s/output_%04d.vtk",
                     params->output_dir, output_count);
            ioWriteVTK(u, filename, "concentration");
            output_count++;
        }
    }

    // Stop timing and report
    gettimeofday(&time_end, NULL);
    double elapsed = (time_end.tv_sec - time_start.tv_sec) +
                     (time_end.tv_usec - time_start.tv_usec) / 1e6;

    printf("\nSimulation time: %.3f seconds\n", elapsed);

    printf("Output: %s/ (%d files)\n", params->output_dir, output_count);

    // Cleanup
    gridDestroy(u);
    gridDestroy(u_new);
    paramsDestroy(params);

    return 0;
}
