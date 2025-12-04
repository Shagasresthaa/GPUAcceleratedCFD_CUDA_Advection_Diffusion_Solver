/*
 * Parameters structure for advection-diffusion solver
 * Reads simulation config from file
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

typedef enum {
    BC_PERIODIC,
    BC_DIRICHLET,
    BC_NEUMANN
} BoundaryType;

typedef struct {
    // Grid dimensions
    int nx, ny, nz;
    double Lx, Ly, Lz;

    // Velocity field
    double vx, vy, vz;

    // Diffusion coefficient
    double D;

    // Time stepping
    double dt;
    double t_final;

    // Boundary conditions
    BoundaryType bc_type;

    // Output control
    int output_interval;
    char output_dir[256];
} Parameters;

// Read parameters from config file
Parameters* paramsReadFromFile(const char *filename);

// Free parameters
void paramsDestroy(Parameters *params);

// Print parameters
void paramsPrint(const Parameters *params);

#endif // PARAMETERS_H
