/*
 * Solver header file
 * Solver functions for 3D advection-diffusion equation
 */

#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"
#include "parameters.h"

// Apply boundary conditions to a grid
void solverApplyBoundary(Grid *grid, BoundaryType bc_type);

// Advection step using upwind scheme
void solverAdvectionStep(const Grid *u_old, Grid *u_new, const Parameters *params);

// Diffusion step using explicit central differences
void solverDiffusionStep(const Grid *u_old, Grid *u_new, const Parameters *params);

// Combined advection-diffusion step (operator splitting)
void solverStep(const Grid *u_old, Grid *u_new, const Parameters *params);

// Calculate L2 norm of a grid
double solverCalcL2Norm(const Grid *grid);

// Calculate maximum value in grid
double solverCalcMax(const Grid *grid);

// Calculate minimum value in grid
double solverCalcMin(const Grid *grid);

#endif // SOLVER_H
