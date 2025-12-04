/*
 * I/O functions for writing grid data to VTK format
 * Compatible with ParaView visualization
 */

#ifndef IO_H
#define IO_H

#include "grid.h"

// Write grid to VTK STRUCTURED_POINTS format for ParaView
int ioWriteVTK(const Grid *grid, const char *filename, const char *scalar_name);

#endif // IO_H
