/*
 * Parameters implementation
 * XML config file parser using libxml2
 * Reads simulation parameters into Parameters struct
 */

#include "../include/parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>

// Helper function to get text content from XPath query
static char* getXPathString(xmlDocPtr doc, const char *xpath) {
    xmlXPathContextPtr ctx = xmlXPathNewContext(doc);
    xmlXPathObjectPtr result = xmlXPathEvalExpression((xmlChar*)xpath, ctx);

    char *value = NULL;
    if (result && result->nodesetval && result->nodesetval->nodeNr > 0) {
        xmlNodePtr node = result->nodesetval->nodeTab[0];
        xmlChar *content = xmlNodeListGetString(doc, node->xmlChildrenNode, 1);
        if (content) {
            value = strdup((char*)content);
            xmlFree(content);
        }
    }

    xmlXPathFreeObject(result);
    xmlXPathFreeContext(ctx);
    return value;
}

static int getXPathInt(xmlDocPtr doc, const char *xpath, int default_val) {
    char *str = getXPathString(doc, xpath);
    int val = default_val;
    if (str) {
        val = atoi(str);
        free(str);
    }
    return val;
}

static double getXPathDouble(xmlDocPtr doc, const char *xpath, double default_val) {
    char *str = getXPathString(doc, xpath);
    double val = default_val;
    if (str) {
        val = atof(str);
        free(str);
    }
    return val;
}

Parameters* paramsReadFromFile(const char *filename) {
    xmlDocPtr doc = xmlReadFile(filename, NULL, 0);
    if (doc == NULL) {
        printf("Error: Could not parse XML file '%s'\n", filename);
        return NULL;
    }

    Parameters *params = (Parameters*)malloc(sizeof(Parameters));

    // Grid
    params->nx = getXPathInt(doc, "/simulation/grid/nx", 101);
    params->ny = getXPathInt(doc, "/simulation/grid/ny", 101);
    params->nz = getXPathInt(doc, "/simulation/grid/nz", 1);

    // Domain
    params->Lx = getXPathDouble(doc, "/simulation/domain/Lx", 1.0);
    params->Ly = getXPathDouble(doc, "/simulation/domain/Ly", 1.0);
    params->Lz = getXPathDouble(doc, "/simulation/domain/Lz", 1.0);

    // Velocity
    params->vx = getXPathDouble(doc, "/simulation/velocity/vx", 0.0);
    params->vy = getXPathDouble(doc, "/simulation/velocity/vy", 0.0);
    params->vz = getXPathDouble(doc, "/simulation/velocity/vz", 0.0);

    // Physics
    params->D = getXPathDouble(doc, "/simulation/physics/D", 0.01);

    // Time
    params->dt = getXPathDouble(doc, "/simulation/time/dt", 0.001);
    params->t_final = getXPathDouble(doc, "/simulation/time/t_final", 1.0);

    // Boundary
    char *bc_str = getXPathString(doc, "/simulation/boundary/type");
    if (bc_str) {
        if (strcmp(bc_str, "periodic") == 0) params->bc_type = BC_PERIODIC;
        else if (strcmp(bc_str, "dirichlet") == 0) params->bc_type = BC_DIRICHLET;
        else if (strcmp(bc_str, "neumann") == 0) params->bc_type = BC_NEUMANN;
        else params->bc_type = BC_PERIODIC;
        free(bc_str);
    } else {
        params->bc_type = BC_PERIODIC;
    }

    // Output
    params->output_interval = getXPathInt(doc, "/simulation/output/interval", 100);
    char *dir_str = getXPathString(doc, "/simulation/output/dir");
    if (dir_str) {
        strncpy(params->output_dir, dir_str, sizeof(params->output_dir) - 1);
        params->output_dir[sizeof(params->output_dir) - 1] = '\0';
        free(dir_str);
    } else {
        strcpy(params->output_dir, "output");
    }

    // Free document and cleanup
    xmlFreeDoc(doc);
    xmlCleanupParser();

    return params;
}

// Free parameters struct
void paramsDestroy(Parameters *params) {
    free(params);
}

// Debug function to print parameters
void paramsPrint(const Parameters *params) {
    printf("Simulation Parameters:\n");
    printf("  Grid: %d x %d x %d\n", params->nx, params->ny, params->nz);
    printf("  Domain: %.2f x %.2f x %.2f\n", params->Lx, params->Ly, params->Lz);
    printf("  Velocity: vx=%.4f, vy=%.4f, vz=%.4f\n", params->vx, params->vy, params->vz);
    printf("  Diffusion: D=%.6f\n", params->D);
    printf("  Time: dt=%.6f, t_final=%.4f\n", params->dt, params->t_final);
    printf("  Boundary: ");
    if (params->bc_type == BC_PERIODIC) printf("periodic\n");
    else if (params->bc_type == BC_DIRICHLET) printf("dirichlet\n");
    else if (params->bc_type == BC_NEUMANN) printf("neumann\n");
    printf("  Output: interval=%d, dir=%s\n", params->output_interval, params->output_dir);
}
