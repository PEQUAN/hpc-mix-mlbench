
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern int layer_size;


int load(BPNN *net) {
    double *units;
    int nr, nc, imgsize, i, k;

    unsigned int seed = 1234;
    srand(seed); /* Seed for rand() in load */
    // bpnn_initialize(seed); /* Seed for backprop.c's LCG */

    nr = layer_size;
    nc = layer_size; /* Assuming square input layer */
    imgsize = nr * nc;
    units = net->input_units;

    if (!units) {
        fprintf(stderr, "load: Null input_units\n");
        return -1;
    }

    k = 1; /* Start at 1 to skip bias unit */
    for (i = 0; i < imgsize; i++) {
        units[k] = (double)rand() / RAND_MAX;
        k++;
    }

    return 0; /* Success */
}
