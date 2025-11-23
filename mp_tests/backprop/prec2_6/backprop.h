#include <half.hpp>
#include <floatx.hpp>
#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff
#define ETA 0.3
#define MOMENTUM 0.3
#define NUM_THREAD 8

typedef struct {
    int input_n;
    int hidden_n;
    int output_n;
    double *input_units;
    double *hidden_units;
    double *output_units;
    double *hidden_delta;
    double *output_delta;
    double *target;
    double **input_weights;
    double **hidden_weights;
    double **input_prev_weights;
    double **hidden_prev_weights;
} BPNN;

void bpnn_initialize(int seed);
BPNN* bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free(BPNN* net);
void bpnn_train(BPNN* net, double* eo, double* eh);
void bpnn_feedforward(BPNN* net);
void bpnn_save(BPNN* net, char* filename);
BPNN* bpnn_read(char* filename);

#endif