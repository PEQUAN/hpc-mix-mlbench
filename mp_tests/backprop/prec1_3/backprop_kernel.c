#include <half.hpp>
#include <floatx.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(double *l1, double *l2, double **conn, int n1,
                              int n2);

extern void bpnn_output_error(double *delta, double *target, double *output,
                              int nj, double *err);

extern void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no,
                              double **who, double *hidden, double *err);

extern void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly,
                                double **w, double **oldw);


extern int setup(int argc, char **argv);

extern double **alloc_2d_dbl(int m, int n);

extern double squash(double x);

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { setup(argc, argv); }


void bpnn_train_kernel(BPNN *net, double *eo, double *eh) {
    int in, hid, out;
    double out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    printf("Performing CPU computation\n");
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                      &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                      net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                        net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                        net->input_weights, net->input_prev_weights);
    
    double output_delta_check[net->output_n + 1];
    for (int i = 0; i <= net->output_n; i++) {
        output_delta_check[i] = net->output_delta[i];
    }

    PROMISE_CHECK_ARRAY(output_delta_check, net->output_n + 1);
}
