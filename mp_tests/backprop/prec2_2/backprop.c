#include <half.hpp>
#include <floatx.hpp>
/* backprop.c
 * Modified from original C code by Jeff Shufelt (1994) and Shuai Che
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <errno.h>
 #include <string.h>
 #include <stdint.h>
 #include <omp.h>
 #include "backprop.h"
 #include <random>
 #define ABS(x) ((x) >= 0.0f ? (x) : -(x))
 
 /* Function prototypes to avoid implicit declarations */
 static void bpnn_randomize_weights(double** w, int m, int n);
 static void bpnn_randomize_row(double* w, int m);
 static void bpnn_zero_weights(double** w, int m, int n);
 

 /* Thread-local random number generator state */
 static unsigned int rng_state = 1;
  
static void init_rng(unsigned int seed) {
     #ifdef _OPENMP
     /* Initialize per-thread seed in parallel region */
     #pragma omp parallel
     {
         rng_state = seed + omp_get_thread_num(); /* Unique seed per thread */
     }
     #else
     rng_state = seed ? seed : 1;
     #endif
     printf("Random number generator seed: %u\n", seed);
 }
 
 /* Generate random double between 0.0 and 1.0 */
 static double drnd(void) {
     rng_state = (unsigned int)(((uint64_t)rng_state * 48271) % 0x7fffffff);
     return (double)rng_state / (double)BIGRND;
 }
 
 /* Generate random double between -1.0 and 1.0 */
 static flx::floatx<5, 2> dpn1(void) {
     return drnd() * 2.0f - 1.0f;
 }
 
 /* Sigmoid activation function */
 static flx::floatx<8, 7> squash(flx::floatx<8, 7> x) {
     return 1.0f / (1.0f + expf(-x));
 }
 
 /* Allocate 1D array of doubles */
 static double* alloc_1d_dbl(int n) {
    double* new_array = (double*)calloc(n, sizeof(double));
     if (!new_array) {
         fprintf(stderr, "alloc_1d_dbl: Couldn't allocate array of %d doubles\n", n);
         return NULL;
     }

     return new_array;
 }
 
 /* Allocate 2D array of doubles */
 static double** alloc_2d_dbl(int m, int n) {
     double** new_array = (double**)calloc(m, sizeof(double*));
     if (!new_array) {
         fprintf(stderr, "alloc_2d_dbl: Couldn't allocate array of %d pointers\n", m);
         return NULL;
     }
 
     for (int i = 0; i < m; i++) {
         new_array[i] = alloc_1d_dbl(n);
         if (!new_array[i]) {
             fprintf(stderr, "alloc_2d_dbl: Failed to allocate row %d\n", i);
             while (i-- > 0) {
                 free(new_array[i]);
             }
             free(new_array);
             return NULL;
         }
     }
     return new_array;
 }
 
 /* Free 2D array of doubles */
 static void free_2d_dbl(double** array, int m) {
     if (!array) return;
     for (int i = 0; i < m; i++) {
         free(array[i]);
     }
     free(array);
 }
 
 void bpnn_initialize(int seed) {
     init_rng((unsigned int)seed);
 }
 
 BPNN* bpnn_create(int n_in, int n_hidden, int n_out) {
     BPNN* newnet = (BPNN*)malloc(sizeof(BPNN));
     if (!newnet) {
         fprintf(stderr, "bpnn_create: Couldn't allocate neural network\n");
         return NULL;
     }
 
     /* Initialize to avoid undefined behavior */
     *newnet = (BPNN){
         .input_n = n_in,
         .hidden_n = n_hidden,
         .output_n = n_out,
         .input_units = NULL,
         .hidden_units = NULL,
         .output_units = NULL,
         .hidden_delta = NULL,
         .output_delta = NULL,
         .target = NULL,
         .input_weights = NULL,
         .hidden_weights = NULL,
         .input_prev_weights = NULL,
         .hidden_prev_weights = NULL
     };
 
     /* Allocate arrays */
     newnet->input_units = alloc_1d_dbl(n_in + 1);
     newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
     newnet->output_units = alloc_1d_dbl(n_out + 1);
     newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
     newnet->output_delta = alloc_1d_dbl(n_out + 1);
     newnet->target = alloc_1d_dbl(n_out + 1);
     newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
     newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
     newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
     newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
 
     /* Check for allocation failures */
     if (!newnet->input_units || !newnet->hidden_units || !newnet->output_units ||
         !newnet->hidden_delta || !newnet->output_delta || !newnet->target ||
         !newnet->input_weights || !newnet->hidden_weights ||
         !newnet->input_prev_weights || !newnet->hidden_prev_weights) {
         bpnn_free(newnet);
         return NULL;
     }
 
 #ifdef INITZERO
     bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
 #else
     bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
 #endif
     bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
     bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
     bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
     bpnn_randomize_row(newnet->target, n_out);
 
     return newnet;
 }
 
 void bpnn_free(BPNN* net) {
     if (!net) return;
     free(net->input_units);
     free(net->hidden_units);
     free(net->output_units);
     free(net->hidden_delta);
     free(net->output_delta);
     free(net->target);
     free_2d_dbl(net->input_weights, net->input_n + 1);
     free_2d_dbl(net->hidden_weights, net->hidden_n + 1);
     free_2d_dbl(net->input_prev_weights, net->input_n + 1);
     free_2d_dbl(net->hidden_prev_weights, net->hidden_n + 1);
     free(net);
 }
 
static void bpnn_randomize_weights(double** w, int m, int n) {
    // Seed for reproducibility (use a fixed seed or pass as parameter)
    unsigned int seed = 42; // Fixed seed for reproducibility; can be dynamic (e.g., time-based)

    #pragma omp parallel num_threads(NUM_THREAD)
    {
        // Each thread gets its own random number generator
        std::mt19937 rng(seed + omp_get_thread_num()); // Thread-specific seed
        std::uniform_real_distribution<double> dist(-1.0, 1.0); // Adjust range as needed

        #pragma omp for collapse(2)
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                w[i][j] = dist(rng); // Generate random weight
            }
        }
    }
}
 
 static void bpnn_randomize_row(double* w, int m) {
     for (int i = 0; i <= m; i++) {
         w[i] = 0.1f;
     }
 }
 
 static void bpnn_zero_weights(double** w, int m, int n) {
     #pragma omp parallel for collapse(2) num_threads(NUM_THREAD)
     for (int i = 0; i <= m; i++) {
         for (int j = 0; j <= n; j++) {
             w[i][j] = 0.0f;
         }
     }
 }
 
 void bpnn_layerforward(double* l1, double* l2, double** conn, int n1, int n2) {
     l1[0] = 1.0f;
     #pragma omp parallel for num_threads(NUM_THREAD)
     for (int j = 1; j <= n2; j++) {
         double sum = 0.0f;
         for (int k = 0; k <= n1; k++) {
             sum += conn[k][j] * l1[k];
         }
         l2[j] = squash(sum);
     }
 }
 
 void bpnn_output_error(double* delta, double* target, double* output, int nj, double* err) {
    float errsum = 0.0f;
     #pragma omp parallel for reduction(+:errsum) num_threads(NUM_THREAD)
     for (int j = 1; j <= nj; j++) {
        flx::floatx<8, 7> o = output[j];
        flx::floatx<8, 7> t = target[j];
         delta[j] = o * (1.0f - o) * (t - o);
         errsum += ABS(delta[j]);
     }
     *err = errsum;
     
 }
 
 void bpnn_hidden_error(double* delta_h, int nh, double* delta_o, int no, double** who, double* hidden, double* err) {
     float errsum = 0.0f;
     #pragma omp parallel for reduction(+:errsum) num_threads(NUM_THREAD)
     for (int j = 1; j <= nh; j++) {
         flx::floatx<5, 2> h = hidden[j];
         flx::floatx<5, 2> sum = 0.0f;
         for (int k = 1; k <= no; k++) {
             sum += delta_o[k] * who[j][k];
         }
         delta_h[j] = h * (1.0f - h) * sum;
         errsum += ABS(delta_h[j]);
     }
     *err = errsum;
 }
 
 void bpnn_adjust_weights(double* delta, int ndelta, double* ly, int nly, double** w, double** oldw) {
     #pragma omp parallel for collapse(2) num_threads(NUM_THREAD)
     for (int j = 1; j <= ndelta; j++) {
         for (int k = 0; k <= nly; k++) {
             flx::floatx<5, 2> new_dw = (ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]);
             w[k][j] += new_dw;
             oldw[k][j] = new_dw;
         }
     }
 }
 
 void bpnn_feedforward(BPNN* net) {
     bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      net->input_n, net->hidden_n);
     bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      net->hidden_n, net->output_n);
 }
 
 void bpnn_train(BPNN* net, double* eo, double* eh) {
     double out_err, hid_err;
     bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      net->input_n, net->hidden_n);
     bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      net->hidden_n, net->output_n);
     bpnn_output_error(net->output_delta, net->target, net->output_units,
                      net->output_n, &out_err);
     bpnn_hidden_error(net->hidden_delta, net->hidden_n, net->output_delta,
                      net->output_n, net->hidden_weights, net->hidden_units, &hid_err);
     *eo = out_err;
     *eh = hid_err;
     bpnn_adjust_weights(net->output_delta, net->output_n, net->hidden_units,
                        net->hidden_n, net->hidden_weights, net->hidden_prev_weights);
     bpnn_adjust_weights(net->hidden_delta, net->hidden_n, net->input_units,
                        net->input_n, net->input_weights, net->input_prev_weights);
 }
 
 void bpnn_save(BPNN* net, char* filename) {
     FILE* file = fopen(filename, "wb");
     if (!file) {
         fprintf(stderr, "bpnn_save: Could not open file '%s': %s\n",
                 filename, strerror(errno));
         return;
     }
 
     printf("Saving %dx%dx%d network to '%s'\n",
            net->input_n, net->hidden_n, net->output_n, filename);
 
     if (fwrite(&net->input_n, sizeof(int), 1, file) != 1 ||
         fwrite(&net->hidden_n, sizeof(int), 1, file) != 1 ||
         fwrite(&net->output_n, sizeof(int), 1, file) != 1) {
         fprintf(stderr, "bpnn_save: Error writing dimensions to '%s'\n", filename);
         fclose(file);
         return;
     }
 
     for (int i = 0; i <= net->input_n; i++) {
         if (fwrite(net->input_weights[i], sizeof(double), net->hidden_n + 1, file) !=
             (size_t)(net->hidden_n + 1)) {
             fprintf(stderr, "bpnn_save: Error writing input weights\n");
             fclose(file);
             return;
         }
     }
 
     for (int i = 0; i <= net->hidden_n; i++) {
         if (fwrite(net->hidden_weights[i], sizeof(double), net->output_n + 1, file) !=
             (size_t)(net->output_n + 1)) {
             fprintf(stderr, "bpnn_save: Error writing hidden weights\n");
             fclose(file);
             return;
         }
     }
 
     fclose(file);
 }
 
 BPNN* bpnn_read(char* filename) {
     FILE* file = fopen(filename, "rb");
     if (!file) {
         fprintf(stderr, "bpnn_read: Could not open file '%s': %s\n",
                 filename, strerror(errno));
         return NULL;
     }
 
     int n1, n2, n3;
     if (fread(&n1, sizeof(int), 1, file) != 1 ||
         fread(&n2, sizeof(int), 1, file) != 1 ||
         fread(&n3, sizeof(int), 1, file) != 1) {
         fprintf(stderr, "bpnn_read: Error reading dimensions from '%s'\n", filename);
         fclose(file);
         return NULL;
     }
 
     printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
     printf("Reading input weights...");
 
     BPNN* newnet = bpnn_create(n1, n2, n3);
     if (!newnet) {
         fclose(file);
         return NULL;
     }
 
     for (int i = 0; i <= n1; i++) {
         if (fread(newnet->input_weights[i], sizeof(double), n2 + 1, file) !=
             (size_t)(n2 + 1)) {
             fprintf(stderr, "bpnn_read: Error reading input weights\n");
             bpnn_free(newnet);
             fclose(file);
             return NULL;
         }
     }
 
     printf("Done\nReading hidden weights...");
 
     for (int i = 0; i <= n2; i++) {
         if (fread(newnet->hidden_weights[i], sizeof(double), n3 + 1, file) !=
             (size_t)(n3 + 1)) {
             fprintf(stderr, "bpnn_read: Error reading hidden weights\n");
             bpnn_free(newnet);
             fclose(file);
             return NULL;
         }
     }
 
     printf("Done\n");
     fclose(file);
 
     bpnn_zero_weights(newnet->input_prev_weights, n1, n2);
     bpnn_zero_weights(newnet->hidden_prev_weights, n2, n3);
 
     return newnet;
 }