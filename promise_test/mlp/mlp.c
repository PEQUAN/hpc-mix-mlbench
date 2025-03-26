#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 

typedef struct {
    __PROMISE__* features;
    int label;
    int feature_size;
} DataPoint;

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    __PROMISE__ learning_rate;
    unsigned int seed;
    __PROMISE__** w1;
    __PROMISE__* b1;
    __PROMISE__** w2;
    double* b2;
} MLPClassifier;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    __PROMISE__ s = sigmoid(x);
    return s * (1 - s);
}

// Simple linear congruential generator for random numbers
double rand_double(unsigned int* seed, double min, double max) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return min + (max - min) * (*seed / (double)0x7fffffff);
}

void initialize_weights(MLPClassifier* clf) {
    int i, j;
    __PROMISE__ limit1 = sqrt(6.0 / (clf->input_size + clf->hidden_size));
    __PROMISE__ limit2 = sqrt(6.0 / (clf->hidden_size + clf->output_size));
    
    clf->w1 = (double**)malloc(clf->hidden_size * sizeof(double*));
    clf->b1 = (double*)calloc(clf->hidden_size, sizeof(double));
    clf->w2 = (double**)malloc(clf->output_size * sizeof(double*));
    clf->b2 = (double*)calloc(clf->output_size, sizeof(double));
    
    for (i = 0; i < clf->hidden_size; i++) {
        clf->w1[i] = (double*)malloc(clf->input_size * sizeof(double));
        for (j = 0; j < clf->input_size; j++) {
            clf->w1[i][j] = rand_double(&clf->seed, -limit1, limit1);
        }
    }
    
    for (i = 0; i < clf->output_size; i++) {
        clf->w2[i] = (double*)malloc(clf->hidden_size * sizeof(double));
        for (j = 0; j < clf->hidden_size; j++) {
            clf->w2[i][j] = rand_double(&clf->seed, -limit2, limit2);
        }
    }
}

MLPClassifier* create_classifier(int in_size, int hid_size, int out_size, 
                                double lr, unsigned int seed) {
    MLPClassifier* clf = (MLPClassifier*)malloc(sizeof(MLPClassifier));
    clf->input_size = in_size;
    clf->hidden_size = hid_size;
    clf->output_size = out_size;
    clf->learning_rate = lr;
    clf->seed = seed;
    initialize_weights(clf);
    return clf;
}

void forward(MLPClassifier* clf, double* x, double* h, double* o) {
    int i, j;
    double* h_input = (double*)calloc(clf->hidden_size, sizeof(double));
    
    for (i = 0; i < clf->hidden_size; i++) {
        for (j = 0; j < clf->input_size; j++) {
            h_input[i] += x[j] * clf->w1[i][j];
        }
        h[i] = sigmoid(h_input[i] + clf->b1[i]);
    }
    
    for (i = 0; i < clf->output_size; i++) {
        o[i] = 0.0;
        for (j = 0; j < clf->hidden_size; j++) {
            o[i] += h[j] * clf->w2[i][j];
        }
        o[i] = sigmoid(o[i] + clf->b2[i]);
    }
    free(h_input);
}

void fit(MLPClassifier* clf, DataPoint* data, int data_size, int epochs) {
    int epoch, i, j, k;
    for (epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (i = 0; i < data_size; i++) {
            __PR_4__* h = (__PR_4__*)calloc(clf->hidden_size, sizeof(__PR_4__));
            __PR_5__* h_input = (__PR_5__*)calloc(clf->hidden_size, sizeof(__PR_5__));
            __PR_7__* o = (__PR_7__*)calloc(clf->output_size, sizeof(__PR_7__));
            double* o_input = (double*)calloc(clf->output_size, sizeof(double));
            double* target = (double*)calloc(clf->output_size, sizeof(double));
            double* o_delta = (double*)calloc(clf->output_size, sizeof(double));
            double* h_delta = (double*)calloc(clf->hidden_size, sizeof(double));
            
            // Forward pass
            for (j = 0; j < clf->hidden_size; j++) {
                for (k = 0; k < clf->input_size; k++) {
                    h_input[j] += data[i].features[k] * clf->w1[j][k];
                }
                h_input[j] += clf->b1[j];
                h[j] = sigmoid(h_input[j]);
            }
            
            for (j = 0; j < clf->output_size; j++) {
                for (k = 0; k < clf->hidden_size; k++) {
                    o_input[j] += h[k] * clf->w2[j][k];
                }
                o_input[j] += clf->b2[j];
                o[j] = sigmoid(o_input[j]);
            }
            
            target[data[i].label] = 1.0;
            for (j = 0; j < clf->output_size; j++) {
                total_loss += (o[j] - target[j]) * (o[j] - target[j]);
            }
            
            // Backward pass
            for (j = 0; j < clf->output_size; j++) {
                o_delta[j] = (o[j] - target[j]) * sigmoid_derivative(o_input[j]);
            }
            
            for (j = 0; j < clf->hidden_size; j++) {
                double error = 0.0;
                for (k = 0; k < clf->output_size; k++) {
                    error += o_delta[k] * clf->w2[k][j];
                }
                h_delta[j] = error * sigmoid_derivative(h_input[j]);
            }
            
            // Update weights
            for (j = 0; j < clf->output_size; j++) {
                for (k = 0; k < clf->hidden_size; k++) {
                    clf->w2[j][k] -= clf->learning_rate * o_delta[j] * h[k];
                }
                clf->b2[j] -= clf->learning_rate * o_delta[j];
            }
            
            for (j = 0; j < clf->hidden_size; j++) {
                for (k = 0; k < clf->input_size; k++) {
                    clf->w1[j][k] -= clf->learning_rate * h_delta[j] * data[i].features[k];
                }
                clf->b1[j] -= clf->learning_rate * h_delta[j];
            }
            
            free(h); free(h_input); free(o); free(o_input);
            free(target); free(o_delta); free(h_delta);
        }
        if (epoch % 10 == 0) {
            printf("Epoch %d Loss: %f\n", epoch, total_loss / data_size);
        }
    }
}

int predict(MLPClassifier* clf, double* features) {
    __PR_1__* h = (__PR_1__*)calloc(clf->hidden_size, sizeof(__PR_1__));
    double * o = (double *)calloc(clf->output_size, sizeof(double));
    forward(clf, features, h, o);
    
    int max_idx = 0;
    for (int i = 1; i < clf->output_size; i++) {
        if (o[i] > o[max_idx]) max_idx = i;
    }
    
    PROMISE_CHECK_VAR(o[max_idx]);
    free(h); free(o);
    return max_idx;
}

DataPoint* scale_features(DataPoint* data, int data_size, int n_features) {
    DataPoint* scaled = (DataPoint*)malloc(data_size * sizeof(DataPoint));
    __PR_2__* means = (__PR_2__*)calloc(n_features, sizeof(__PR_2__));
    __PR_3__* stds = (__PR_3__*)calloc(n_features, sizeof(__PR_3__));
    int i, j;
    
    for (i = 0; i < data_size; i++) {
        for (j = 0; j < n_features; j++) {
            means[j] += data[i].features[j];
        }
    }
    for (j = 0; j < n_features; j++) {
        means[j] /= data_size;
    }
    
    for (i = 0; i < data_size; i++) {
        for (j = 0; j < n_features; j++) {
            __PROMISE__ diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (j = 0; j < n_features; j++) {
        stds[j] = sqrt(stds[j] / data_size);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
    
    for (i = 0; i < data_size; i++) {
        scaled[i].features = (double*)malloc(n_features * sizeof(double));
        scaled[i].feature_size = n_features;
        scaled[i].label = data[i].label;
        for (j = 0; j < n_features; j++) {
            scaled[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
    }
    
    free(means); free(stds);
    return scaled;
}

DataPoint* read_csv(const char* filename, int* data_size, int* n_features) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;
    
    char line[1024];
    fgets(line, 1024, file); // Skip header
    
    int count = 0;
    while (fgets(line, 1024, file)) count++;
    rewind(file);
    fgets(line, 1024, file);
    
    DataPoint* data = (DataPoint*)malloc(count * sizeof(DataPoint));
    *data_size = count;
    
    int i = 0;
    while (fgets(line, 1024, file)) {
        int feature_count = 0;
        char* token = strtok(line, ",");
        double* features = (double*)malloc(20 * sizeof(double)); // Max 20 features
        
        while (token) {
            features[feature_count++] = atof(token);
            token = strtok(NULL, ",");
        }
        
        data[i].features = (double*)malloc((feature_count-1) * sizeof(double));
        for (int j = 0; j < feature_count-1; j++) {
            data[i].features[j] = features[j];
        }
        data[i].label = (int)features[feature_count-1];
        data[i].feature_size = feature_count-1;
        free(features);
        i++;
    }
    
    *n_features = data[0].feature_size;
    fclose(file);
    return data;
}

void free_classifier(MLPClassifier* clf) {
    for (int i = 0; i < clf->hidden_size; i++) free(clf->w1[i]);
    for (int i = 0; i < clf->output_size; i++) free(clf->w2[i]);
    free(clf->w1); free(clf->b1); free(clf->w2); free(clf->b2);
    free(clf);
}

void free_data(DataPoint* data, int data_size) {
    for (int i = 0; i < data_size; i++) free(data[i].features);
    free(data);
}

int main() {
    int data_size, n_features;
    DataPoint* raw_data = read_csv("../../data/classification/iris.csv", &data_size, &n_features);
    DataPoint* data = scale_features(raw_data, data_size, n_features);
    
    int train_size = (int)(0.8 * data_size);
    DataPoint* train_data = data;
    DataPoint* test_data = data + train_size;
    int test_size = data_size - train_size;
    
    int output_size = 0;
    for (int i = 0; i < data_size; i++) {
        if (data[i].label + 1 > output_size) output_size = data[i].label + 1;
    }
    
    unsigned int seed = 12345;
    MLPClassifier* clf = create_classifier(n_features, n_features + output_size, 
                                         output_size, 0.01, seed);
    
    clock_t start = clock();
    fit(clf, train_data, train_size, 200);
    clock_t end = clock();
    printf("Training time: %ld ms\n", (end - start) * 1000 / CLOCKS_PER_SEC);
    
    int* predictions = (int*)malloc(test_size * sizeof(int));
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        predictions[i] = predict(clf, test_data[i].features);
        if (predictions[i] == test_data[i].label) correct++;
    }
    
    double acc = (double)correct / test_size * 100;
    printf("Accuracy: %f%%\n", acc);


    free_classifier(clf);
    free_data(raw_data, data_size);
    free_data(data, data_size);
    free(predictions);
    return 0;
}