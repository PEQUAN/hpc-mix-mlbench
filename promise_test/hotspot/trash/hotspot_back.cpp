#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <filesystem>

constexpr int BLOCK_SIZE = 16;
constexpr double MAX_PD = 3.0e6;
constexpr double PRECISION = 0.001;
constexpr double SPEC_HEAT_SI = 1.75e6;
constexpr double K_SI = 100;
constexpr double FACTOR_CHIP = 0.5;

constexpr float t_chip = 0.0005f;
constexpr float chip_height = 0.016f;
constexpr float chip_width = 0.016f;
constexpr float amb_temp = 80.0f;

using namespace std;
namespace fs = std::filesystem;

// Returns current system time in microseconds
long long get_time() {
    auto now = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();
}

void single_iteration(float* result, const float* temp, const float* power,
                     int row, int col, float cap_1, float rx_1, float ry_1, float rz_1, float step) {
    int num_chunk = row * col / (BLOCK_SIZE * BLOCK_SIZE);
    int chunks_in_row = col / BLOCK_SIZE;
    int chunks_in_col = row / BLOCK_SIZE;

    for (int chunk = 0; chunk < num_chunk; ++chunk) {
        int r_start = BLOCK_SIZE * (chunk / chunks_in_col);
        int c_start = BLOCK_SIZE * (chunk % chunks_in_row);
        int r_end = min(r_start + BLOCK_SIZE, row);
        int c_end = min(c_start + BLOCK_SIZE, col);

        if (r_start == 0 || c_start == 0 || r_end == row || c_end == col) {
            for (int r = r_start; r < r_end; ++r) {
                for (int c = c_start; c < c_end; ++c) {
                    float delta;
                    int idx = r * col + c;

                    // Corner cases
                    if (r == 0 && c == 0) {
                        delta = cap_1 * (power[0] +
                                        (temp[1] - temp[0]) * rx_1 +
                                        (temp[col] - temp[0]) * ry_1 +
                                        (amb_temp - temp[0]) * rz_1);
                    } else if (r == 0 && c == col - 1) {
                        delta = cap_1 * (power[c] +
                                        (temp[c - 1] - temp[c]) * rx_1 +
                                        (temp[c + col] - temp[c]) * ry_1 +
                                        (amb_temp - temp[c]) * rz_1);
                    } else if (r == row - 1 && c == col - 1) {
                        delta = cap_1 * (power[idx] +
                                        (temp[idx - 1] - temp[idx]) * rx_1 +
                                        (temp[(r - 1) * col + c] - temp[idx]) * ry_1 +
                                        (amb_temp - temp[idx]) * rz_1);
                    } else if (r == row - 1 && c == 0) {
                        delta = cap_1 * (power[idx] +
                                        (temp[idx + 1] - temp[idx]) * rx_1 +
                                        (temp[(r - 1) * col] - temp[idx]) * ry_1 +
                                        (amb_temp - temp[idx]) * rz_1);
                    }
                    // Edge cases
                    else if (r == 0) {
                        delta = cap_1 * (power[c] +
                                        (temp[c + 1] + temp[c - 1] - 2.0f * temp[c]) * rx_1 +
                                        (temp[col + c] - temp[c]) * ry_1 +
                                        (amb_temp - temp[c]) * rz_1);
                    } else if (c == col - 1) {
                        delta = cap_1 * (power[idx] +
                                        (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0f * temp[idx]) * ry_1 +
                                        (temp[idx - 1] - temp[idx]) * rx_1 +
                                        (amb_temp - temp[idx]) * rz_1);
                    } else if (r == row - 1) {
                        delta = cap_1 * (power[idx] +
                                        (temp[idx + 1] + temp[idx - 1] - 2.0f * temp[idx]) * rx_1 +
                                        (temp[(r - 1) * col + c] - temp[idx]) * ry_1 +
                                        (amb_temp - temp[idx]) * rz_1);
                    } else if (c == 0) {
                        delta = cap_1 * (power[idx] +
                                        (temp[(r + 1) * col] + temp[(r - 1) * col] - 2.0f * temp[idx]) * ry_1 +
                                        (temp[idx + 1] - temp[idx]) * rx_1 +
                                        (amb_temp - temp[idx]) * rz_1);
                    }
                    result[idx] = temp[idx] + delta;
                }
            }
            continue;
        }

        // Inner cells
        for (int r = r_start; r < r_end; ++r) {
            for (int c = c_start; c < c_end; ++c) {
                int idx = r * col + c;
                result[idx] = temp[idx] + cap_1 * (
                    power[idx] +
                    (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0f * temp[idx]) * ry_1 +
                    (temp[idx + 1] + temp[idx - 1] - 2.0f * temp[idx]) * rx_1 +
                    (amb_temp - temp[idx]) * rz_1
                );
            }
        }
    }
}

void compute_tran_temp(float* result, int num_iterations, float* temp, const float* power, int row, int col) {
    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
    float ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
    float rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0f;

    float rx_1 = 1.0f / rx;
    float ry_1 = 1.0f / ry;
    float rz_1 = 1.0f / rz;
    float cap_1 = step / cap;

    for (int i = 0; i < num_iterations; ++i) {
        single_iteration(result, temp, power, row, col, cap_1, rx_1, ry_1, rz_1, step);
        swap(temp, result);
    }
}

void write_output(const float* vect, int grid_rows, int grid_cols, const string& file) {
    if (!file.empty() && !fs::exists(fs::path(file).parent_path()) && fs::path(file).has_parent_path()) {
        throw runtime_error("Output directory does not exist: " + fs::path(file).parent_path().string());
    }
    ofstream out(file);
    if (!out.is_open()) {
        throw runtime_error("Could not open output file: " + file + " (check permissions or path)");
    }
    for (int i = 0; i < grid_rows; ++i) {
        for (int j = 0; j < grid_cols; ++j) {
            out << i * grid_cols + j << "\t" << vect[i * grid_cols + j] << "\n";
        }
    }
}

float* read_input(int grid_rows, int grid_cols, const string& file) {
    if (!fs::exists(file)) {
        throw runtime_error("Input file does not exist: " + file);
    }
    ifstream in(file);
    if (!in.is_open()) {
        throw runtime_error("Could not open input file: " + file + " (check permissions)");
    }
    float* vect = new float[grid_rows * grid_cols];
    for (int i = 0; i < grid_rows * grid_cols; ++i) {
        if (!in.good()) {
            delete[] vect;
            throw runtime_error("Not enough lines in file: " + file);
        }
        in >> vect[i];
        if (in.fail()) {
            delete[] vect;
            throw runtime_error("Invalid file format in: " + file);
        }
    }
    return vect;
}


int main(int argc, char* argv[]) {



    int grid_rows = stoi(argv[1]);
    int grid_cols = stoi(argv[2]);
    int sim_time = stoi(argv[3]);



    string tfile = argv[4];
    string pfile = argv[5];
    string ofile = argv[6];

    float* temp = new float[grid_rows * grid_cols];
    float* power = new float[grid_rows * grid_cols];
    float* result = new float[grid_rows * grid_cols];

    try {
        temp = read_input(grid_rows, grid_cols, tfile);
        power = read_input(grid_rows, grid_cols, pfile);
        result = new float[grid_rows * grid_cols];

        cout << "Starting transient temperature computation\n";
        long long start_time = get_time();
        compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
        long long end_time = get_time();

        cout << "Simulation completed in " << (end_time - start_time) / 1e6 << " seconds\n";

        write_output(temp, grid_rows, grid_cols, ofile);
        cout << "Output written to " << ofile << "\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        delete[] temp;
        delete[] power;
        delete[] result;
        return 1;
    }

    delete[] temp;
    delete[] power;
    delete[] result;
    return 0;
}