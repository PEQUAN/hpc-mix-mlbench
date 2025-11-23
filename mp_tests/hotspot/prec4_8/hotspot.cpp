#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <array>
#include <algorithm>

namespace {
    int BLOCK_SIZE = 16;
    int BLOCK_SIZE_C = BLOCK_SIZE;
    int BLOCK_SIZE_R = BLOCK_SIZE;

    double MAX_PD = 3.0e6;
    double PRECISION = 0.001;
    double SPEC_HEAT_SI = 1.75e6;
    double K_SI = 100.0;
    double FACTOR_CHIP = 0.5;

    double t_chip = 0.0005;
    double chip_height = 0.016;
    double chip_width = 0.016;
    double amb_temp = 80.0;
}

// Returns the current system time in microseconds
long long get_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void single_iteration(double* result, const double* temp, 
                     const double* power, int row, int col,
                     flx::floatx<8, 7> Cap_1, flx::floatx<4, 3> Rx_1, flx::floatx<4, 3> Ry_1, flx::floatx<8, 7> Rz_1, flx::floatx<4, 3> step) {
    flx::floatx<8, 7> delta;
    int num_chunk = row * col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col / BLOCK_SIZE_C;
    int chunks_in_col = row / BLOCK_SIZE_R;

    for (int chunk = 0; chunk < num_chunk; ++chunk) {
        int r_start = BLOCK_SIZE_R * (chunk / chunks_in_col);
        int c_start = BLOCK_SIZE_C * (chunk % chunks_in_row);
        int r_end = std::min(r_start + BLOCK_SIZE_R, row);
        int c_end = std::min(c_start + BLOCK_SIZE_C, col);

        if (r_start == 0 || c_start == 0 || r_end == row || c_end == col) {
            for (int r = r_start; r < r_end; ++r) {
                for (int c = c_start; c < c_end; ++c) {
                    if (r == 0 && c == 0) {
                        delta = Cap_1 * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[col] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }
                    else if (r == 0 && c == col - 1) {
                        delta = Cap_1 * (power[c] +
                            (temp[c - 1] - temp[c]) * Rx_1 +
                            (temp[c + col] - temp[c]) * Ry_1 +
                            (amb_temp - temp[c]) * Rz_1);
                    }
                    else if (r == row - 1 && c == col - 1) {
                        delta = Cap_1 * (power[r * col + c] +
                            (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 +
                            (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                            (amb_temp - temp[r * col + c]) * Rz_1);
                    }
                    else if (r == row - 1 && c == 0) {
                        delta = Cap_1 * (power[r * col] +
                            (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                            (temp[(r - 1) * col] - temp[r * col]) * Ry_1 +
                            (amb_temp - temp[r * col]) * Rz_1);
                    }
                    else if (r == 0) {
                        delta = Cap_1 * (power[c] +
                            (temp[c + 1] + temp[c - 1] - 2.0 * temp[c]) * Rx_1 +
                            (temp[col + c] - temp[c]) * Ry_1 +
                            (amb_temp - temp[c]) * Rz_1);
                    }
                    else if (c == col - 1) {
                        delta = Cap_1 * (power[r * col + c] +
                            (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0 * temp[r * col + c]) * Ry_1 +
                            (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 +
                            (amb_temp - temp[r * col + c]) * Rz_1);
                    }
                    else if (r == row - 1) {
                        delta = Cap_1 * (power[r * col + c] +
                            (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.0 * temp[r * col + c]) * Rx_1 +
                            (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                            (amb_temp - temp[r * col + c]) * Rz_1);
                    }
                    else if (c == 0) {
                        delta = Cap_1 * (power[r * col] +
                            (temp[(r + 1) * col] + temp[(r - 1) * col] - 2.0 * temp[r * col]) * Ry_1 +
                            (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                            (amb_temp - temp[r * col]) * Rz_1);
                    }
                    result[r * col + c] = temp[r * col + c] + delta;
                }
            }
            continue;
        }

        for (int r = r_start; r < r_end; ++r) {
            for (int c = c_start; c < c_end; ++c) {
                result[r * col + c] = temp[r * col + c] +
                    (Cap_1 * (power[r * col + c] +
                    (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0 * temp[r * col + c]) * Ry_1 +
                    (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.0 * temp[r * col + c]) * Rx_1 +
                    (amb_temp - temp[r * col + c]) * Rz_1));
            }
        }
    }
}

void compute_tran_temp(double* result, int num_iterations, 
                      double* temp, const double* power, 
                      int row, int col) {
    flx::floatx<8, 7> grid_height = chip_height / row;
    flx::floatx<8, 7> grid_width = chip_width / col;

    flx::floatx<8, 7> Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    flx::floatx<4, 3> Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    flx::floatx<4, 3> Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    flx::floatx<8, 7> Rz = t_chip / (K_SI * grid_height * grid_width);

    flx::floatx<8, 7> max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    flx::floatx<8, 7> step = PRECISION / max_slope / 1000.0;

    flx::floatx<4, 3> Rx_1 = 1.0 / Rx;
    flx::floatx<4, 3> Ry_1 = 1.0 / Ry;
    flx::floatx<8, 7> Rz_1 = 1.0 / Rz;
    flx::floatx<8, 7> Cap_1 = step / Cap;

    for (int i = 0; i < num_iterations; ++i) {
        single_iteration(result, temp, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        std::swap(temp, result);
    }
}

void write_output(const double* vect, int grid_rows, int grid_cols, const std::string& file) {
    std::ofstream fp(file);
    if (!fp.is_open()) {
        throw std::runtime_error("Unable to open output file");
    }

    int index = 0;
    for (int i = 0; i < grid_rows; ++i) {
        for (int j = 0; j < grid_cols; ++j) {
            fp << index << "\t" << vect[i * grid_cols + j] << "\n";
            ++index;
        }
    }
}

void read_input(double* vect, int grid_rows, int grid_cols, const std::string& file) {
    std::ifstream fp(file);
    if (!fp.is_open()) {
        throw std::runtime_error("Unable to open input file");
    }

    std::string line;
    for (int i = 0; i < grid_rows * grid_cols; ++i) {
        if (!std::getline(fp, line)) {
            throw std::runtime_error("Not enough lines in file");
        }
        try {
            vect[i] = std::stod(line);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid file format");
        }
    }
}

void usage(const std::string& program) {
    std::cerr << "Usage: " << program 
              << " <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n"
              << "\t<grid_rows>  - number of rows in the grid (positive integer)\n"
              << "\t<grid_cols>  - number of columns in the grid (positive integer)\n"
              << "\t<sim_time>   - number of iterations\n"
              << "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n"
              << "\t<power_file> - name of the file containing the dissipated power values of each cell\n"
              << "\t<output_file> - name of the output file\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    double* temp = nullptr;
    double* power = nullptr;
    double* result = nullptr;
    
    try {
        if (argc != 7) {
            usage(argv[0]);
        }

        int grid_rows = std::stoi(argv[1]);
        int grid_cols = std::stoi(argv[2]);
        int sim_time = std::stoi(argv[3]);
        
        if (grid_rows <= 0 || grid_cols <= 0 || sim_time <= 0) {
            usage(argv[0]);
        }

        size_t size = static_cast<size_t>(grid_rows) * grid_cols;
        
        temp = new double[size]();
        power = new double[size]();
        result = new double[size]();

        std::string tfile = argv[4];
        std::string pfile = argv[5];
        std::string ofile = argv[6];

        read_input(temp, grid_rows, grid_cols, tfile);
        read_input(power, grid_rows, grid_cols, pfile);

        std::cout << "Start computing the transient temperature\n";
        
        auto start_time = get_time();
        compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
        auto end_time = get_time();

        std::cout << "Ending simulation\n";
        std::cout << "Total time: " << ((end_time - start_time) / 1000000.0) << " seconds\n";

        write_output((sim_time & 1) ? result : temp, grid_rows, grid_cols, ofile);

        PROMISE_CHECK_ARRAY(temp, grid_rows * grid_cols);

        delete[] temp;
        delete[] power;
        delete[] result;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        delete[] temp;
        delete[] power;
        delete[] result;
        return 1;
    }
}


