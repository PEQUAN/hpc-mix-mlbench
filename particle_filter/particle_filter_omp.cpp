#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <array>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace particle_filter {
    constexpr double PI = 3.1415926535897932;
    constexpr long M = std::numeric_limits<int>::max();
    constexpr int A = 1103515245;
    constexpr int C = 12345;
    constexpr int DISK_RADIUS = 5;
    constexpr int FIXED_SEED = 123456789; // Fixed seed for reproducibility

    class ParticleFilter {
    private:
        int IszX, IszY, Nfr, Nparticles, num_threads;
        int* I;
        int* seed;
        int* disk;
        double* objxy;
        double* weights;
        double* likelihood;
        double* arrayX;
        double* arrayY;
        double* xj;
        double* yj;
        double* CDF;
        double* u;
        int* ind;
        int countOnes;
        double xe, ye;

        using TimePoint = std::chrono::high_resolution_clock::time_point;

        TimePoint get_time() {
            return std::chrono::high_resolution_clock::now();
        }

        double elapsed_time(const TimePoint& start, const TimePoint& end) {
            return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);
        }

        double roundDouble(double value) {
            int newValue = static_cast<int>(value);
            return (value - newValue < 0.5) ? newValue : newValue + 1;
        }

        void setIf(int testValue, int newValue, int* array3D, int dimX, int dimY, int dimZ) {
            #pragma omp parallel for collapse(3)
            for (int x = 0; x < dimX; ++x) {
                for (int y = 0; y < dimY; ++y) {
                    for (int z = 0; z < dimZ; ++z) {
                        if (array3D[x * dimY * dimZ + y * dimZ + z] == testValue) {
                            array3D[x * dimY * dimZ + y * dimZ + z] = newValue;
                        }
                    }
                }
            }
        }

        double randu(int* seed, int index) {
            if (index < 0 || index >= num_threads) {
                throw std::runtime_error("Invalid seed index: " + std::to_string(index));
            }
            int num = A * seed[index] + C;
            seed[index] = num % M;
            return std::abs(seed[index] / static_cast<double>(M));
        }

        double randn(int* seed, int index) {
            double u = randu(seed, index);
            double v = randu(seed, index);
            double cosine = std::cos(2 * PI * v);
            double rt = -2 * std::log(u);
            return std::sqrt(rt) * cosine;
        }

        void addNoise(int* array3D, int dimX, int dimY, int dimZ) {
            #pragma omp parallel for collapse(3)
            for (int x = 0; x < dimX; ++x) {
                for (int y = 0; y < dimY; ++y) {
                    for (int z = 0; z < dimZ; ++z) {
                        int tid = omp_get_thread_num();
                        array3D[x * dimY * dimZ + y * dimZ + z] += static_cast<int>(5 * randn(seed, tid));
                    }
                }
            }
        }

        void strelDisk(int* disk, int radius) {
            int diameter = radius * 2 - 1;
            #pragma omp parallel for collapse(2)
            for (int x = 0; x < diameter; ++x) {
                for (int y = 0; y < diameter; ++y) {
                    double distance = std::sqrt(std::pow(static_cast<double>(x - radius + 1), 2) + 
                                              std::pow(static_cast<double>(y - radius + 1), 2));
                    if (distance < radius) {
                        disk[x * diameter + y] = 1;
                    }
                }
            }
        }

        void dilate_matrix(int* matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
            int startX = std::max(posX - error, 0);
            int startY = std::max(posY - error, 0);
            int endX = std::min(posX + error, dimX);
            int endY = std::min(posY + error, dimY);

            for (int x = startX; x < endX; ++x) {
                for (int y = startY; y < endY; ++y) {
                    double distance = std::sqrt(std::pow(static_cast<double>(x - posX), 2.0) + 
                                              std::pow(static_cast<double>(y - posY), 2.0));
                    if (distance < error) {
                        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
                    }
                }
            }
        }

        void imdilate_disk(int* matrix, int dimX, int dimY, int dimZ, int error, int* newMatrix) {
            #pragma omp parallel for collapse(3)
            for (int z = 0; z < dimZ; ++z) {
                for (int x = 0; x < dimX; ++x) {
                    for (int y = 0; y < dimY; ++y) {
                        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                            dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                        }
                    }
                }
            }
        }

        void getneighbors(int* se, int numOnes, double* neighbors, int radius) {
            int diameter = radius * 2 - 1;
            int center = radius - 1;
            int neighY = 0;
            for (int x = 0; x < diameter; ++x) {
                for (int y = 0; y < diameter; ++y) {
                    if (se[x * diameter + y]) {
                        neighbors[neighY * 2] = y - center;
                        neighbors[neighY * 2 + 1] = x - center;
                        ++neighY;
                    }
                }
            }
        }

        void videoSequence() {
            int max_size = IszX * IszY * Nfr;
            int x0 = static_cast<int>(roundDouble(IszY / 2.0));
            int y0 = static_cast<int>(roundDouble(IszX / 2.0));
            I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

            for (int k = 1; k < Nfr; ++k) {
                int xk = std::abs(x0 + (k - 1));
                int yk = std::abs(y0 - 2 * (k - 1));
                int pos = yk * IszY * Nfr + xk * Nfr + k;
                if (xk >= IszX || yk >= IszY || pos >= max_size) {
                    pos = 0;
                }
                I[pos] = 1;
            }

            int* newMatrix = new int[IszX * IszY * Nfr]();
            if (!newMatrix) throw std::bad_alloc();
            try {
                imdilate_disk(I, IszX, IszY, Nfr, DISK_RADIUS, newMatrix);
                for (int x = 0; x < IszX; ++x) {
                    for (int y = 0; y < IszY; ++y) {
                        for (int k = 0; k < Nfr; ++k) {
                            I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
                        }
                    }
                }
                delete[] newMatrix;
            }
            catch (...) {
                delete[] newMatrix;
                throw;
            }

            setIf(0, 100, I, IszX, IszY, Nfr);
            setIf(1, 228, I, IszX, IszY, Nfr);
            addNoise(I, IszX, IszY, Nfr);
        }

        double calcLikelihoodSum(int* I, int* ind, int numOnes) {
            double likelihoodSum = 0.0;
            for (int y = 0; y < numOnes; ++y) {
                likelihoodSum += (std::pow(static_cast<double>(I[ind[y]] - 100), 2) - 
                                std::pow(static_cast<double>(I[ind[y]] - 228), 2)) / 50.0;
            }
            return likelihoodSum;
        }

        int findIndex(double* CDF, int lengthCDF, double value) {
            for (int x = 0; x < lengthCDF; ++x) {
                if (CDF[x] >= value) {
                    return x;
                }
            }
            return lengthCDF - 1;
        }

        void cleanup() {
            delete[] I;
            delete[] seed;
            delete[] disk;
            delete[] objxy;
            delete[] weights;
            delete[] likelihood;
            delete[] arrayX;
            delete[] arrayY;
            delete[] xj;
            delete[] yj;
            delete[] CDF;
            delete[] u;
            delete[] ind;
        }

    public:
        ParticleFilter(int x, int y, int frames, int particles, int threads)
            : IszX(x), IszY(y), Nfr(frames), Nparticles(particles), num_threads(threads),
              xe(roundDouble(y / 2.0)), ye(roundDouble(x / 2.0)),
              I(nullptr), seed(nullptr), disk(nullptr), objxy(nullptr),
              weights(nullptr), likelihood(nullptr), arrayX(nullptr), arrayY(nullptr),
              xj(nullptr), yj(nullptr), CDF(nullptr), u(nullptr), ind(nullptr) {
            if (IszX <= 0 || IszY <= 0 || Nfr <= 0 || Nparticles <= 0 || num_threads <= 0) {
                throw std::runtime_error("Invalid input parameters");
            }

            omp_set_num_threads(num_threads);

            try {
                I = new int[IszX * IszY * Nfr]();
                if (!I) throw std::bad_alloc();
                seed = new int[num_threads]();
                if (!seed) throw std::bad_alloc();
                for (int i = 0; i < num_threads; ++i) {
                    seed[i] = FIXED_SEED + i; // Fixed seed with offset for reproducibility
                }

                int diameter = DISK_RADIUS * 2 - 1;
                disk = new int[diameter * diameter]();
                if (!disk) throw std::bad_alloc();
                strelDisk(disk, DISK_RADIUS);
                countOnes = 0;
                for (int x = 0; x < diameter; ++x) {
                    for (int y = 0; y < diameter; ++y) {
                        if (disk[x * diameter + y] == 1) {
                            ++countOnes;
                        }
                    }
                }

                objxy = new double[countOnes * 2]();
                if (!objxy) throw std::bad_alloc();
                getneighbors(disk, countOnes, objxy, DISK_RADIUS);

                weights = new double[Nparticles]();
                if (!weights) throw std::bad_alloc();
                likelihood = new double[Nparticles]();
                if (!likelihood) throw std::bad_alloc();
                arrayX = new double[Nparticles]();
                if (!arrayX) throw std::bad_alloc();
                arrayY = new double[Nparticles]();
                if (!arrayY) throw std::bad_alloc();
                xj = new double[Nparticles]();
                if (!xj) throw std::bad_alloc();
                yj = new double[Nparticles]();
                if (!yj) throw std::bad_alloc();
                CDF = new double[Nparticles]();
                if (!CDF) throw std::bad_alloc();
                u = new double[Nparticles]();
                if (!u) throw std::bad_alloc();
                ind = new int[countOnes * Nparticles]();
                if (!ind) throw std::bad_alloc();
            }
            catch (...) {
                cleanup();
                throw;
            }
        }

        ~ParticleFilter() {
            cleanup();
        }

        void run() {
            auto start = get_time();
            videoSequence();
            auto endVideoSequence = get_time();
            std::cout << "VIDEO SEQUENCE TOOK " << elapsed_time(start, endVideoSequence) << "\n";

            auto get_neighbors = get_time();
            std::cout << "TIME TO GET NEIGHBORS TOOK: " << elapsed_time(start, get_neighbors) << "\n";

            #pragma omp parallel for
            for (int x = 0; x < Nparticles; ++x) {
                weights[x] = 1.0 / Nparticles;
            }
            auto get_weights = get_time();
            std::cout << "TIME TO GET WEIGHTS TOOK: " << elapsed_time(get_neighbors, get_weights) << "\n";

            #pragma omp parallel for
            for (int x = 0; x < Nparticles; ++x) {
                arrayX[x] = xe;
                arrayY[x] = ye;
            }
            auto set_arrays = get_time();
            std::cout << "TIME TO SET ARRAYS TOOK: " << elapsed_time(get_weights, set_arrays) << "\n";

            int max_size = IszX * IszY * Nfr;
            for (int k = 1; k < Nfr; ++k) {
                auto set_arrays_time = get_time();
                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    int tid = omp_get_thread_num();
                    arrayX[x] += 1 + 5 * randn(seed, tid);
                    arrayY[x] += -2 + 2 * randn(seed, tid);
                }
                auto error = get_time();
                std::cout << "TIME TO SET ERROR TOOK: " << elapsed_time(set_arrays_time, error) << "\n";

                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    for (int y = 0; y < countOnes; ++y) {
                        int indX = std::clamp(static_cast<int>(roundDouble(arrayX[x]) + objxy[y * 2 + 1]), 0, IszX - 1);
                        int indY = std::clamp(static_cast<int>(roundDouble(arrayY[x]) + objxy[y * 2]), 0, IszY - 1);
                        ind[x * countOnes + y] = indX * IszY * Nfr + indY * Nfr + k;
                        if (ind[x * countOnes + y] >= max_size) {
                            ind[x * countOnes + y] = 0;
                        }
                    }
                    likelihood[x] = calcLikelihoodSum(I, ind + x * countOnes, countOnes) / countOnes;
                }
                auto likelihood_time = get_time();
                std::cout << "TIME TO GET LIKELIHOODS TOOK: " << elapsed_time(error, likelihood_time) << "\n";

                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    weights[x] *= std::exp(likelihood[x]);
                }
                auto exponential = get_time();
                std::cout << "TIME TO GET EXP TOOK: " << elapsed_time(likelihood_time, exponential) << "\n";

                double sumWeights = 0.0;
                #pragma omp parallel for reduction(+:sumWeights)
                for (int x = 0; x < Nparticles; ++x) {
                    sumWeights += weights[x];
                }
                auto sum_time = get_time();
                std::cout << "TIME TO SUM WEIGHTS TOOK: " << elapsed_time(exponential, sum_time) << "\n";

                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    weights[x] /= sumWeights;
                }
                auto normalize = get_time();
                std::cout << "TIME TO NORMALIZE WEIGHTS TOOK: " << elapsed_time(sum_time, normalize) << "\n";

                xe = 0.0;
                ye = 0.0;
                #pragma omp parallel for reduction(+:xe, ye)
                for (int x = 0; x < Nparticles; ++x) {
                    xe += arrayX[x] * weights[x];
                    ye += arrayY[x] * weights[x];
                }
                auto move_time = get_time();
                std::cout << "TIME TO MOVE OBJECT TOOK: " << elapsed_time(normalize, move_time) << "\n";
                std::cout << "XE: " << xe << "\nYE: " << ye << "\n";
                double distance = std::sqrt(std::pow(static_cast<double>(xe - roundDouble(IszY / 2.0)), 2) + 
                                          std::pow(static_cast<double>(ye - roundDouble(IszX / 2.0)), 2));
                std::cout << distance << "\n";

                CDF[0] = weights[0];
                for (int x = 1; x < Nparticles; ++x) {
                    CDF[x] = weights[x] + CDF[x - 1];
                }
                auto cum_sum = get_time();
                std::cout << "TIME TO CALC CUM SUM TOOK: " << elapsed_time(move_time, cum_sum) << "\n";

                double u1 = (1.0 / Nparticles) * randu(seed, 0);
                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    u[x] = u1 + x / static_cast<double>(Nparticles);
                }
                auto u_time = get_time();
                std::cout << "TIME TO CALC U TOOK: " << elapsed_time(cum_sum, u_time) << "\n";

                #pragma omp parallel for
                for (int j = 0; j < Nparticles; ++j) {
                    int i = findIndex(CDF, Nparticles, u[j]);
                    if (i == -1) {
                        i = Nparticles - 1;
                    }
                    xj[j] = arrayX[i];
                    yj[j] = arrayY[i];
                }
                auto xyj_time = get_time();
                std::cout << "TIME TO CALC NEW ARRAY X AND Y TOOK: " << elapsed_time(u_time, xyj_time) << "\n";

                #pragma omp parallel for
                for (int x = 0; x < Nparticles; ++x) {
                    arrayX[x] = xj[x];
                    arrayY[x] = yj[x];
                    weights[x] = 1.0 / Nparticles;
                }
                auto reset = get_time();
                std::cout << "TIME TO RESET WEIGHTS TOOK: " << elapsed_time(xyj_time, reset) << "\n";
            }

            auto endParticleFilter = get_time();
            std::cout << "PARTICLE FILTER TOOK " << elapsed_time(endVideoSequence, endParticleFilter) << "\n";
            std::cout << "ENTIRE PROGRAM TOOK " << elapsed_time(start, endParticleFilter) << "\n";
        }
    };

    void usage(const std::string& program) {
        throw std::runtime_error(
            "Usage: " + program + " -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles> -t <threads>\n"
            "  -x <dimX>: X dimension of the video\n"
            "  -y <dimY>: Y dimension of the video\n"
            "  -z <Nfr>: Number of frames\n"
            "  -np <Nparticles>: Number of particles\n"
            "  -t <threads>: Number of OpenMP threads"
        );
    }
}

int main(int argc, char* argv[]) {
    using namespace particle_filter;
    try {
        if (argc != 11 || std::strcmp(argv[1], "-x") != 0 || std::strcmp(argv[3], "-y") != 0 ||
            std::strcmp(argv[5], "-z") != 0 || std::strcmp(argv[7], "-np") != 0 || 
            std::strcmp(argv[9], "-t") != 0) {
            usage(argv[0]);
        }

        int IszX = std::stoi(argv[2]);
        int IszY = std::stoi(argv[4]);
        int Nfr = std::stoi(argv[6]);
        int Nparticles = std::stoi(argv[8]);
        int num_threads = std::stoi(argv[10]);

        ParticleFilter filter(IszX, IszY, Nfr, Nparticles, num_threads);
        filter.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}