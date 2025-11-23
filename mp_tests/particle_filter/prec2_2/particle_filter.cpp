#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <array>
#include <limits>

namespace particle_filter {
    float PI = 3.1415926535897932;
    long M = 99999999;
    int A = 1103515245;
    int C = 12345;
    int DISK_RADIUS = 5;
    int FIXED_SEED = 123456789; // Fixed seed for reproducibility

    class ParticleFilter {
    private:
        int IszX, IszY, Nfr, Nparticles;
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
        flx::floatx<5, 2> xe, ye;

        using TimePoint = std::chrono::high_resolution_clock::time_point;

        TimePoint get_time() {
            return std::chrono::high_resolution_clock::now();
        }

        flx::floatx<5, 2> elapsed_time(const TimePoint& start, const TimePoint& end) {
            return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);
        }

        flx::floatx<8, 7> roundDouble(float value) {
            int newValue = static_cast<int>(value);
            return (value - newValue < 0.5) ? newValue : newValue + 1;
        }

        void setIf(int testValue, int newValue, int* array3D, int dimX, int dimY, int dimZ) {
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

        float randu(int* seed, int index) {
            int num = A * seed[index] + C;
            seed[index] = num % M;
            return abs(seed[index] / static_cast<float>(M));
        }

        float randn(int* seed, int index) {
            float u = randu(seed, index);
            float v = randu(seed, index);
            float cosine = cos(2 * PI * v);
            float rt = -2 * log(u);
            return sqrt(rt) * cosine;
        }

        void addNoise(int* array3D, int dimX, int dimY, int dimZ) {
            for (int x = 0; x < dimX; ++x) {
                for (int y = 0; y < dimY; ++y) {
                    for (int z = 0; z < dimZ; ++z) {
                        array3D[x * dimY * dimZ + y * dimZ + z] += static_cast<int>(5 * randn(seed, 0));
                    }
                }
            }
        }

        void strelDisk(int* disk, int radius) {
            int diameter = radius * 2 - 1;
            for (int x = 0; x < diameter; ++x) {
                for (int y = 0; y < diameter; ++y) {
                    double temp = 2.0;
                    flx::floatx<5, 2> distance = sqrt(pow(static_cast<double>(x - radius + 1), temp) + 
                                              pow(static_cast<double>(y - radius + 1), temp));
                    if (distance < radius) {
                        disk[x * diameter + y] = 1;
                    }
                }
            }
        }

        void dilate_matrix(int* matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
            int startX = max(posX - error, 0);
            int startY = max(posY - error, 0);
            int endX = min(posX + error, dimX);
            int endY = min(posY + error, dimY);
            double temp = 2.0;
            for (int x = startX; x < endX; ++x) {
                for (int y = startY; y < endY; ++y) {
                    flx::floatx<5, 2> distance = sqrt(pow(static_cast<double>(x - posX), temp) + 
                                              pow(static_cast<double>(y - posY), temp));
                    if (distance < error) {
                        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
                    }
                }
            }
        }

        void imdilate_disk(int* matrix, int dimX, int dimY, int dimZ, int error, int* newMatrix) {
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
                int xk = abs(x0 + (k - 1));
                int yk = abs(y0 - 2 * (k - 1));
                int pos = yk * IszY * Nfr + xk * Nfr + k;
                if (pos >= max_size) {
                    pos = 0;
                }
                I[pos] = 1;
            }

            int* newMatrix = new int[IszX * IszY * Nfr]();
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

        flx::floatx<5, 2> calcLikelihoodSum(int* I, int* ind, int numOnes) {
            flx::floatx<8, 7> likelihoodSum = 0.0;
            double temp = 2.0;
            for (int y = 0; y < numOnes; ++y) {
                likelihoodSum += (pow(static_cast<double>(I[ind[y]] - 100), temp) - 
                                pow(static_cast<double>(I[ind[y]] - 228), temp)) / 50.0;
            }
            return likelihoodSum;
        }

        int findIndex(double* CDF, int lengthCDF, float value) {
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
        ParticleFilter(int x, int y, int frames, int particles)
            : IszX(x), IszY(y), Nfr(frames), Nparticles(particles), 
              xe(roundDouble(y / 2.0)), ye(roundDouble(x / 2.0)),
              I(nullptr), seed(nullptr), disk(nullptr), objxy(nullptr),
              weights(nullptr), likelihood(nullptr), arrayX(nullptr), arrayY(nullptr),
              xj(nullptr), yj(nullptr), CDF(nullptr), u(nullptr), ind(nullptr) {
            if (IszX <= 0 || IszY <= 0 || Nfr <= 0 || Nparticles <= 0) {
                throw std::runtime_error("Invalid input parameters");
            }

            try {
                I = new int[IszX * IszY * Nfr]();
                seed = new int[Nparticles]();
                for (int i = 0; i < Nparticles; ++i) {
                    seed[i] = FIXED_SEED + i; // Fixed seed with offset for reproducibility
                }

                int diameter = DISK_RADIUS * 2 - 1;
                disk = new int[diameter * diameter]();
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
                getneighbors(disk, countOnes, objxy, DISK_RADIUS);

                weights = new double[Nparticles]();
                likelihood = new double[Nparticles]();
                arrayX = new double[Nparticles]();
                arrayY = new double[Nparticles]();
                xj = new double[Nparticles]();
                yj = new double[Nparticles]();
                CDF = new double[Nparticles]();
                u = new double[Nparticles]();
                ind = new int[countOnes * Nparticles]();
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

            for (int x = 0; x < Nparticles; ++x) {
                weights[x] = 1.0 / Nparticles;
            }
            auto get_weights = get_time();
            std::cout << "TIME TO GET WEIGHTS TOOK: " << elapsed_time(get_neighbors, get_weights) << "\n";

            for (int x = 0; x < Nparticles; ++x) {
                arrayX[x] = xe;
                arrayY[x] = ye;
            }
            auto set_arrays = get_time();
            std::cout << "TIME TO SET ARRAYS TOOK: " << elapsed_time(get_weights, set_arrays) << "\n";

            int max_size = IszX * IszY * Nfr;
            for (int k = 1; k < Nfr; ++k) {
                auto set_arrays_time = get_time();
                for (int x = 0; x < Nparticles; ++x) {
                    arrayX[x] += 1 + 5 * randn(seed, x);
                    arrayY[x] += -2 + 2 * randn(seed, x);
                }
                auto error = get_time();
                std::cout << "TIME TO SET ERROR TOOK: " << elapsed_time(set_arrays_time, error) << "\n";

                for (int x = 0; x < Nparticles; ++x) {
                    for (int y = 0; y < countOnes; ++y) {
                        int indX = static_cast<int>(roundDouble(arrayX[x]) + objxy[y * 2 + 1]);
                        int indY = static_cast<int>(roundDouble(arrayY[x]) + objxy[y * 2]);
                        ind[x * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
                        if (ind[x * countOnes + y] >= max_size) {
                            ind[x * countOnes + y] = 0;
                        }
                    }
                    likelihood[x] = calcLikelihoodSum(I, ind + x * countOnes, countOnes) / countOnes;
                }
                auto likelihood_time = get_time();
                std::cout << "TIME TO GET LIKELIHOODS TOOK: " << elapsed_time(error, likelihood_time) << "\n";

                for (int x = 0; x < Nparticles; ++x) {
                    weights[x] *= exp(likelihood[x]);
                }
                auto exponential = get_time();
                std::cout << "TIME TO GET EXP TOOK: " << elapsed_time(likelihood_time, exponential) << "\n";

                double sumWeights = 0.0;
                for (int x = 0; x < Nparticles; ++x) {
                    sumWeights += weights[x];
                }
                auto sum_time = get_time();
                std::cout << "TIME TO SUM WEIGHTS TOOK: " << elapsed_time(exponential, sum_time) << "\n";

                for (int x = 0; x < Nparticles; ++x) {
                    weights[x] /= sumWeights;
                }
                auto normalize = get_time();
                std::cout << "TIME TO NORMALIZE WEIGHTS TOOK: " << elapsed_time(sum_time, normalize) << "\n";

                xe = 0.0;
                ye = 0.0;
                for (int x = 0; x < Nparticles; ++x) {
                    xe += arrayX[x] * weights[x];
                    ye += arrayY[x] * weights[x];
                }
                auto move_time = get_time();
                std::cout << "TIME TO MOVE OBJECT TOOK: " << elapsed_time(normalize, move_time) << "\n";
                std::cout << "XE: " << xe << "\nYE: " << ye << "\n";
                
                double temp = 2.0;
                flx::floatx<5, 2> distance = sqrt(pow(static_cast<double>(xe - roundDouble(IszY / 2.0)), temp) + 
                                          pow(static_cast<double>(ye - roundDouble(IszX / 2.0)), temp));
                std::cout << distance << "\n";

                CDF[0] = weights[0];
                for (int x = 1; x < Nparticles; ++x) {
                    CDF[x] = weights[x] + CDF[x - 1];
                }
                auto cum_sum = get_time();
                std::cout << "TIME TO CALC CUM SUM TOOK: " << elapsed_time(move_time, cum_sum) << "\n";

                flx::floatx<5, 2> u1 = (1.0 / Nparticles) * randu(seed, 0);
                for (int x = 0; x < Nparticles; ++x) {
                    u[x] = u1 + x / static_cast<float>(Nparticles);
                }
                auto u_time = get_time();
                std::cout << "TIME TO CALC U TOOK: " << elapsed_time(cum_sum, u_time) << "\n";

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
            
            double *check_arrayX = new double[Nparticles];
            for (int x = 0; x < Nparticles; ++x) {
                check_arrayX[x] = arrayX[x];
            }
            PROMISE_CHECK_ARRAY(check_arrayX, Nparticles);
        }
    };

    void usage(const std::string& program) {
        throw std::runtime_error(
            "Usage: " + program + " -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>\n"
            "  -x <dimX>: X dimension of the video\n"
            "  -y <dimY>: Y dimension of the video\n"
            "  -z <Nfr>: Number of frames\n"
            "  -np <Nparticles>: Number of particles"
        );
    }
}

int main(int argc, char* argv[]) {
    using namespace particle_filter;
    try {
        if (argc != 9 || std::strcmp(argv[1], "-x") != 0 || std::strcmp(argv[3], "-y") != 0 ||
            std::strcmp(argv[5], "-z") != 0 || std::strcmp(argv[7], "-np") != 0) {
            usage(argv[0]);
        }

        int IszX = std::stoi(argv[2]);
        int IszY = std::stoi(argv[4]);
        int Nfr = std::stoi(argv[6]);
        int Nparticles = std::stoi(argv[8]);

        ParticleFilter filter(IszX, IszY, Nfr, Nparticles);
        filter.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}