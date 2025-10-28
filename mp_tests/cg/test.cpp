#include <iostream>
#include <random>

int main() {
    std::mt19937 gen(1000);  // Seed the random number generator
    std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform distribution in [0, 1)

    double b = dis(gen);  // Generate random value for b in [0, 1)
    std::cout << "Random value b = " << b << std::endl;

    return 0;
}