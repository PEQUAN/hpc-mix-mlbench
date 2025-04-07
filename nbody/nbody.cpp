#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>

struct Body {
    double mass;
    double x, y;      // Position
    double vx, vy;    // Velocity
    double ax, ay;    // Acceleration
};

class NBodySimulation {
private:
    std::vector<Body> bodies;
    double G;         // Gravitational constant
    double dt;        // Time step
    double softening; // Softening length to avoid singularities

public:
    NBodySimulation(double g = 1.0, double timestep = 0.01, double soft = 0.01)
        : G(g), dt(timestep), softening(soft) {}

    void add_body(double m, double x, double y, double vx, double vy) {
        bodies.push_back({m, x, y, vx, vy, 0.0, 0.0});
    }

    void compute_accelerations() {
        // Reset accelerations
        for (auto& body : bodies) {
            body.ax = 0.0;
            body.ay = 0.0;
        }

        // Compute pairwise forces
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double r2 = dx * dx + dy * dy + softening * softening;
                double r = std::sqrt(r2);
                double force = G * bodies[i].mass * bodies[j].mass / r2;
                double fx = force * dx / r;
                double fy = force * dy / r;

                bodies[i].ax += fx / bodies[i].mass;
                bodies[i].ay += fy / bodies[i].mass;
                bodies[j].ax -= fx / bodies[j].mass;
                bodies[j].ay -= fy / bodies[j].mass;
            }
        }
    }

    void update_positions() {
        for (auto& body : bodies) {
            // Update velocity (Euler)
            body.vx += body.ax * dt;
            body.vy += body.ay * dt;
            // Update position
            body.x += body.vx * dt;
            body.y += body.vy * dt;
        }
    }

    double total_energy() {
        double kinetic = 0.0;
        double potential = 0.0;

        // Kinetic energy
        for (const auto& body : bodies) {
            double v2 = body.vx * body.vx + body.vy * body.vy;
            kinetic += 0.5 * body.mass * v2;
        }

        // Potential energy
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double r = std::sqrt(dx * dx + dy * dy + softening * softening);
                potential -= G * bodies[i].mass * bodies[j].mass / r;
            }
        }

        return kinetic + potential;
    }

    void simulate(int steps, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening output file" << std::endl;
            return;
        }
        file << "step,x1,y1,x2,y2,...,energy\n";

        auto start = std::chrono::high_resolution_clock::now();

        double initial_energy = total_energy();
        for (int step = 0; step < steps; ++step) {
            compute_accelerations();
            update_positions();

            // Write positions and energy every 10 steps
            if (step % 10 == 0) {
                file << step;
                for (const auto& body : bodies) {
                    file << "," << body.x << "," << body.y;
                }
                file << "," << total_energy() << "\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Number of bodies: " << bodies.size() << std::endl;
        std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;
        std::cout << "Time steps: " << steps << std::endl;
        std::cout << "Initial energy: " << initial_energy << std::endl;
        std::cout << "Final energy: " << total_energy() << std::endl;
        std::cout << "Energy drift: " << std::abs(total_energy() - initial_energy) / std::abs(initial_energy) << std::endl;
    }
};

NBodySimulation create_test_system() { 
    // Generate a simple test system
    NBodySimulation sim(1.0, 0.01, 0.01);  // G=1 for simplicity, dt=0.01, softening=0.01

    // Three-body system (scaled units)
    sim.add_body(1.0, 0.0, 0.0, 0.0, 0.0);      // Central mass
    sim.add_body(0.1, 1.0, 0.0, 0.0, 1.0);      // Orbiting body 1
    sim.add_body(0.05, 0.0, 1.0, -1.0, 0.0);    // Orbiting body 2

    return sim;
}

int main() {
    NBodySimulation sim = create_test_system();
    sim.simulate(1000, "../results/nbody/nbody_output.csv");

    return 0;
}