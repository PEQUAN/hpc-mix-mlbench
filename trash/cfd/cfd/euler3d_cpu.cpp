// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
#include <cmath>
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif

#include <omp.h>

struct f3 {
    double x, y, z;
};
/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3 // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

static int block_length;


#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

/*
 * Generic functions
 */
template <typename T> T *alloc(int N) { return new T[N]; }

template <typename T> void dealloc(T *array) { delete[] array; }

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
template <typename T> void copy(T *dst, T *src, int N) {
#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < N; i++) {
        dst[i] = src[i];
    }
}
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif


void dump(double *variables, int nel, int nelr) {


    {
        std::ofstream file("density");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++)
            file << variables[i + VAR_DENSITY * nelr] << std::endl;
    }


    {
        std::ofstream file("momentum");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++) {
            for (int j = 0; j != NDIM; j++)
                file << variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            file << std::endl;
        }
    }

    {
        std::ofstream file("density_energy");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++)
            file << variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
    }
}

void initialize_variables(int nelr, double *variables, double *ff_variable) {
#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < nelr; i++) {
        for (int j = 0; j < NVAR; j++)
            variables[i + j * nelr] = ff_variable[j];
    }
}

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
inline void compute_flux_contribution(double &density, f3 &momentum,
                                      double &density_energy, double &pressure,
                                      f3 &velocity, f3 &fc_momentum_x,
                                      f3 &fc_momentum_y,
                                      f3 &fc_momentum_z,
                                      f3 &fc_density_energy) {
    fc_momentum_x.x = velocity.x * momentum.x + pressure;
    fc_momentum_x.y = velocity.x * momentum.y;
    fc_momentum_x.z = velocity.x * momentum.z;

    fc_momentum_y.x = fc_momentum_x.y;
    fc_momentum_y.y = velocity.y * momentum.y + pressure;
    fc_momentum_y.z = velocity.y * momentum.z;

    fc_momentum_z.x = fc_momentum_x.z;
    fc_momentum_z.y = fc_momentum_y.z;
    fc_momentum_z.z = velocity.z * momentum.z + pressure;

    double de_p = density_energy + pressure;
    fc_density_energy.x = velocity.x * de_p;
    fc_density_energy.y = velocity.y * de_p;
    fc_density_energy.z = velocity.z * de_p;
}

inline void compute_velocity(double &density, f3 &momentum,
                             f3 &velocity) {
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

inline double compute_speed_sqd(f3 &velocity) {
    return velocity.x * velocity.x + velocity.y * velocity.y +
           velocity.z * velocity.z;
}

inline double compute_pressure(double &density, double &density_energy,
                              double &speed_sqd) {
    return (double(GAMMA) - double(1.0f)) *
           (density_energy - double(0.5f) * density * speed_sqd);
}

inline double compute_speed_of_sound(double &density, double &pressure) {
    return sqrt(double(GAMMA) * pressure / density);
}


void compute_step_factor(int nelr, double *__restrict variables, double *areas,
                         double *__restrict step_factors) {
#pragma omp parallel for default(shared) schedule(auto)
    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end =
            (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
#pragma omp simd
        for (int i = b_start; i < b_end; i++) {
            double density = variables[i + VAR_DENSITY * nelr];

            f3 momentum;
            momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
            momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
            momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

            double density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
            f3 velocity;
            compute_velocity(density, momentum, velocity);
            double speed_sqd = compute_speed_sqd(velocity);
            double pressure =
                compute_pressure(density, density_energy, speed_sqd);
            double speed_of_sound = compute_speed_of_sound(density, pressure);

            // dt = double(0.5f) * sqrt(areas[i]) /  (||v|| + c).... but
            // when we do time stepping, this later would need to be divided by
            // the area, so we just do it all at once
            step_factors[i] =
                double(0.5f) /
                (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
        }
    }
}


/*
 *
 *
*/

void compute_flux(int nelr, int *elements_surrounding_elements, double *normals,
                  double *variables, double *fluxes, double *ff_variable,
                  f3 ff_flux_contribution_momentum_x,
                  f3 ff_flux_contribution_momentum_y,
                  f3 ff_flux_contribution_momentum_z,
                  f3 ff_flux_contribution_density_energy) {
    const double smoothing_coefficient = double(0.2f);

#pragma omp parallel for default(shared) schedule(auto)
    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end =
            (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
#pragma omp simd
        for (int i = b_start; i < b_end; ++i) {
            double density_i = variables[i + VAR_DENSITY * nelr];
            f3 momentum_i;
            momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
            momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
            momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

            double density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

            f3 velocity_i;
            compute_velocity(density_i, momentum_i, velocity_i);
            double speed_sqd_i = compute_speed_sqd(velocity_i);
            double speed_i = sqrt(speed_sqd_i);
            double pressure_i =
                compute_pressure(density_i, density_energy_i, speed_sqd_i);
            double speed_of_sound_i =
                compute_speed_of_sound(density_i, pressure_i);
            f3 flux_contribution_i_momentum_x,
                flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
            f3 flux_contribution_i_density_energy;
            compute_flux_contribution(
                density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
                flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
                flux_contribution_i_momentum_z,
                flux_contribution_i_density_energy);

            double flux_i_density = double(0.0f);
            f3 flux_i_momentum;
            flux_i_momentum.x = double(0.0f);
            flux_i_momentum.y = double(0.0f);
            flux_i_momentum.z = double(0.0f);
            double flux_i_density_energy = double(0.0f);

            f3 velocity_nb;
            double density_nb, density_energy_nb;
            f3 momentum_nb;
            f3 flux_contribution_nb_momentum_x,
                flux_contribution_nb_momentum_y,
                flux_contribution_nb_momentum_z;
            f3 flux_contribution_nb_density_energy;
            double speed_sqd_nb, speed_of_sound_nb, pressure_nb;
#pragma unroll
            for (int j = 0; j < NNB; j++) {
                f3 normal;
                double normal_len;
                double factor;

                int nb = elements_surrounding_elements[i + j * nelr];
                normal.x = normals[i + (j + 0 * NNB) * nelr];
                normal.y = normals[i + (j + 1 * NNB) * nelr];
                normal.z = normals[i + (j + 2 * NNB) * nelr];
                normal_len =
                    sqrt(normal.x * normal.x + normal.y * normal.y +
                              normal.z * normal.z);

                if (nb >= 0) // a legitimate neighbor
                {
                    density_nb = variables[nb + VAR_DENSITY * nelr];
                    momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
                    momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
                    momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
                    density_energy_nb =
                        variables[nb + VAR_DENSITY_ENERGY * nelr];
                    compute_velocity(density_nb, momentum_nb, velocity_nb);
                    speed_sqd_nb = compute_speed_sqd(velocity_nb);
                    pressure_nb = compute_pressure(
                        density_nb, density_energy_nb, speed_sqd_nb);
                    speed_of_sound_nb =
                        compute_speed_of_sound(density_nb, pressure_nb);
                    compute_flux_contribution(
                        density_nb, momentum_nb, density_energy_nb, pressure_nb,
                        velocity_nb, flux_contribution_nb_momentum_x,
                        flux_contribution_nb_momentum_y,
                        flux_contribution_nb_momentum_z,
                        flux_contribution_nb_density_energy);

                    // artificial viscosity
                    factor = -normal_len * smoothing_coefficient * double(0.5f) *
                             (speed_i + sqrt(speed_sqd_nb) +
                              speed_of_sound_i + speed_of_sound_nb);
                    flux_i_density += factor * (density_i - density_nb);
                    flux_i_density_energy +=
                        factor * (density_energy_i - density_energy_nb);
                    flux_i_momentum.x +=
                        factor * (momentum_i.x - momentum_nb.x);
                    flux_i_momentum.y +=
                        factor * (momentum_i.y - momentum_nb.y);
                    flux_i_momentum.z +=
                        factor * (momentum_i.z - momentum_nb.z);

                    // accumulate cell-centered fluxes
                    factor = double(0.5f) * normal.x;
                    flux_i_density += factor * (momentum_nb.x + momentum_i.x);
                    flux_i_density_energy +=
                        factor * (flux_contribution_nb_density_energy.x +
                                  flux_contribution_i_density_energy.x);
                    flux_i_momentum.x +=
                        factor * (flux_contribution_nb_momentum_x.x +
                                  flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y +=
                        factor * (flux_contribution_nb_momentum_y.x +
                                  flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z +=
                        factor * (flux_contribution_nb_momentum_z.x +
                                  flux_contribution_i_momentum_z.x);

                    factor = double(0.5f) * normal.y;
                    flux_i_density += factor * (momentum_nb.y + momentum_i.y);
                    flux_i_density_energy +=
                        factor * (flux_contribution_nb_density_energy.y +
                                  flux_contribution_i_density_energy.y);
                    flux_i_momentum.x +=
                        factor * (flux_contribution_nb_momentum_x.y +
                                  flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y +=
                        factor * (flux_contribution_nb_momentum_y.y +
                                  flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z +=
                        factor * (flux_contribution_nb_momentum_z.y +
                                  flux_contribution_i_momentum_z.y);

                    factor = double(0.5f) * normal.z;
                    flux_i_density += factor * (momentum_nb.z + momentum_i.z);
                    flux_i_density_energy +=
                        factor * (flux_contribution_nb_density_energy.z +
                                  flux_contribution_i_density_energy.z);
                    flux_i_momentum.x +=
                        factor * (flux_contribution_nb_momentum_x.z +
                                  flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y +=
                        factor * (flux_contribution_nb_momentum_y.z +
                                  flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z +=
                        factor * (flux_contribution_nb_momentum_z.z +
                                  flux_contribution_i_momentum_z.z);
                } else if (nb == -1) // a wing boundary
                {
                    flux_i_momentum.x += normal.x * pressure_i;
                    flux_i_momentum.y += normal.y * pressure_i;
                    flux_i_momentum.z += normal.z * pressure_i;
                } else if (nb == -2) // a far field boundary
                {
                    factor = double(0.5f) * normal.x;
                    flux_i_density +=
                        factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
                    flux_i_density_energy +=
                        factor * (ff_flux_contribution_density_energy.x +
                                  flux_contribution_i_density_energy.x);
                    flux_i_momentum.x +=
                        factor * (ff_flux_contribution_momentum_x.x +
                                  flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y +=
                        factor * (ff_flux_contribution_momentum_y.x +
                                  flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z +=
                        factor * (ff_flux_contribution_momentum_z.x +
                                  flux_contribution_i_momentum_z.x);

                    factor = double(0.5f) * normal.y;
                    flux_i_density +=
                        factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
                    flux_i_density_energy +=
                        factor * (ff_flux_contribution_density_energy.y +
                                  flux_contribution_i_density_energy.y);
                    flux_i_momentum.x +=
                        factor * (ff_flux_contribution_momentum_x.y +
                                  flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y +=
                        factor * (ff_flux_contribution_momentum_y.y +
                                  flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z +=
                        factor * (ff_flux_contribution_momentum_z.y +
                                  flux_contribution_i_momentum_z.y);

                    factor = double(0.5f) * normal.z;
                    flux_i_density +=
                        factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
                    flux_i_density_energy +=
                        factor * (ff_flux_contribution_density_energy.z +
                                  flux_contribution_i_density_energy.z);
                    flux_i_momentum.x +=
                        factor * (ff_flux_contribution_momentum_x.z +
                                  flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y +=
                        factor * (ff_flux_contribution_momentum_y.z +
                                  flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z +=
                        factor * (ff_flux_contribution_momentum_z.z +
                                  flux_contribution_i_momentum_z.z);
                }
            }
            fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
            fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
            fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
            fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
            fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
        }
    }
}

void time_step(int j, int nelr, double *old_variables, double *variables,
               double *step_factors, double *fluxes) {
#pragma omp parallel for default(shared) schedule(auto)
    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end =
            (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
#pragma omp simd
        for (int i = b_start; i < b_end; ++i) {
            double factor = step_factors[i] / double(RK + 1 - j);

            variables[i + VAR_DENSITY * nelr] =
                old_variables[i + VAR_DENSITY * nelr] +
                factor * fluxes[i + VAR_DENSITY * nelr];
            variables[i + (VAR_MOMENTUM + 0) * nelr] =
                old_variables[i + (VAR_MOMENTUM + 0) * nelr] +
                factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
            variables[i + (VAR_MOMENTUM + 1) * nelr] =
                old_variables[i + (VAR_MOMENTUM + 1) * nelr] +
                factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
            variables[i + (VAR_MOMENTUM + 2) * nelr] =
                old_variables[i + (VAR_MOMENTUM + 2) * nelr] +
                factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
            variables[i + VAR_DENSITY_ENERGY * nelr] =
                old_variables[i + VAR_DENSITY_ENERGY * nelr] +
                factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
        }
    }
}
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif
/*
 * Main function
 */

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "specify data file name" << std::endl;
        return 0;
    }
    const char *data_file_name = argv[1];

    block_length = omp_get_max_threads();

    __PROMISE__ ff_variable[NVAR];
    f3 ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
        ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy;

    // set far field conditions
    {
        const __PROMISE__ angle_of_attack =
            double(3.1415926535897931 / 180.0f) * double(deg_angle_of_attack);

        ff_variable[VAR_DENSITY] = double(1.4);

        __PROMISE__ ff_pressure = double(1.0f);
        __PROMISE__ ff_speed_of_sound =
            sqrt(GAMMA * ff_pressure / ff_variable[VAR_DENSITY]);
            __PROMISE__ ff_speed = double(ff_mach) * ff_speed_of_sound;

        f3 ff_velocity;
        ff_velocity.x = ff_speed * double(cos((double)angle_of_attack));
        ff_velocity.y = ff_speed * double(sin((double)angle_of_attack));
        ff_velocity.z = 0.0f;

        ff_variable[VAR_MOMENTUM + 0] =
            ff_variable[VAR_DENSITY] * ff_velocity.x;
        ff_variable[VAR_MOMENTUM + 1] =
            ff_variable[VAR_DENSITY] * ff_velocity.y;
        ff_variable[VAR_MOMENTUM + 2] =
            ff_variable[VAR_DENSITY] * ff_velocity.z;

        ff_variable[VAR_DENSITY_ENERGY] =
            ff_variable[VAR_DENSITY] * (double(0.5f) * (ff_speed * ff_speed)) +
            (ff_pressure / double(GAMMA - 1.0f));

        f3 ff_momentum;
        ff_momentum.x = *(ff_variable + VAR_MOMENTUM + 0);
        ff_momentum.y = *(ff_variable + VAR_MOMENTUM + 1);
        ff_momentum.z = *(ff_variable + VAR_MOMENTUM + 2);
        compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum,
                                  ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
                                  ff_velocity, ff_flux_contribution_momentum_x,
                                  ff_flux_contribution_momentum_y,
                                  ff_flux_contribution_momentum_z,
                                  ff_flux_contribution_density_energy);
    }
    int nel;
    int nelr;


    // read in domain geometry
    __PR_1__ *areas;
    int *elements_surrounding_elements;
    __PR_2__ *normals;
    {
        std::ifstream file(data_file_name);

        file >> nel;
        nelr = block_length *
               ((nel / block_length) + std::min(1, nel % block_length));

        areas = new __PR_1__[nelr];
        elements_surrounding_elements = new int[nelr * NNB];
        normals = new __PR_2__[NDIM * NNB * nelr];

        // read in data
        for (int i = 0; i < nel; i++) {
            file >> areas[i];
            for (int j = 0; j < NNB; j++) {
                file >> elements_surrounding_elements[i + j * nelr];
                if (elements_surrounding_elements[i + j * nelr] < 0)
                    elements_surrounding_elements[i + j * nelr] = -1;
                elements_surrounding_elements[i + j * nelr]--; // it's coming in
                                                               // with Fortran
                                                               // numbering

                for (int k = 0; k < NDIM; k++) {
                    file >> normals[i + (j + k * NNB) * nelr];
                    normals[i + (j + k * NNB) * nelr] =
                        -normals[i + (j + k * NNB) * nelr];
                }
            }
        }

        // fill in remaining data
        int last = nel - 1;
        for (int i = nel; i < nelr; i++) {
            areas[i] = areas[last];
            for (int j = 0; j < NNB; j++) {
                // duplicate the last element
                elements_surrounding_elements[i + j * nelr] =
                    elements_surrounding_elements[last + j * nelr];
                for (int k = 0; k < NDIM; k++)
                    normals[i + (j + k * NNB) * nelr] =
                        normals[last + (j + k * NNB) * nelr];
            }
        }

        
    }

    PROMISE_CHECK_ARRAY(areas, nelr);
    // Create arrays and set initial conditions
    double *variables = alloc<double>(nelr * NVAR);
    initialize_variables(nelr, variables, ff_variable);

    double *old_variables = alloc<double>(nelr * NVAR);
    double *fluxes = alloc<double>(nelr * NVAR);
    double *step_factors = alloc<double>(nelr);

    // these need to be computed the first time in order to compute time step
    std::cout << "Starting..." << std::endl;
#ifdef _OPENMP
    double start = omp_get_wtime();
#ifdef OMP_OFFLOAD
#pragma omp target map(alloc : old_variables[0 : (nelr *NVAR)]) map(           \
    to : nelr,                                                                 \
    areas[0 : nelr],                                                           \
          step_factors[0 : nelr], elements_surrounding_elements                \
                       [0 : (nelr *NNB)],                                      \
                        normals[0 : (NDIM *NNB *nelr)],                        \
                                fluxes[0 : (nelr *NVAR)], ff_variable          \
                                       [0 : NVAR],                             \
                                        ff_flux_contribution_momentum_x,       \
                                        ff_flux_contribution_momentum_y,       \
                                        ff_flux_contribution_momentum_z,       \
                                        ff_flux_contribution_density_energy)   \
                                           map(variables[0 : (nelr *NVAR)])
#endif
#endif
    // Begin iterations
    for (int i = 0; i < iterations; i++) {
        copy<double>(old_variables, variables, nelr * NVAR);

        // for the first iteration we compute the time step
        compute_step_factor(nelr, variables, areas, step_factors);

        for (int j = 0; j < RK; j++) {
            compute_flux(nelr, elements_surrounding_elements, normals,
                         variables, fluxes, ff_variable,
                         ff_flux_contribution_momentum_x,
                         ff_flux_contribution_momentum_y,
                         ff_flux_contribution_momentum_z,
                         ff_flux_contribution_density_energy);
            time_step(j, nelr, old_variables, variables, step_factors, fluxes);
        }
    }

#ifdef _OPENMP
    double end = omp_get_wtime();
    std::cout << "Compute time: " << (end - start) << std::endl;
#endif


    if (getenv("OUTPUT")) {
        std::cout << "Saving solution..." << std::endl;
        dump(variables, nel, nelr);
        std::cout << "Saved solution..." << std::endl;
    }


    std::cout << "Cleaning up..." << std::endl;
    dealloc<double>(areas);
    dealloc<int>(elements_surrounding_elements);
    dealloc<double>(normals);

    dealloc<double>(variables);
    dealloc<double>(old_variables);
    dealloc<double>(fluxes);
    dealloc<double>(step_factors);

    std::cout << "Done..." << std::endl;

    return 0;
}