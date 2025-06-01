// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <cmath>

struct f3 {
    __PROMISE__ x, y, z;
};

/*
 * Options
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

/*
 * Generic functions
 */
template <typename T> T *alloc(int N) { return new T[N]; }

template <typename T> void dealloc(T *array) { delete[] array; }

template <typename T> void copy(T *dst, T *src, int N) {
    for (int i = 0; i < N; i++) {
        dst[i] = src[i];
    }
}

void dump(__PROMISE__ *variables, int nel, int nelr) {
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

void initialize_variables(int nelr, __PROMISE__ *variables, __PROMISE__ *ff_variable) {
    for (int i = 0; i < nelr; i++) {
        for (int j = 0; j < NVAR; j++)
            variables[i + j * nelr] = ff_variable[j];
    }
}

inline void compute_flux_contribution(__PROMISE__ &density, f3 &momentum,
                                     __PROMISE__ &density_energy, __PROMISE__ &pressure,
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

    __PROMISE__ de_p = density_energy + pressure;
    fc_density_energy.x = velocity.x * de_p;
    fc_density_energy.y = velocity.y * de_p;
    fc_density_energy.z = velocity.z * de_p;
}

inline void compute_velocity(__PROMISE__ &density, f3 &momentum,
                            f3 &velocity) {
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

inline __PROMISE__ compute_speed_sqd(f3 &velocity) {
    return velocity.x * velocity.x + velocity.y * velocity.y +
           velocity.z * velocity.z;
}

inline __PROMISE__ compute_pressure(__PROMISE__ &density, __PROMISE__ &density_energy,
                             __PROMISE__ &speed_sqd) {
    return (__PROMISE__(GAMMA) - __PROMISE__(1.0f)) *
           (density_energy - __PROMISE__(0.5f) * density * speed_sqd);
}

inline __PROMISE__ compute_speed_of_sound(__PROMISE__ &density, __PROMISE__ &pressure) {
    return std::sqrt(__PROMISE__(GAMMA) * pressure / density);
}

void compute_step_factor(int nelr, __PROMISE__ *variables, __PROMISE__ *areas,
                        __PROMISE__ *step_factors) {
    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
        for (int i = b_start; i < b_end; i++) {
            __PROMISE__ density = variables[i + VAR_DENSITY * nelr];

            f3 momentum;
            momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
            momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
            momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

            __PROMISE__ density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
            f3 velocity;
            compute_velocity(density, momentum, velocity);
            __PROMISE__ speed_sqd = compute_speed_sqd(velocity);
            __PROMISE__ pressure = compute_pressure(density, density_energy, speed_sqd);
            __PROMISE__ speed_of_sound = compute_speed_of_sound(density, pressure);

            step_factors[i] = __PROMISE__(0.5f) /
                (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
        }
    }
}

void compute_flux(int nelr, int *elements_surrounding_elements, __PROMISE__ *normals,
                 __PROMISE__ *variables, __PROMISE__ *fluxes, __PROMISE__ *ff_variable,
                 f3 ff_flux_contribution_momentum_x,
                 f3 ff_flux_contribution_momentum_y,
                 f3 ff_flux_contribution_momentum_z,
                 f3 ff_flux_contribution_density_energy) {
    const __PROMISE__ smoothing_coefficient = __PROMISE__(0.2f);

    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
        for (int i = b_start; i < b_end; ++i) {
            __PROMISE__ density_i = variables[i + VAR_DENSITY * nelr];
            f3 momentum_i;
            momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
            momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
            momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

            __PROMISE__ density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

            f3 velocity_i;
            compute_velocity(density_i, momentum_i, velocity_i);
            __PROMISE__ speed_sqd_i = compute_speed_sqd(velocity_i);
            __PROMISE__ speed_i = std::sqrt(speed_sqd_i);
            __PROMISE__ pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
            __PROMISE__ speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
            f3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
            f3 flux_contribution_i_density_energy;
            compute_flux_contribution(
                density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
                flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
                flux_contribution_i_momentum_z,
                flux_contribution_i_density_energy);

            __PROMISE__ flux_i_density = __PROMISE__(0.0f);
            f3 flux_i_momentum;
            flux_i_momentum.x = __PROMISE__(0.0f);
            flux_i_momentum.y = __PROMISE__(0.0f);
            flux_i_momentum.z = __PROMISE__(0.0f);
            __PROMISE__ flux_i_density_energy = __PROMISE__(0.0f);

            f3 velocity_nb;
            __PROMISE__ density_nb, density_energy_nb;
            f3 momentum_nb;
            f3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
            f3 flux_contribution_nb_density_energy;
            __PROMISE__ speed_sqd_nb, speed_of_sound_nb, pressure_nb;
            for (int j = 0; j < NNB; j++) {
                f3 normal;
                __PROMISE__ normal_len;
                __PROMISE__ factor;

                int nb = elements_surrounding_elements[i + j * nelr];
                normal.x = normals[i + (j + 0 * NNB) * nelr];
                normal.y = normals[i + (j + 1 * NNB) * nelr];
                normal.z = normals[i + (j + 2 * NNB) * nelr];
                normal_len = std::sqrt(normal.x * normal.x + normal.y * normal.y +
                                      normal.z * normal.z);

                if (nb >= 0) // a legitimate neighbor
                {
                    density_nb = variables[nb + VAR_DENSITY * nelr];
                    momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
                    momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
                    momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
                    density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
                    compute_velocity(density_nb, momentum_nb, velocity_nb);
                    speed_sqd_nb = compute_speed_sqd(velocity_nb);
                    pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
                    speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
                    compute_flux_contribution(
                        density_nb, momentum_nb, density_energy_nb, pressure_nb,
                        velocity_nb, flux_contribution_nb_momentum_x,
                        flux_contribution_nb_momentum_y,
                        flux_contribution_nb_momentum_z,
                        flux_contribution_nb_density_energy);

                    // artificial viscosity
                    factor = -normal_len * smoothing_coefficient * __PROMISE__(0.5f) *
                             (speed_i + std::sqrt(speed_sqd_nb) +
                              speed_of_sound_i + speed_of_sound_nb);
                    flux_i_density += factor * (density_i - density_nb);
                    flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
                    flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
                    flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
                    flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

                    // accumulate cell-centered fluxes
                    factor = __PROMISE__(0.5f) * normal.x;
                    flux_i_density += factor * (momentum_nb.x + momentum_i.x);
                    flux_i_density_energy += factor * (flux_contribution_nb_density_energy.x +
                                                     flux_contribution_i_density_energy.x);
                    flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.x +
                                                  flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.x +
                                                  flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.x +
                                                  flux_contribution_i_momentum_z.x);

                    factor = __PROMISE__(0.5f) * normal.y;
                    flux_i_density += factor * (momentum_nb.y + momentum_i.y);
                    flux_i_density_energy += factor * (flux_contribution_nb_density_energy.y +
                                                     flux_contribution_i_density_energy.y);
                    flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.y +
                                                  flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.y +
                                                  flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.y +
                                                  flux_contribution_i_momentum_z.y);

                    factor = __PROMISE__(0.5f) * normal.z;
                    flux_i_density += factor * (momentum_nb.z + momentum_i.z);
                    flux_i_density_energy += factor * (flux_contribution_nb_density_energy.z +
                                                     flux_contribution_i_density_energy.z);
                    flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.z +
                                                  flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.z +
                                                  flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.z +
                                                  flux_contribution_i_momentum_z.z);
                } else if (nb == -1) // a wing boundary
                {
                    flux_i_momentum.x += normal.x * pressure_i;
                    flux_i_momentum.y += normal.y * pressure_i;
                    flux_i_momentum.z += normal.z * pressure_i;
                } else if (nb == -2) // a far field boundary
                {
                    factor = __PROMISE__(0.5f) * normal.x;
                    flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
                    flux_i_density_energy += factor * (ff_flux_contribution_density_energy.x +
                                                     flux_contribution_i_density_energy.x);
                    flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.x +
                                                  flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.x +
                                                  flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.x +
                                                  flux_contribution_i_momentum_z.x);

                    factor = __PROMISE__(0.5f) * normal.y;
                    flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
                    flux_i_density_energy += factor * (ff_flux_contribution_density_energy.y +
                                                     flux_contribution_i_density_energy.y);
                    flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.y +
                                                  flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.y +
                                                  flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.y +
                                                  flux_contribution_i_momentum_z.y);

                    factor = __PROMISE__(0.5f) * normal.z;
                    flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
                    flux_i_density_energy += factor * (ff_flux_contribution_density_energy.z +
                                                     flux_contribution_i_density_energy.z);
                    flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.z +
                                                  flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.z +
                                                  flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.z +
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

void time_step(int j, int nelr, __PROMISE__ *old_variables, __PROMISE__ *variables,
               __PROMISE__ *step_factors, __PROMISE__ *fluxes) {
    for (int blk = 0; blk < nelr / block_length; ++blk) {
        int b_start = blk * block_length;
        int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
        for (int i = b_start; i < b_end; ++i) {
            __PROMISE__ factor = step_factors[i] / __PROMISE__(RK + 1 - j);

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

/*
 * Main function
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "specify data file name" << std::endl;
        return 0;
    }
    const char *data_file_name = argv[1];

    // Set block_length to a reasonable default since omp_get_max_threads() is removed
    block_length = 256;

    __PROMISE__ ff_variable[NVAR];
    f3 ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
       ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy;

    // set far field conditions
    {
        const __PROMISE__ angle_of_attack =
            __PROMISE__(3.1415926535897931 / 180.0f) * __PROMISE__(deg_angle_of_attack);

        ff_variable[VAR_DENSITY] = __PROMISE__(1.4);

        __PROMISE__ ff_pressure = __PROMISE__(1.0f);
        __PROMISE__ ff_speed_of_sound =
            sqrt(GAMMA * ff_pressure / ff_variable[VAR_DENSITY]);
        __PROMISE__ ff_speed = __PROMISE__(ff_mach) * ff_speed_of_sound;

        f3 ff_velocity;
        ff_velocity.x = ff_speed * __PROMISE__(cos((__PROMISE__)angle_of_attack));
        ff_velocity.y = ff_speed * __PROMISE__(sin((__PROMISE__)angle_of_attack));
        ff_velocity.z = 0.0f;

        ff_variable[VAR_MOMENTUM + 0] =
            ff_variable[VAR_DENSITY] * ff_velocity.x;
        ff_variable[VAR_MOMENTUM + 1] =
            ff_variable[VAR_DENSITY] * ff_velocity.y;
        ff_variable[VAR_MOMENTUM + 2] =
            ff_variable[VAR_DENSITY] * ff_velocity.z;

        ff_variable[VAR_DENSITY_ENERGY] =
            ff_variable[VAR_DENSITY] * (__PROMISE__(0.5f) * (ff_speed * ff_speed)) +
            (ff_pressure / __PROMISE__(GAMMA - 1.0f));

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
    __PROMISE__ *areas;
    int *elements_surrounding_elements;
    __PROMISE__ *normals;
    {
        std::ifstream file(data_file_name);

        file >> nel;
        nelr = block_length *
               ((nel / block_length) + std::min(1, nel % block_length));

        areas = new __PROMISE__[nelr];
        elements_surrounding_elements = new int[nelr * NNB];
        normals = new __PROMISE__[NDIM * NNB * nelr];

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

    // Create arrays and set initial conditions
    __PROMISE__ *variables = alloc<__PROMISE__>(nelr * NVAR);
    initialize_variables(nelr, variables, ff_variable);

    __PROMISE__ *old_variables = alloc<__PROMISE__>(nelr * NVAR);
    __PROMISE__ *fluxes = alloc<__PROMISE__>(nelr * NVAR);
    __PROMISE__ *step_factors = alloc<__PROMISE__>(nelr);

    // these need to be computed the first time in order to compute time step
    std::cout << "Starting..." << std::endl;

    // Begin iterations
    for (int i = 0; i < iterations; i++) {
        copy<__PROMISE__>(old_variables, variables, nelr * NVAR);

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
    std::cout << "Saving solution..." << std::endl;
    dump(variables, nel, nelr);
    std::cout << "Saved solution..." << std::endl;

    std::cout << "Cleaning up..." << std::endl;
    PROMISE_CHECK_ARRAY(areas, nelr);
    dealloc<__PROMISE__>(areas);
    dealloc<int>(elements_surrounding_elements);
    dealloc<__PROMISE__>(normals);

    dealloc<__PROMISE__>(variables);
    dealloc<__PROMISE__>(old_variables);
    dealloc<__PROMISE__>(fluxes);
    dealloc<__PROMISE__>(step_factors);

    std::cout << "Done..." << std::endl;

    return 0;
}