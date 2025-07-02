g++ particle_filter.cpp -o particle_filter
./particle_filter -x 128 -y 128 -z 10 -np 10000

g++ particle_filter_omp.cpp -o particle_filter -fopenmp
./particle_filter -x 128 -y 128 -z 10 -np 10000 -t 12