g++ hotspot.cpp -o hotspot -fopenmp
./hotspot 512 512 2 42 temp_512 power_512 output.out

g++ hotspot_omp.cpp -o hotspot -fopenmp
./hotspot 512 512 2 42 temp_512 power_512 output.out