cmake_minimum_required(VERSION 3.10)
project(PCA)


set(LAPACKE_LIB /lib/x86_64-linux-gnu/liblapacke64.so)
set(OPENBLAS_LIB /lib/x86_64-linux-gnu/libopenblas.so)

add_executable(pca pca.cpp)
target_link_libraries(pca ${LAPACKE_LIB} ${OPENBLAS_LIB})