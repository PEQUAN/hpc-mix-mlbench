g++ -O2 -std=c++17 matrix_check.cpp -o matrix_check
./matrix_check rdb5000.mtx
echo "-----------------------------------"
./matrix_check psmigr_2.mtx
echo "-----------------------------------"
./matrix_check gre_512.mtx
echo "-----------------------------------"
./matrix_check 1138_bus.mtx
echo "-----------------------------------"
./matrix_check bp_0.mtx