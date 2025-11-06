#include <vector>
#include <iostream>

int main(){
    std::vector<__PROMISE__> num1 = {1, 2, 3};

    std::vector<__PROMISE__> num2 = {2, 5, 8};

    std::vector<__PROMISE__> sum(0);

    for(int i=0; i<3; i++){
        sum.push_back(num1[i] + num2[i]);
    }

    for(int i=0; i<3; i++){
        std::cout << sum[i] << " ";
    }

    PROMISE_CHECK_ARRAY(sum.data(), 3);
    std::cout << std::endl;
    return 0;
}