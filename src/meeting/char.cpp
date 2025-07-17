#include <iostream>

void printMessage(const char* msg) {
    std::cout << msg << std::endl;
}

const char* check_char(){
    return "hello";
}

int main() {
    // char* str = "Hello, world!";
    // str[0] = 'h';
    const char* str = "Hello, world!";
    printMessage(str);  
    std::cout << check_char();
    return 0;
}
