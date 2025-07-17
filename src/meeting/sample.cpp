
#include <vector>
#include <map>

namespace ns {
    struct MyClass {
        int val;
    };
}

template<typename T>
struct XIN {
    T data;
    XIN operator+(const XIN& other) {
        XIN result;
        result.data = this->data + other.data;
        return result;
    }
};

int compute(int a, double b) {
    return a + static_cast<int>(b);
}

int main() {
    std::vector<double> v1;
    XIN<double> c1, c2;
    int i = 5;
    int a = i;
    double d = 2.9;
    bool b = i == d;
    XIN<double> c3 = c1 + c2;
    d = i + d;
    v1 = std::vector<double>();
    c1.data = d;
    int x = a * b + i;  // Complex expression
    int y, z;
    x = y = z = 10;     // Chained assignment
    ns::MyClass* ptr = new ns::MyClass();
    ptr->val = x;       // Pointer access
    int& ref = i;       // Reference
    int w = compute(a, d);  // Function call
    int u = *ptr;       // Dereference
    return 0;
}
