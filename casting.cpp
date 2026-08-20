#include <iostream>
#include <cstdint>
using namespace std;

int main() {
    int* p = new int(100);
    long c = reinterpret_cast<long>(p);
    uintptr_t i = reinterpret_cast<uintptr_t>(p);
    cout << "long c = " << c << " , unsigned int i = " << i << '\n';

    int* tmp1 = reinterpret_cast<int*>(c);
    int* tmp2 = reinterpret_cast<int*>(i);
    cout << "tmp1 : " << *tmp1 << " , tmp2 : " << *tmp2 << '\n';

    return 0;
}
