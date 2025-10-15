#include "moqui_rev/library.hpp"
#include <iostream>

int main() {
    std::cout << "Hello from main!" << std::endl;

    moqui_rev::Library lib;
    lib.hello();
    int result = lib.add(5, 3);
    std::cout << "5 + 3 = " << result << std::endl;

    return 0;
}
