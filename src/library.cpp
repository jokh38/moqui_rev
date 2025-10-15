#include "moqui_rev/library.hpp"
#include <iostream>

namespace moqui_rev {

void Library::hello() {
    std::cout << "Hello from Library class!" << std::endl;
}

int Library::add(int a, int b) {
    return a + b;
}

} // namespace moqui_rev
