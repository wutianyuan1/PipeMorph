#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include "event_handler.h"

int main() {
    auto eh = EventHandler("localhost", 6379);
    while (1) {
        std::cout << "sleep_time:" << eh.get_sleep_time() << std::endl;
        auto ret = eh.get_slow_links();
        for (const auto& [a, b] : ret)
            printf("(%d %d), ", a, b);
        printf("\n");
        sleep(1);
    }
    return 0;
}