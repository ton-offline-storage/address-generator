#include "utils/getopt.h"
#include "utils/UIManager.h"
#include "cuda/gpu_search.cuh"

#include <iostream>


int main(int argc, char* const argv[]) {
    int flag;
    int64_t BLOCKS = -1, THREADS = -1;
    bool only_benchmark = false;
    while ((flag = getopt(argc, argv, "bB:T:")) != -1) {
        switch(flag) {
            case 'B':
                BLOCKS = atoll(optarg);
                if(BLOCKS > (1ll << 25)) {
                    std::cout << "Too many thread blocks, no more than " << (1ull << 25) << " is allowed\n";
                    return 0;
                }
                if(BLOCKS <= 0) {
                    std::cout << "Invalid number of blocks: " << optarg << "\n";
                    return 0;
                }
                break;
            case 'T':
                THREADS = atoll(optarg);
                if(THREADS > (1ll << 25)) {
                    std::cout << "Too many threads per block, no more than " << (1ull << 25) << " is allowed\n";
                    return 0;
                }
                if(THREADS <= 0) {
                    std::cout << "Invalid number of threads: " << optarg << "\n";
                    return 0;
                }
                break;
            case 'b':
                only_benchmark = true;
                break;
            default:
                return 0;
        }
    }
    if(only_benchmark) {
        uint64_t _b, _t;
        benchmark(_b, _t, -1, -1);
        return 0;
    }
    if(BLOCKS > 0 && THREADS > 0 && BLOCKS * THREADS > (1ll << 32)) {
        std::cout << "Multiplication of number of blocks and number of threads per block must not exceed " <<
        (1ull << 32) << "(2^32)\n";
        return 0;
    }
    UIManager::start_info();
    AddressChecker address_checker = UIManager::get_address_checker();
    gpu_find_address(address_checker, BLOCKS, THREADS);
}