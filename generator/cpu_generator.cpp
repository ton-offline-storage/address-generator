#include "tonlib/keys/Mnemonic.h"
#include "utils/InitStateCell.h"
#include "utils/UIManager.h"

#include <thread>
#include <limits>
#include <iostream>
#include <chrono>
#include <iomanip>


void find_address(const AddressChecker& address_checker) {
    const int cores = std::thread::hardware_concurrency();
    std::vector<std::string> result_address(cores);
    std::vector<std::uint32_t> result_id(cores);
    std::vector<bool> has_found(cores, false);
    std::atomic_bool found{false};
    std::atomic_uint64_t total_tries = 0;
    std::vector<td::SecureString> mnemonic_words;
    std::uint64_t total_time = 0;
    auto global_start = std::chrono::high_resolution_clock::now();
    const std::uint32_t SYNC_RATE = (1 << 15) - 1;
    const std::uint32_t PROGRESS_UPDATE_RATE = (1 << 21) - 1;
    const std::uint32_t max_wallet_id = std::numeric_limits<std::uint32_t>::max();
    UIManager::progress_info();
    while(!found) {
        std::vector<std::thread> threads;
        tonlib::Mnemonic mnemonic = tonlib::Mnemonic::create_new().move_as_ok();
        mnemonic_words = mnemonic.get_words();
        UIManager::print_columns();
        td::SecureString publicKey = mnemonic.to_private_key().get_public_key().move_as_ok().as_octet_string();
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t core = 0; core < cores; ++core) {
            threads.emplace_back([=,&found, &publicKey, &result_address, 
                                 &result_id, &has_found, &total_tries] {
                InitStateCell init_state = InitStateCell(publicKey);
                std::uint64_t local_tries = 0;
                for(std::uint64_t i = core; i <= max_wallet_id && !found; i += cores, ++local_tries) {
                    init_state.update_wallet_id(i);
                    char* addr = init_state.getAddress();
                    if(address_checker.check_address(addr)) {
                        has_found[core] = true;
                        result_id[core] = i;
                        result_address[core] = std::string(addr);
                        found = true;
                    }
                    if((local_tries & SYNC_RATE) == 0) {
                        total_tries += local_tries;
                        local_tries = 0;
                    }
                    if(((i + 1) & PROGRESS_UPDATE_RATE) == 0) {
                        auto stop = std::chrono::high_resolution_clock::now();
                        UIManager::display_progress(total_time + std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count(),
                        address_checker, total_tries);
                    }
                }
                total_tries += local_tries;
                local_tries = 0;
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        auto stop = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - global_start);
    std::cout << std::endl << std::endl << "Searching took: " << duration.count() << " seconds\n";
    for(int i = 0; i < cores; ++i) {
        if(has_found[i]) {
            std::cout << "Speed " << total_tries / (duration.count() + (duration.count() == 0)) << " addr/sec\n\n";
            UIManager::display_results(mnemonic_words, result_id[i], result_address[i]);
            break;
        }
    }
}

int main() {
    UIManager::start_info();
    AddressChecker address_checker = UIManager::get_address_checker();
    find_address(address_checker);
    return 0;
}