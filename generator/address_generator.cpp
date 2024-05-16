#include "tonlib/keys/Mnemonic.h"
#include "InitStateCell.h"
#include "QueryParser.h"

#include <thread>
#include <limits>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>

void display_results(const std::vector<td::SecureString>& words, std::uint32_t wallet_id, const std::string& address) {
    std::cout << "Your mnemonic phrase:\n\n";
    for(int row = 0; row < 6; ++row) {
        for(int col = 0; col < 4; ++col) {
            std::cout << std::string(words[row * 4 + col].data()).substr(0, words[row * 4 + col].size()) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout << "Your wallet id:\n\n";
    std::cout << wallet_id << "\n\n";
    std::cout << "Your address:\n\n";
    std::cout << address  << "\n\n";
    std::cout << "YOU MUST WRITE DOWN MNEMONIC PHRASE AND WALLET ID"  << '\n';
    std::cout << "You can copy them to wallet now\n";
    std::cout << "Type \"quit\" and press enter to quit. ";
    std::cout << "SAVE WORDS AND ID BEFORE QUITTING" << '\n';
    while(true) {
        std::string command;
        std::cin >> command;
        if(command == "quit") {
            break;
        }
    }
}

void find_address(int cores, const AddressChecker& address_checker) {
    std::vector<std::string> result_address(cores);
    std::vector<std::uint32_t> result_id(cores);
    std::vector<bool> has_found(cores, false);
    std::atomic_bool found{false};
    std::atomic_uint64_t total_tries = 0;
    std::vector<td::SecureString> mnemonic_words;
    std::uint64_t total_time = 0;
    auto global_start = std::chrono::high_resolution_clock::now();
    const std::uint32_t SYNC_RATE = (1 << 14) - 1;
    const std::uint32_t PROGRESS_UPDATE_RATE = (1 << 19) - 1;
    const std::uint32_t max_wallet_id = std::numeric_limits<std::uint32_t>::max();
    std::cout << "\nNote, that you may need more, or fewer tries than the excepted number.\n"
    "Address may be found before, or after reaching 100% progress\n"
    "Feel free to abort the search - mathematics works so that you won't lose progress,\n"
    "however estimated time will always be same in the beginning\n";
    while(!found) {
        std::vector<std::thread> threads;
        tonlib::Mnemonic mnemonic = tonlib::Mnemonic::create_new().move_as_ok();
        mnemonic_words = mnemonic.get_words();
        td::SecureString publicKey = mnemonic.to_private_key().get_public_key().move_as_ok().as_octet_string();
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t core = 0; core < cores; ++core) {
            threads.emplace_back([total_time, start, core, cores, &found, &publicKey, &result_address, 
                                 &result_id, &has_found, address_checker, &total_tries] {
                const long double hour_inv = 1.0 / 3600, minute_inv = 1.0 / 60;
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
                        std::stringstream stream;
                        auto stop = std::chrono::high_resolution_clock::now();
                        std::int64_t nano_seconds = total_time + std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                        long double progress = address_checker.progress(total_tries);
                        long double remain_time = nano_seconds * (100 / progress - 1);
                        bool negative = false;
                        if(remain_time < -1e-7) {
                            negative = true;
                            remain_time = -remain_time;
                        }
                        std::int64_t hours = std::floor(remain_time * (long double)1e-9 * hour_inv);
                        std::int64_t minutes = std::floor(remain_time * (long double)1e-9 * minute_inv);
                        minutes %= 60;
                        remain_time -= (hours * 3600 + minutes * 60) * 1000000000;

                        stream << std::fixed << std::setprecision(4) << "Expected progress: "
                        << progress << "%  Time remaining: " << (negative ? "-" : "") << hours << "H:" << minutes << "M:" <<
                        std::fixed << std::setprecision(2) << remain_time * 1e-9  <<
                         "S  Addresses tried: " << total_tries << "\r";
                        std::cout << stream.str();
                        std::cout.flush();
                    }
                }
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
            display_results(mnemonic_words, result_id[i], result_address[i]);
            break;
        }
    }
}

AddressChecker get_address_checker() {
    QueryParser parser;
    AddressChecker result;
    std::string query;
    bool parsed = false;
    while(!parsed) {
        std::cout << "Enter your query:\n";
        std::getline(std::cin, query);
        result.clear();
        parsed = parser.parse_query(query, result);
    }
    return result;
}
#include "common/util.h"

int main() {
    const int cores = std::thread::hardware_concurrency();
    std::cout << "\n\n\nThis utility allows you to find TON address matching specific constraints.\n"
     "You can specify a number of consecutive symbols at the end of address.\n"
     "For every symbol constraint looks like [...], and inside brackets are\n"
     "the characters you allow symbol at this position to be, or * if you allow any character.\n"
     "Remember, that TON address consists only of characters from A-Z a-z 0-9 and _ and -\n"
     "To constrain end of the address, use \"end\" command with list of constraints for symbols,\n\n"
     "like this: end[T][O][N], or like this: end[Tt][Oo][Nn]\n\n"
     "You can also specify a number of symbols at the start of address, but:\n"
     "TON address always starts with 2 characters \"UQ\", and you can\'t change that.\n"
     "Third symbol, after \"UQ\", can only be one of A, B, C, D.\n"
     "Use \"start\" command similarly to \"end\" command,\n"
     "but remember, the first symbol in your command is third in the address.\n"
     "With that in mind, you can specify start of the address like this:\n\n"
     "start[A][P][P][L][E], or like this: start[*][T][O][N]\n\n"
     "You can also specify both start and end at the same time, like this:\n\n"
     "start[*][T][O][N] & end[T][O][N]\n\n"
     "You can also add several variants of constraints, such that any\n"
     "of this constraints, if matched, satisfies you, like this:\n\n"
     "start[*][T][O][N] & end[T][O][N] | start[D][D][D] | end[0][0][0]\n\n";
    AddressChecker address_checker = get_address_checker();
    find_address(cores, address_checker);
    return 0;
}