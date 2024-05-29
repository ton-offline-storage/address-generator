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

const long double hour_inv = 1.0L / 3600, minute_inv = 1.0L / 60;

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

void display_progress(std::uint32_t col1_length, std::uint32_t col2_length, std::uint32_t col3_length,
                      std::int64_t nano_seconds, const AddressChecker& address_checker,
                      const std::atomic_uint64_t& total_tries, bool carriage_return = true) {
    std::stringstream stream, col1, col2, col3;
    long double progress = address_checker.progress(total_tries);
    long double remain_time = nano_seconds * (100L / progress - 1);
    bool negative = false;
    if(remain_time < -1e-7L) {
        negative = true;
        remain_time = -remain_time;
    }               
    col1 << std::fixed << std::setprecision(4) << progress << "% ";
    if(remain_time > 1e9L * 24.0 * 365.0 * 10.0 * 3600.0) {
        col2 << "> 10 years ";
    } else {
        std::int64_t hours = std::floor(remain_time * 1e-9L * hour_inv);
        std::int64_t minutes = std::floor(remain_time * 1e-9L * minute_inv);
        minutes %= 60;
        remain_time -= (hours * 3600 + minutes * 60) * 1000000000;
        col2 << (negative ? "-" : "") << hours << "H:" << minutes << "M:" <<
        std::fixed << std::setprecision(2) << remain_time * 1e-9L  << "S ";
    }
    long double speed = ((long double)total_tries) / ((long double)nano_seconds * 1e-9L);
    if(speed < 1e6L) {
        col3 << std::fixed << std::setprecision(2) << speed * 1e-3L << " thousand/s ";
    } else if(speed < 1e9L) {
        col3 << std::fixed << std::setprecision(2) << speed * 1e-6L << " million/s ";
    } else {
        col3 << std::fixed << std::setprecision(2) << speed * 1e-9L << " billion/s ";
    }
    stream << std::string(col1_length - col1.str().size(), ' ') << col1.str()
    << std::string(col2_length - col2.str().size(), ' ') << col2.str()
    << std::string(col3_length - col3.str().size(), ' ') << col3.str()
    << (carriage_return ? "\r" : "\n") << std::flush;
    std::cout << stream.str();
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
    const std::uint32_t SYNC_RATE = (1 << 15) - 1;
    const std::uint32_t PROGRESS_UPDATE_RATE = (1 << 21) - 1;
    const std::uint32_t max_wallet_id = std::numeric_limits<std::uint32_t>::max();
    std::cout << "\nNote, that you may need more, or fewer tries than the expected number.\n"
    "Address may be found before, or after reaching 100% progress\n"
    "Remaining time may show strange results at the beginning, time may increase\n"
    "Feel free to abort the search - mathematics works so that you won't lose progress,\n"
    "however estimated time will always be same in the beginning\n";
    std::string col1_name = "Expected progress:", col2_name = "    Time remaining:", col3_name = "    Speed addresses/s:";
    std::uint32_t col1_length = col1_name.size(), col2_length = col2_name.size(), col3_length = col3_name.size();
    while(!found) {
        std::vector<std::thread> threads;
        tonlib::Mnemonic mnemonic = tonlib::Mnemonic::create_new().move_as_ok();
        mnemonic_words = mnemonic.get_words();
        std::cout << col1_name << col2_name << col3_name << '\n';
        td::SecureString publicKey = mnemonic.to_private_key().get_public_key().move_as_ok().as_octet_string();
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t core = 0; core < cores; ++core) {
            threads.emplace_back([total_time, start, core, cores, &found, &publicKey, &result_address, 
                                 &result_id, &has_found, address_checker, &total_tries,
                                 col1_length, col2_length, col3_length] {
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
                        display_progress(col1_length, col2_length, col3_length,
                        total_time + std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count(),
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
        display_progress(col1_length, col2_length, col3_length,
        total_time, address_checker, total_tries, false);
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