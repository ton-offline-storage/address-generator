#include "UIManager.h"
#include "QueryParser.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

void UIManager::display_results(const std::vector<td::SecureString>& mnemonic_words, std::uint32_t wallet_id, const std::string& address) {
    std::vector<std::string> words;
    for(int i = 0; i < 24; ++i) {
        words.emplace_back(mnemonic_words[i].data(), mnemonic_words[i].size());
    }
    display_results(words, wallet_id, address);
}

void UIManager::display_results(const std::vector<std::string>& words, std::uint32_t wallet_id, const std::string& address) {
    std::cout << "Your mnemonic phrase:\n\n";
    for(int row = 0; row < 6; ++row) {
        for(int col = 0; col < 4; ++col) {
            std::cout << words[row * 4 + col] << ' ';
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

std::string UIManager::format_speed(long double speed) {
    std::stringstream ss;
    if(speed < 1e6L) {
        ss << std::fixed << std::setprecision(2) << speed * 1e-3L << " thousand/s";
    } else if(speed < 1e9L) {
        ss << std::fixed << std::setprecision(2) << speed * 1e-6L << " million/s";
    } else {
        ss << std::fixed << std::setprecision(2) << speed * 1e-9L << " billion/s";
    }
    return ss.str();
}

void UIManager::display_progress(std::int64_t nano_seconds, const AddressChecker& address_checker,
                      const uint64_t& total_tries, bool carriage_return) {
    std::stringstream stream, col1, col2, col3;
    long double progress = address_checker.progress(total_tries);
    col1 << std::fixed << std::setprecision(4) << progress << "% ";
    if(total_tries == 0) {
        col2 << "?? ";
    } else {
        long double remain_time = nano_seconds * (100L / progress - 1);
        bool negative = false;
        if(remain_time < -1e-7L) {
            negative = true;
            remain_time = -remain_time;
        }     
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
    }
    if(total_tries == 0) {
        col3 << "?? ";
    } else {
        long double speed = ((long double)total_tries) / ((long double)nano_seconds * 1e-9L);
        col3 << format_speed(speed) << " ";
    }
    stream << std::string(col1_length - col1.str().size(), ' ') << col1.str()
    << std::string(col2_length - col2.str().size(), ' ') << col2.str()
    << std::string(col3_length - col3.str().size(), ' ') << col3.str()
    << (carriage_return ? "\r" : "\n");
    std::cout << stream.str() << std::flush;
}

AddressChecker UIManager::get_address_checker() {
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

void UIManager::progress_info() {
    std::cout << "\nNote, that you may need more, or fewer tries than the expected number.\n"
    "Address may be found before, or after reaching 100% progress\n"
    "Remaining time may show strange results at the beginning, time may increase\n"
    "Feel free to abort the search - mathematics works so that you won't lose progress,\n"
    "however estimated time will always be same in the beginning\n\n\n";
}

void UIManager::print_columns() {
    std::cout << col1_name << col2_name << col3_name << '\n';
}

void UIManager::start_info() {
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
}


