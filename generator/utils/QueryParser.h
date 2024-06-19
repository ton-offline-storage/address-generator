#pragma once

#include "AddressChecker.h"

#include <string>

class QueryParser {
    static std::bitset<256> get_char_set(bool pos3 = false);
    static std::bitset<256> get_pos3_char_set();
    const std::bitset<256> is_address_char, is_pos3_char;
    enum PatternType {
        START, END
    };
    std::vector<std::string> split(const std::string& query, char delim);

    std::vector<std::string> parse_pattern(const std::string& pattern, PatternType& type);
 public:
    QueryParser();
    bool parse_query(std::string& query, AddressChecker& address_checker);
};
