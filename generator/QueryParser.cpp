#include "QueryParser.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

std::bitset<256> QueryParser::get_char_set(bool pos3) {
    std::bitset<256> is_address_char;
    if(pos3) {
        for(unsigned c : "ABCD") {
            is_address_char[c] = 1;
        }
    } else {
        for(unsigned c : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_") {
            is_address_char[c] = 1;
        }
    }
    return is_address_char;
}

std::bitset<256> QueryParser::get_pos3_char_set() {
    std::bitset<256> is_address_char;
    for(unsigned c : "ABCD") {
        is_address_char[c] = 1;
    }
    return is_address_char;
}

std::vector<std::string> QueryParser::split(const std::string& query, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(query);
    std::string item;
    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> QueryParser::parse_pattern(const std::string& pattern, PatternType& type) {
    std::vector<std::string> char_lists = split(pattern, '[');
    if(char_lists.size() < 2) {
        throw std::invalid_argument("Empty pattern provided");
    }
    if(char_lists.front() == "start") {
        type = START;
    } else if(char_lists.front() == "end") {
        type = END;
    } else {
        throw std::invalid_argument("Wrong pattern type");
    }
    char_lists = std::vector<std::string>(char_lists.begin() + 1, char_lists.end());
    for(std::string& list : char_lists) {
        if(list.size() < 2 || list.back() != ']') {
            throw std::invalid_argument("Wrong character list in pattern");
        }
        list.pop_back();
        if(list.size() == 1 && list.front() == '*') {
            continue;
        }
        for(unsigned char c : list) {
            if(!is_address_char[c]) {
                std::string message = "This character cannot occur in TON address: ";
                message += c;
                throw std::invalid_argument(message);
            }
        }
    }
    if(type == START) {
        if(char_lists.size() > 48 - 2) {
            throw std::invalid_argument("Start pattern is too long");
        }
        for(char c : char_lists.front()) {
            if(!is_pos3_char[c] && c != '*') {
                throw std::invalid_argument("Position 3 of the address(first position of start pattern) can only be one of symbols A,B,C,D");
            }
        }
    }
    if(type == END) {
        if(char_lists.size() > 48 - 2) {
            throw std::invalid_argument("End pattern is too long");
        }
    }
    return char_lists;
}

QueryParser::QueryParser() : is_address_char(get_char_set()), is_pos3_char(get_char_set(true)) {}
    
bool QueryParser::parse_query(std::string& query, AddressChecker& address_checker) {
    query.erase(std::remove_if(query.begin(), query.end(), isspace), query.end());
    std::vector<std::string> variants = split(query, '|');
    if(variants.empty()) {
        std::cout << "No address variants provided\n";
        return false;
    }
    for(const std::string& variant : variants) {
        std::vector<std::string> patterns = split(variant, '&');
        std::vector<std::string> start_pattern, end_pattern;
        if(patterns.empty()) {
            std::cout << "No patterns provided\n";
        }
        if(patterns.size() > 2) {
            std::cout << "No more than 2 patterns - start and end - is allowed\n";
            return false;
        }
        for(const std::string& pattern : patterns) {
            PatternType type;
            std::vector<std::string> current_pattern;
            try {
                current_pattern = parse_pattern(pattern, type);
            } catch(std::invalid_argument error) {
                std::cout << "Invalid pattern provided: " << error.what() << '\n';
                return false;
            }
            if(type == START) {
                start_pattern = current_pattern;
            } else {
                end_pattern = current_pattern;
            }
        }
        address_checker.add_variant(start_pattern, end_pattern);
    }
    return true;
}     
