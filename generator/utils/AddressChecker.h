#pragma once

#include <bitset>
#include <vector>
#include <string>
#include <cstdint>

class VariantChecker {
 public:
    std::vector<std::bitset<256>> prefix, suffix;
    const long double symbol_prob = 1.0 / 64; 
    long double prob = 1;
    friend class AddressChecker;
    VariantChecker(const std::vector<std::string>& pref_pattern, 
                const std::vector<std::string>& suff_pattern);
    bool check(char* address) const;
};

class AddressChecker {
 public:
    std::vector<VariantChecker> variants;
    long double prob = 0;
    void add_variant(const std::vector<std::string>& pref_pattern, 
                    const std::vector<std::string>& suff_pattern);
    void clear();
    bool check_address(char* address) const;
    long double progress(std::uint64_t tries) const;
};