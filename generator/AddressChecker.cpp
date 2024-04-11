#include "AddressChecker.h"

#include <vector>
#include <string>

VariantChecker::VariantChecker(const std::vector<std::string>& pref_pattern, 
                const std::vector<std::string>& suff_pattern) {
    for(const std::string& char_list : pref_pattern) {
        std::bitset<256> mask;
        if(char_list.front() == '*') {
            mask.set();
        } else {
            prob *= symbol_prob;
            for(unsigned char c : char_list) {
                mask[c] = 1;
            }
            if(mask.count() > 1) prob *= mask.count();
        }
        prefix.push_back(mask);
    }
    if(!pref_pattern.empty() && pref_pattern.front().front() != '*') {
        prob *= 16;
    }
    for(const std::string& char_list : suff_pattern) {
        std::bitset<256> mask;
        if(char_list.front() == '*') {
            mask.set();
        } else {
            prob *= symbol_prob;
            for(unsigned char c : char_list) {
                mask[c] = 1;
            }
            if(mask.count() > 1) prob *= mask.count();
        }
        suffix.push_back(mask);
    }
}

bool VariantChecker::check(char* address) const {
    bool match = true;
    for(int i = 0; i < prefix.size() && match; ++i) {
        match &= prefix[i][address[2 + i]];
    }
    for(int i = 0; i < suffix.size() && match; ++i){
        match &= suffix[i][address[48 - suffix.size() + i]];
    }
    return match;
}

void AddressChecker::add_variant(const std::vector<std::string>& pref_pattern, 
                    const std::vector<std::string>& suff_pattern) {
    variants.emplace_back(pref_pattern, suff_pattern);
    prob += variants.back().prob;
}

void AddressChecker::clear() {
    variants.clear();
    prob = 0;
}

bool AddressChecker::check_address(char* address) const {
    bool match = false;
    for(const VariantChecker& variant : variants) {
        match |= variant.check(address);
    }
    return match;
}

long double AddressChecker::progress(std::uint64_t tries) const {
    return (tries * 100) * prob;
}