#pragma once

#include "../utils/AddressChecker.h"

#include <bitset>
#include <cstdint>
#include <cuda.h>

struct CharMap {
    bool map[256] = {0};
    __device__ CharMap() {}
    CharMap(const std::bitset<256>& bs_map) {
        for(int i = 0; i < 256; ++i) map[i] = bs_map[i];
    }
    __device__ bool operator[](uint32_t i) const {
        return map[i];
    }
    void set(bool val = true) {
        for(int i = 0; i < 256; ++i) map[i] = val;
    }
};

template<uint32_t prefix_len, uint32_t suffix_len>
class FastVariantChecker {
public:
    CharMap prefix[prefix_len];
    CharMap suffix[suffix_len];
    __device__ FastVariantChecker() {}
    __device__ bool check(char* address) const {
        bool match = true;
        for(int i = 0; i < prefix_len && match; ++i) {
            match &= prefix[i][address[2 + i]];
        }
        for(int i = 0; i < suffix_len && match; ++i){
            match &= suffix[suffix_len - 1 - i][address[48 - 1 - i]];
        }
       return match;
   }
};

template<uint32_t suffix_len>
class FastVariantChecker<0, suffix_len> {
 public:
    CharMap suffix[suffix_len];
    __device__ FastVariantChecker() {}
    __device__ bool check(char* address) const {
        bool match = true;
        for(int i = 0; i < suffix_len && match; ++i){
            match &= suffix[suffix_len - 1 - i][address[48 - 1 - i]];
        }
        return match;
    }
};

template<uint32_t prefix_len>
class FastVariantChecker<prefix_len, 0> {
public:
    CharMap prefix[prefix_len];
    __device__ FastVariantChecker() {}
    __device__ bool check(char* address) const {
        bool match = true;
        for(int i = 0; i < prefix_len && match; ++i) {
            match &= prefix[i][address[2 + i]];
        }
        return match;
    }
};

template<uint32_t prefix_len, uint32_t suffix_len>
class FastAddressChecker {
public:
    FastVariantChecker<prefix_len, suffix_len>* variants;
    uint32_t num_variants;
    __device__ bool check_address(char* address) const {
        bool match = false;
        for(int i = 0; i < num_variants; ++i) {
            match |= variants[i].check(address);
        }
        return match;
   }
};

template<uint32_t prefix_len, uint32_t suffix_len>
FastAddressChecker<prefix_len, suffix_len>* allocFastChecker(const AddressChecker& cpu_checker) {
    FastAddressChecker<prefix_len, suffix_len>* fast_checker;
    uint32_t num_variants = cpu_checker.variants.size();
    cudaMalloc(&fast_checker, sizeof(FastAddressChecker<prefix_len, suffix_len>));
    cudaMemcpy(&(fast_checker->num_variants), &num_variants, sizeof(uint32_t), cudaMemcpyHostToDevice);
    FastVariantChecker<prefix_len, suffix_len>* variants;
    cudaMalloc(&variants, num_variants * sizeof(FastVariantChecker<prefix_len, suffix_len>));
    for(uint32_t var_id = 0; var_id < num_variants; ++var_id) {
        const VariantChecker& var_checker = cpu_checker.variants[var_id];
        
        if constexpr(prefix_len > 0) {
            uint32_t true_prefix_len = var_checker.prefix.size();
            std::vector<CharMap> tmp_prefix(prefix_len);
            for(uint32_t i = 0; i < true_prefix_len; ++i) tmp_prefix[i] = CharMap(var_checker.prefix[i]);
            for(uint32_t i = true_prefix_len; i < prefix_len; ++i) tmp_prefix[i].set();
            cudaMemcpy(variants[var_id].prefix, tmp_prefix.data(), prefix_len * sizeof(CharMap), cudaMemcpyHostToDevice);
        }
        if constexpr(suffix_len > 0) {
            uint32_t true_suffix_len = var_checker.suffix.size();
            std::vector<CharMap> tmp_suffix(suffix_len);
            for(uint32_t i = 0; i < true_suffix_len; ++i) tmp_suffix[suffix_len - true_suffix_len + i] = CharMap(var_checker.suffix[i]);
            for(uint32_t i = 0; i < suffix_len - true_suffix_len; ++i) tmp_suffix[i].set();
            cudaMemcpy(variants[var_id].suffix, tmp_suffix.data(), suffix_len * sizeof(CharMap), cudaMemcpyHostToDevice);
        }
    }
    cudaMemcpy(&(fast_checker->variants), &variants, sizeof(FastVariantChecker<prefix_len, suffix_len>*), cudaMemcpyHostToDevice);
    return fast_checker;
}

template<uint32_t prefix_len, uint32_t suffix_len>
void freeFastChecker(FastAddressChecker<prefix_len, suffix_len>* fast_checker) {
    FastVariantChecker<prefix_len, suffix_len>* variants;
    cudaMemcpy(&variants, &(fast_checker->variants), sizeof(FastVariantChecker<prefix_len, suffix_len>*), cudaMemcpyDeviceToHost);
    cudaFree(variants);
    cudaFree(fast_checker);
}