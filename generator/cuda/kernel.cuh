#pragma once

#include "GPUInitStateCell.cuh"
#include "FastAddressChecker.cuh"
#include "../utils/AddressChecker.h"

#include <cstdint>
#include <chrono>

template<uint32_t prefix_len, uint32_t suffix_len>
__device__ void gpu_thread_search(uint64_t thread_id, 
    FastAddressChecker<prefix_len, suffix_len>* fast_checker,
    uint64_t* result, uint64_t n, uint64_t total,
    unsigned char* gpu_public_key, unsigned char* gpu_wallet_init_code, uint64_t start_id) {
    
    result[thread_id] = 1ull << 32;
	GPUInitStateCell cell(gpu_public_key, gpu_wallet_init_code, thread_id);
	for(uint64_t i = start_id + thread_id; i < start_id + total; i += n) {
		cell.update_wallet_id(i);
		if(fast_checker->check_address(cell.address)) {
			result[thread_id] = i;
			break;
		}
	}
}

template<uint32_t prefix_len, uint32_t suffix_len>
__global__ void kernel(FastAddressChecker<prefix_len, suffix_len>* fast_checker,
    uint64_t* result, uint64_t n, uint64_t total, unsigned char* gpu_public_key, 
    unsigned char* gpu_wallet_init_code, uint64_t start_id) {
	
	uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n){
        gpu_thread_search(thread_id, fast_checker, result, n, total, gpu_public_key, gpu_wallet_init_code, start_id);	
	}
}

cudaError_t launch_kernel(const AddressChecker& cpu_checker, uint64_t* result, 
    uint64_t total, uint64_t BLOCKS, uint64_t THREADS, unsigned char* gpu_public_key, 
    unsigned char* gpu_wallet_init_code, uint64_t start_id = 0) {
    uint32_t max_prefix = 0, max_suffix = 0, N = BLOCKS * THREADS;
    cudaError_t cuda_result;
    for(const VariantChecker& var_checker : cpu_checker.variants) {
        max_prefix = max(max_prefix, (uint32_t)var_checker.prefix.size());
        max_suffix = max(max_suffix, (uint32_t)var_checker.suffix.size());
    }
    uint32_t max_template = max(max_prefix, max_suffix);
    if(max_prefix == 0) {
        FastAddressChecker<0, 48>* fast_checker = allocFastChecker<0, 48>(cpu_checker);
        kernel<0, 48><<<BLOCKS, THREADS>>>(fast_checker, result, N, total, gpu_public_key, gpu_wallet_init_code, start_id);
        cuda_result = cudaGetLastError();
        freeFastChecker(fast_checker);
    } else if(max_suffix == 0) {
        FastAddressChecker<48, 0>* fast_checker = allocFastChecker<48, 0>(cpu_checker);
        kernel<48, 0><<<BLOCKS, THREADS>>>(fast_checker, result, N, total, gpu_public_key, gpu_wallet_init_code, start_id);
        cuda_result = cudaGetLastError();
        freeFastChecker(fast_checker);
    } else if(max_template <= 7) {
        FastAddressChecker<7, 7>* fast_checker = allocFastChecker<7, 7>(cpu_checker);
        kernel<7, 7><<<BLOCKS, THREADS>>>(fast_checker, result, N, total, gpu_public_key, gpu_wallet_init_code, start_id);
        cuda_result = cudaGetLastError();
        freeFastChecker(fast_checker);
    } else if(max_template <= 12) {
        FastAddressChecker<12, 12>* fast_checker = allocFastChecker<12, 12>(cpu_checker);
        kernel<12, 12><<<BLOCKS, THREADS>>>(fast_checker, result, N, total, gpu_public_key, gpu_wallet_init_code, start_id);
        cuda_result = cudaGetLastError();
        freeFastChecker(fast_checker);
    } else {
        FastAddressChecker<48, 48>* fast_checker = allocFastChecker<48, 48>(cpu_checker);
        kernel<48, 48><<<BLOCKS, THREADS>>>(fast_checker, result, N, total, gpu_public_key, gpu_wallet_init_code, start_id);
        cuda_result = cudaGetLastError();
        freeFastChecker(fast_checker);
    }
    return cuda_result;
}

void process_batch(const AddressChecker& cpu_checker, uint64_t* result, 
    bool& found, uint64_t& total_tries, std::vector<uint64_t>& cpu_result, 
    uint32_t& result_id, std::string& result_address,
    const unsigned char* publicKey, const unsigned char* wallet_init_code,
    std::chrono::time_point<std::chrono::high_resolution_clock> global_start,
    uint64_t N, uint64_t BATCH) {
    cudaMemcpy(cpu_result.data(), result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for(uint64_t i = 0; i < N && !found; ++i) {
        if(cpu_result[i] < (1ull << 32)) {
            found = true;
            result_id = cpu_result[i];
            result_address = cpu_get_address(publicKey, wallet_init_code, result_id);
        }
    }
    total_tries += BATCH;
    auto stop = std::chrono::high_resolution_clock::now();
    UIManager::display_progress(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - global_start).count(),
    cpu_checker, total_tries);
}

void run_batch_kernels(const AddressChecker& cpu_checker, uint64_t* result, 
    uint64_t total, uint64_t BLOCKS, uint64_t THREADS, 
    unsigned char* gpu_public_key, unsigned char* gpu_wallet_init_code,
     bool& found, uint64_t& total_tries, std::vector<uint64_t>& cpu_result, 
     uint32_t& result_id, std::string& result_address,
    const unsigned char* publicKey, const unsigned char* wallet_init_code,
    std::chrono::time_point<std::chrono::high_resolution_clock> global_start) {


    const int BATCH = 1ull << 28;
    uint32_t max_prefix = 0, max_suffix = 0, N = BLOCKS * THREADS;
    for(const VariantChecker& var_checker : cpu_checker.variants) {
        max_prefix = max(max_prefix, (uint32_t)var_checker.prefix.size());
        max_suffix = max(max_suffix, (uint32_t)var_checker.suffix.size());
    }
    uint32_t max_template = max(max_prefix, max_suffix);
    if(max_prefix == 0) {
        FastAddressChecker<0, 48>* fast_checker = allocFastChecker<0, 48>(cpu_checker);
        for(uint64_t start_id = 0; start_id < total && !found; start_id += BATCH) {
            kernel<0, 48><<<BLOCKS, THREADS>>>(fast_checker, result, N, BATCH,
                 gpu_public_key, gpu_wallet_init_code, start_id);
            process_batch(cpu_checker, result, found, total_tries, cpu_result, result_id,
            result_address, publicKey, wallet_init_code, global_start, N, BATCH);
        }
        freeFastChecker(fast_checker);
    } else if(max_suffix == 0) {
        FastAddressChecker<48, 0>* fast_checker = allocFastChecker<48, 0>(cpu_checker);
        for(uint64_t start_id = 0; start_id < total && !found; start_id += BATCH) {
            kernel<48, 0><<<BLOCKS, THREADS>>>(fast_checker, result, N, BATCH,
                 gpu_public_key, gpu_wallet_init_code, start_id);
            process_batch(cpu_checker, result, found, total_tries, cpu_result, result_id,
            result_address, publicKey, wallet_init_code, global_start, N, BATCH);
        }
        freeFastChecker(fast_checker);
    } else if(max_template <= 7) {
        FastAddressChecker<7, 7>* fast_checker = allocFastChecker<7, 7>(cpu_checker);
        for(uint64_t start_id = 0; start_id < total && !found; start_id += BATCH) {
            kernel<7, 7><<<BLOCKS, THREADS>>>(fast_checker, result, N, BATCH,
                 gpu_public_key, gpu_wallet_init_code, start_id);
            process_batch(cpu_checker, result, found, total_tries, cpu_result, result_id,
            result_address, publicKey, wallet_init_code, global_start, N, BATCH);
        }
        freeFastChecker(fast_checker);
    } else if(max_template <= 12) {
        FastAddressChecker<12, 12>* fast_checker = allocFastChecker<12, 12>(cpu_checker);
        for(uint64_t start_id = 0; start_id < total && !found; start_id += BATCH) {
            kernel<12, 12><<<BLOCKS, THREADS>>>(fast_checker, result, N, BATCH,
                 gpu_public_key, gpu_wallet_init_code, start_id);
            process_batch(cpu_checker, result, found, total_tries, cpu_result, result_id,
            result_address, publicKey, wallet_init_code, global_start, N, BATCH);
        }
        freeFastChecker(fast_checker);
    } else {
        FastAddressChecker<48, 48>* fast_checker = allocFastChecker<48, 48>(cpu_checker);
        for(uint64_t start_id = 0; start_id < total && !found; start_id += BATCH) {
            kernel<48, 48><<<BLOCKS, THREADS>>>(fast_checker, result, N, BATCH,
                 gpu_public_key, gpu_wallet_init_code, start_id);
            process_batch(cpu_checker, result, found, total_tries, cpu_result, result_id,
            result_address, publicKey, wallet_init_code, global_start, N, BATCH);
        }
        freeFastChecker(fast_checker);
    }
}