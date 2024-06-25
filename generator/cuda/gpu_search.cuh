#pragma once

#include "kernel.cuh"
#include "../utils/AddressChecker.h"
#include "tonlib/keys/Mnemonic.h"
#include "smc-envelope/WalletV3.h"
#include "../utils/UIManager.h"
#include "../utils/BufferedChannel.h"

#include <iostream>
#include <chrono>
#include <cuda.h>
#include <vector>
#include <thread>

void benchmark(uint64_t& best_blocks, uint64_t& best_threads, int64_t T_BLOCKS, int64_t T_THREADS) {
    if(T_BLOCKS >= 0 && T_THREADS >= 0) {
        best_blocks = (uint64_t)T_BLOCKS, best_threads = (uint64_t)T_THREADS;
        return;
    }
    auto global_start = std::chrono::high_resolution_clock::now();
    const uint64_t address_batch = 1ull << 28;
    AddressChecker checker;
    checker.add_variant({}, {"1", "2", "3", "4", "5"});

    std::vector<uint64_t> threads_variants = {128, 256, 512, 1024};
    

    if(T_THREADS >= 0) threads_variants = {(uint64_t)T_THREADS};
    std::cout << "Running benchmark...\n\n";

    int64_t best_time = -1, best_time_ms;
    for(uint64_t threads : threads_variants) {
        std::vector<uint64_t> blocks_variants;
        for(uint64_t blocks = 128; blocks * threads <= address_batch && 
            blocks * threads <= 1ull << 27; blocks <<= 1) blocks_variants.push_back(blocks);
        if(T_BLOCKS >= 0) blocks_variants = {(uint64_t)T_BLOCKS};

        for(uint64_t blocks : blocks_variants) {
            unsigned char* gpu_public_key;
            unsigned char* gpu_wallet_init_code;
            cudaMalloc(&gpu_public_key, 32 * sizeof(unsigned char));
            cudaMalloc(&gpu_wallet_init_code, 32 * sizeof(unsigned char));
            uint64_t N = blocks * threads;
            uint64_t* result;
            cudaMalloc(&result, N * sizeof(uint64_t));
            auto start = std::chrono::high_resolution_clock::now();
            cudaError_t launch_result = launch_kernel(checker, result, address_batch, blocks, threads, gpu_public_key, gpu_wallet_init_code);
            cudaDeviceSynchronize();
            auto stop = std::chrono::high_resolution_clock::now();
            cudaDeviceReset();
            if(launch_result != cudaSuccess) {
                std::cout << "Run with BLOCKS: " << blocks << " THREADS: " << threads << " failed:\n";
                printf("Reason: cudaError %d (%s)\n", launch_result, cudaGetErrorString(launch_result));
                continue;
            }
            uint64_t precise_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            uint64_t cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            std::cout << "Run with BLOCKS: " << blocks << " THREADS: " << threads << " time: " << cur_time << " ms\n";
            if(best_time == -1 || precise_time < best_time) {
                best_time = precise_time;
                best_time_ms = cur_time;
                best_blocks = blocks, best_threads = threads;
            }
        }
    }
    auto global_stop = std::chrono::high_resolution_clock::now();
    std::cout << "Benchmark finished in: " << 
    std::chrono::duration_cast<std::chrono::seconds>(global_stop - global_start).count() <<
    " s\n";
    std::cout << "Best run:\n" << "BLOCKS: " << best_blocks << "\nTHREADS: " << best_threads << "\ntime: " << best_time_ms << " ms\n";

}

class MnemonicData {
 public:
    unsigned char publicKey[32];
    std::vector<std::string> words;
    MnemonicData(const tonlib::Mnemonic& mnemonic) {
        std::vector<td::SecureString> mnemonic_words = mnemonic.get_words();
        for(int i = 0; i < 24; ++i) {
            words.emplace_back(mnemonic_words[i].data(), mnemonic_words[i].size());
        }
        td::SecureString mnemonicPublicKey = mnemonic.to_private_key().get_public_key().move_as_ok().as_octet_string();
        td::Slice tmp_publicKey = mnemonicPublicKey.as_slice();
        memcpy(publicKey, tmp_publicKey.ubegin(), 32 * sizeof(unsigned char));
    }
};

void gpu_find_address(const AddressChecker& cpu_checker, int64_t T_BLOCKS, int64_t T_THREADS) {
    uint64_t BLOCKS, THREADS;
    benchmark(BLOCKS, THREADS, T_BLOCKS, T_THREADS);

    const int cores = std::thread::hardware_concurrency();
    BufferedChannel<MnemonicData> mnemonic_queue(32);
    std::vector<std::thread> threads;
    for(uint64_t core = 0; core < cores; ++core) {
        threads.emplace_back([&]{
            while(true) {
                tonlib::Mnemonic mnemonic = tonlib::Mnemonic::create_new().move_as_ok();
                try {
                    mnemonic_queue.Send(MnemonicData(mnemonic));
                } catch(std::runtime_error) {
                    break;
                }
            }
        });
    }

    UIManager::progress_info();
    uint64_t N = BLOCKS * THREADS, address_batch = 1ull << 32;

    std::string result_address;
    uint32_t result_id;
    uint64_t total_tries = 0, total_time = 0;
    bool found = false;
    std::vector<std::string> result_mnemonic_words;

    unsigned char* gpu_public_key;
    unsigned char* gpu_wallet_init_code;
    cudaMalloc(&gpu_public_key, 32 * sizeof(unsigned char));
    cudaMalloc(&gpu_wallet_init_code, 32 * sizeof(unsigned char));
    const unsigned char* wallet_init_code = ton::WalletV3::get_init_code(2)->get_hash().as_array().data();
    cudaMemcpy(gpu_wallet_init_code, wallet_init_code, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    uint64_t* result;
    std::vector<uint64_t> cpu_result(N);
    cudaMalloc(&result, N * sizeof(uint64_t));

    auto global_start = std::chrono::high_resolution_clock::now();
    UIManager::print_columns();
    UIManager::display_progress(total_time, cpu_checker, total_tries);
    while(!found) {
        MnemonicData mnemonic_data = mnemonic_queue.Recv().value();
        cudaMemcpy(gpu_public_key, mnemonic_data.publicKey, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        run_batch_kernels(cpu_checker, result, address_batch, BLOCKS, THREADS, 
            gpu_public_key, gpu_wallet_init_code, found, total_tries, cpu_result,
            result_id, result_address, mnemonic_data.publicKey, wallet_init_code, global_start);
        if(found) {
            mnemonic_queue.Close();
            result_mnemonic_words = mnemonic_data.words;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - global_start).count();
    }
    for (auto& t : threads) {
        t.join();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - global_start);
    long double duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - global_start).count() * 1e-9L;
    std::cout << std::endl << std::endl << "Searching took: " << duration.count() << " seconds\n";
    std::cout << "Speed " << UIManager::format_speed(total_tries / (duration_nano + (duration_nano == 0))) << "\n\n";
    UIManager::display_results(result_mnemonic_words, result_id, result_address);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}