#pragma once

#include "utils.cuh"
#include "sha256.cuh"

#include <cstdint>
#include <string>

struct GPUDataCell {
    static constexpr int DATA_LENGTH = 1 + 1 + 4 + 4 + 32;
    unsigned char data[64] = {0};
    friend class GPUInitStateCell;
 public:
    __device__ GPUDataCell() {}
    __device__ void update_wallet_id(uint32_t id) {
        data[6] = ((255 << 24) & id) >> 24;
        data[7] = ((255 << 16) & id) >> 16;
        data[8] = ((255 << 8) & id) >> 8;
        data[9] = (255 & id);
    }
    __device__ void prepare(const unsigned char* publicKey) {
        data[0] = 0;
        data[1] = 80;
        data[2] = data[3] = data[4] = data[5] = 0;
        cuda_memcpy<32>(data + 10, publicKey);
        data[42] = 0x80;
        memset(data + 43, 0, 19);
        data[63] = 336;
        data[62] = 336 >> 8;
    }
    __device__ void sha256_update(SHA256_STATE* ctx, unsigned char* hash) {
        sha256_init_state(ctx);
        sha256_transform_state(ctx, data);
        reverse_bytes_state(ctx, hash);
    }
};

struct GPUInitStateCell {
    static constexpr int DATA_LENGTH = 1 + 1 + 1 + 2 * 2 + 32 * 2;
    unsigned char data[128] = {0};
    unsigned char address_bytes[2 + 32 + 2];
    char address[48];
    GPUDataCell data_cell;
    SHA256_STATE ctx;
 public:
    __device__ GPUInitStateCell(const unsigned char* publicKey, const unsigned char* wallet_init_code, uint32_t id) {
        prepare(publicKey, wallet_init_code);
        update_wallet_id(id);
    }
    __device__ void prepare(const unsigned char* publicKey, const unsigned char* wallet_init_code) {
        data_cell.prepare(publicKey);
        data[0] = 2;
        data[1] = 1;
        data[2] = 0b00110100;
        data[3] = data[4] = data[5] = data[6] = 0;
        cuda_memcpy<32>(data + 7, wallet_init_code);
        data[64 + 7] = 0x80;
        memset(data + 64 + 8, 0, 54);

        data[64 + 63] = 568;
        data[64 + 62] = 568 >> 8;

        address_bytes[0] = (unsigned char)(0x51 - 0 * 0x40);
        address_bytes[1] = 0;
    }
    __device__ void sha256_update() {
        data_cell.sha256_update(&ctx, data + 39);

        sha256_init_state(&ctx);
        sha256_transform_state(&ctx, data);
        sha256_transform_state(&ctx, data + 64);

        reverse_bytes_state(&ctx, address_bytes + 2);
    }
    __device__  void update_wallet_id(uint32_t id) {
        data_cell.update_wallet_id(id);
        sha256_update();
        uint32_t crc = crc16(address_bytes);
		address_bytes[34] = (unsigned char)(crc >> 8);
		address_bytes[35] = (unsigned char)(crc & 0xff);
		buff_base64_encode(address, address_bytes);
    }
};

std::string cpu_get_address(const unsigned char* publicKey, const unsigned char* wallet_init_code, uint32_t id) {
    SHA256_STATE ctx;
    unsigned char data_cell[64] = {0};
    unsigned char data[128] = {0};
    unsigned char address_bytes[2 + 32 + 2];
    char address[48];
    data_cell[0] = 0;
    data_cell[1] = 80;
    data_cell[2] = data_cell[3] = data_cell[4] = data_cell[5] = 0;
    data_cell[6] = ((255 << 24) & id) >> 24;
    data_cell[7] = ((255 << 16) & id) >> 16;
    data_cell[8] = ((255 << 8) & id) >> 8;
    data_cell[9] = (255 & id);
    memcpy(data_cell + 10, publicKey, 32);
    data_cell[42] = 0x80;
    memset(data_cell + 43, 0, 19);
    data_cell[63] = 336;
    data_cell[62] = 336 >> 8;

    data[0] = 2;
    data[1] = 1;
    data[2] = 0b00110100;
    data[3] = data[4] = data[5] = data[6] = 0;
    memcpy(data + 7, wallet_init_code, 32);
    data[64 + 7] = 0x80;
    memset(data + 64 + 8, 0, 54);

    data[64 + 63] = 568;
    data[64 + 62] = 568 >> 8;

    address_bytes[0] = (unsigned char)(0x51 - 0 * 0x40);
    address_bytes[1] = 0;

    sha256_init_state(&ctx);
    cpu_sha256_transform_state(&ctx, data_cell);
    reverse_bytes_state(&ctx, data + 39);

    sha256_init_state(&ctx);
    cpu_sha256_transform_state(&ctx, data);
    cpu_sha256_transform_state(&ctx, data + 64);

    reverse_bytes_state(&ctx, address_bytes + 2);

    uint32_t crc = cpu_crc16(address_bytes);
	address_bytes[34] = (unsigned char)(crc >> 8);
	address_bytes[35] = (unsigned char)(crc & 0xff);
	cpu_buff_base64_encode(address, address_bytes);
    return std::string(address, 48);
}
