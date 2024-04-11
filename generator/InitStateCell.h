#pragma once

#include "td/utils/Slice.h"

#include <openssl/sha.h>

class DataCell {
    static constexpr int DATA_LENGTH = 1 + 1 + 4 + 4 + 32;
    unsigned char data[DATA_LENGTH] = {0};
    friend class InitStateCell;
 public:
    DataCell(td::Slice publicKey);
    void write_wallet_id(std::uint32_t id);
};

class InitStateCell {
    static constexpr int DATA_LENGTH = 1 + 1 + 1 + 2 * 2 + SHA256_DIGEST_LENGTH * 2;
    unsigned char data[DATA_LENGTH] = {0};
    unsigned char address_bytes[2 + SHA256_DIGEST_LENGTH + 2];
    char address[48];
    DataCell data_cell;
    SHA256_CTX sha256;
 public:
    InitStateCell(td::Slice publicKey);
    void update_wallet_id(std::uint32_t id);
    char* getAddress();
};