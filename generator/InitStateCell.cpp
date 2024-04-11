#include "InitStateCell.h"
#include "common/util.h"
#include "smc-envelope/WalletV3.h"

#include <openssl/sha.h>

DataCell::DataCell(td::Slice publicKey) {
    data[0] = 0;
    data[1] = 80;
    data[2] = data[3] = data[4] = data[5] = 0;
    data[6] = data[7] = data[8] = data[9] = 0;
    memcpy(data + 10, publicKey.ubegin(), publicKey.size());
}
void DataCell::write_wallet_id(std::uint32_t id) {
    data[6] = ((255 << 24) & id) >> 24;
    data[7] = ((255 << 16) & id) >> 16;
    data[8] = ((255 << 8) & id) >> 8;
    data[9] = (255 & id);
}


InitStateCell::InitStateCell(td::Slice publicKey): data_cell(publicKey) {
    data[0] = 2;
    data[1] = 1;
    data[2] = 0b00110100;
    data[3] = data[4] = data[5] = data[6] = 0;
    std::array<td::uint8, vm::CellTraits::hash_bytes> hash_bytes = ton::WalletV3::get_init_code(2)->get_hash().as_array();
    memcpy(data + 7, hash_bytes.data(), hash_bytes.size());

    address_bytes[0] = (unsigned char)(0x51 - 0 * 0x40);
    address_bytes[1] = 0;
}

void InitStateCell::update_wallet_id(std::uint32_t id) {
    data_cell.write_wallet_id(id);
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data_cell.data, DataCell::DATA_LENGTH);
    SHA256_Final(data + 39, &sha256);
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, DATA_LENGTH);
    SHA256_Final(address_bytes + 2, &sha256);
    unsigned int crc = td::crc16(td::Slice{address_bytes, 34});
    address_bytes[34] = (unsigned char)(crc >> 8);
    address_bytes[35] = (unsigned char)(crc & 0xff);
}

char* InitStateCell::getAddress() {
    td::buff_base64_encode(td::MutableSlice{address, 48}, td::Slice{address_bytes, 36}, true);
    return address;
}