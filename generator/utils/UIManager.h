#pragma once

#include "AddressChecker.h"
#include "td/utils/SharedSlice.h"

#include <string>
#include <string_view>

class UIManager {
    static constexpr long double hour_inv = 1.0L / 3600, minute_inv = 1.0L / 60;
    static constexpr std::string_view col1_name = "Expected progress:", 
    col2_name = "    Time remaining:", 
    col3_name = "    Speed addresses/s:";
    static constexpr uint32_t col1_length = col1_name.size(), 
    col2_length = col2_name.size(), col3_length = col3_name.size();
 public:
    static void display_results(const std::vector<td::SecureString>& words, std::uint32_t wallet_id, const std::string& address);
    static void display_results(const std::vector<std::string>& words, std::uint32_t wallet_id, const std::string& address);
    static std::string format_speed(long double speed);
    static void display_progress(std::int64_t nano_seconds, const AddressChecker& address_checker,
                      const uint64_t& total_tries, bool carriage_return = true);
    static AddressChecker get_address_checker();
    static bool init_address_checker(std::string& query, AddressChecker& checker);
    static void progress_info();
    static void print_columns();
    static void start_info();
    static void quit(bool save_words = true);
};