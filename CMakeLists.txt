cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Address-generator)

set(ADDRESS_GENERATOR_SOURCE
    generator/InitStateCell.h
    generator/AddressChecker.h
    generator/QueryParser.h

    generator/InitStateCell.cpp
    generator/AddressChecker.cpp
    generator/QueryParser.cpp
    generator/address_generator.cpp
)

add_executable(generator ${ADDRESS_GENERATOR_SOURCE})

add_subdirectory(ton)

target_link_libraries(generator Tonlib)
target_link_libraries(generator ton_crypto)