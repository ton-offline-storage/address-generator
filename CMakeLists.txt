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

set(TON_VERSION 0.5)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(OPENSSL_USE_STATIC_LIBS TRUE)

# Define the two required variables before including the source code for watching a git repository.
set(PRE_CONFIGURE_FILE "ton/git.cc.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.cc")
include(ton/git_watcher.cmake)

# Create a library out of the compiled post-configure file.
add_library(git STATIC ${POST_CONFIGURE_FILE})
target_include_directories(git PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ton)
add_dependencies(git check_git)

# Prevent in-source build
get_filename_component(TON_REAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(TON_REAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)

if (TON_REAL_BINARY_DIR STREQUAL TON_REAL_SOURCE_DIR)
  message("  Out-of-source build should be used to build TON.")
  message("  You need to remove the files already created by CMake and")
  message("  rerun CMake from a new directory:")
  message("  rm -rf CMakeFiles CMakeCache.txt")
  message("  mkdir build")
  message("  cd build")
  message("  cmake ..")
  message(FATAL_ERROR "In-source build failed.")
endif()

# HAVE_SSE42 for crc32c and rocksdb
include(CheckCXXSourceCompiles)
# Check for SSE4.2 support in the compiler.
set(OLD_CMAKE_REQURED_FLAGS ${CMAKE_REQUIRED_FLAGS})
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /arch:AVX")
else(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -msse4.2")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
check_cxx_source_compiles("
#if defined(_MSC_VER)
#include <intrin.h>
#else  // !defined(_MSC_VER)
#include <cpuid.h>
#include <nmmintrin.h>
#endif  // defined(_MSC_VER)

int main() {
  _mm_crc32_u8(0, 0); _mm_crc32_u32(0, 0);
#if defined(_M_X64) || defined(__x86_64__)
   _mm_crc32_u64(0, 0);
#endif // defined(_M_X64) || defined(__x86_64__)
  return 0;
}
"  CRC32C_HAVE_SSE42)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQURED_FLAGS})

if(NOT MSVC)
  set(CMAKE_REQUIRED_FLAGS "-msse4.2 -mpclmul")
endif()
CHECK_CXX_SOURCE_COMPILES("
#include <cstdint>
#include <nmmintrin.h>
#include <wmmintrin.h>
int main() {
  volatile uint32_t x = _mm_crc32_u32(0, 0);
  const auto a = _mm_set_epi64x(0, 0);
  const auto b = _mm_set_epi64x(0, 0);
  const auto c = _mm_clmulepi64_si128(a, b, 0x00);
  auto d = _mm_cvtsi128_si64(c);
}
" ROCKSDB_HAVE_SSE42)
unset(CMAKE_REQUIRED_FLAGS)

if (ROCKSDB_HAVE_SSE42 AND CRC32C_HAVE_SSE42)
  set(HAVE_SSE42 TRUE)
else()
  set(HAVE_SSE42 FALSE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

#BEGIN internal
option(USE_EMSCRIPTEN "Use \"ON\" for config building wasm." OFF)
option(TON_ONLY_TONLIB "Use \"ON\" to build only tonlib." OFF)
set(TON_ONLY_TONLIB true)
if (USE_EMSCRIPTEN)
  set(TON_ONLY_TONLIB true)
endif()
if (TON_ONLY_TONLIB)
  set(NOT_TON_ONLY_TONLIB false)
else()
  set(NOT_TON_ONLY_TONLIB true)
endif()


option(TON_USE_ROCKSDB "Use \"ON\" to enable RocksDb." ${NOT_TON_ONLY_TONLIB})
option(TON_USE_ABSEIL "Use \"ON\" to enable Abseil." ${NOT_TON_ONLY_TONLIB})
option(TON_USE_JEMALLOC "Use \"ON\" to enable JeMalloc." OFF)
#END internal

option(TONLIB_ENABLE_JNI "Use \"ON\" to enable JNI-compatible TonLib API.")
option(TON_USE_ASAN "Use \"ON\" to enable AddressSanitizer." OFF)
option(TON_USE_TSAN "Use \"ON\" to enable ThreadSanitizer." OFF)
option(TON_USE_UBSAN "Use \"ON\" to enable UndefinedBehaviorSanitizer." OFF)
set(TON_ARCH "native" CACHE STRING "Architecture, will be passed to -march=")

#BEGIN M1 support
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

if ((ARCHITECTURE MATCHES "arm64") AND (CMAKE_SYSTEM_NAME STREQUAL "Darwin") AND
    (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)) # only clang 13+ supports cpu=apple-m1
    set(TON_ARCH "apple-m1")
endif()
#END M1 support

if (TON_USE_ABSEIL)
  message("Add abseil-cpp")
  set(ABSL_PROPAGATE_CXX_STD TRUE)
  add_subdirectory(ton/third-party/abseil-cpp EXCLUDE_FROM_ALL)
  set(ABSL_FOUND 1)
endif()

#add_subdirectory(third-party/libcuckoo EXCLUDE_FROM_ALL)
#add_subdirectory(third-party/junction EXCLUDE_FROM_ALL)

if (WIN32)
  message("Add wingetopt")
  add_subdirectory(ton/third-party/wingetopt EXCLUDE_FROM_ALL)
  set(WINGETOPT_FOUND 1)
  message(STATUS "Use wingetopt")
endif()

set(CRC32C_BUILD_TESTS OFF CACHE BOOL "Build CRC32C's unit tests")
set(CRC32C_BUILD_BENCHMARKS OFF CACHE BOOL "Build CRC32C's benchmarks")
set(CRC32C_USE_GLOG OFF CACHE BOOL "Build CRC32C's tests with Google Logging")
set(CRC32C_INSTALL OFF CACHE BOOL "Install CRC32C's header and library")
message("Add crc32c")
if (NOT MSVC)
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  # fix aarch64 build @ crc32c/src/crc32c_arm64_linux_check.h
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=address")
  add_subdirectory(ton/third-party/crc32c EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  unset(OLD_CMAKE_CXX_FLAGS)
else()
  add_subdirectory(ton/third-party/crc32c EXCLUDE_FROM_ALL)
endif()
set(CRC32C_FOUND 1)

if (TON_USE_ROCKSDB)
  if (ANDROID)
    set(PORTABLE ON CACHE BOOL "portable")
  endif()
  set(WITH_GFLAGS OFF CACHE BOOL "build with GFlags")
  set(WITH_TESTS OFF CACHE BOOL "build with tests")
  set(WITH_TOOLS OFF CACHE BOOL "build with tools")
  set(FAIL_ON_WARNINGS OFF CACHE BOOL "fail on warnings")
  message("Add rocksdb")
  add_subdirectory(ton/third-party/rocksdb EXCLUDE_FROM_ALL)
endif()

option(USE_COROUTINES "experimental support of coroutines" OFF)
if (USE_COROUTINES)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(TD_HAVE_COROUTINES 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines-ts")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
endif()

option(USE_LIBRAPTORQ "use libraptorq for tests" OFF)
if (USE_LIBRAPTORQ)
  set(USE_LZ4 OFF CACHE BOOL "use lz4")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CLANG_STDLIB "ON") # for libraptorq
  endif()
  message("Add libraptorq")
  add_subdirectory(ton/third-party/libraptorq EXCLUDE_FROM_ALL)
endif()

message("Add ton")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
#set(CCACHE_FOUND 0)
if (CCACHE_FOUND)
  message(STATUS "Found ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else()
  message(STATUS "Could NOT find ccache")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(GCC 1)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CLANG 1)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(INTEL 1)
elseif (NOT MSVC)
  message(FATAL_ERROR "Compiler isn't supported")
endif()

include(CheckCXXCompilerFlag)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)

if (TON_ARCH AND NOT MSVC)
  CHECK_CXX_COMPILER_FLAG( "-march=${TON_ARCH}" COMPILER_OPT_ARCH_SUPPORTED )
  if (TON_ARCH STREQUAL "apple-m1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${TON_ARCH}")	
  elseif(COMPILER_OPT_ARCH_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${TON_ARCH}")
  elseif(NOT TON_ARCH STREQUAL "native")
    message(FATAL_ERROR "Compiler doesn't support arch ${TON_ARCH}")
  endif()
endif()
if (THREADS_HAVE_PTHREAD_ARG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if (TON_USE_JEMALLOC)
  find_package(JeMalloc REQUIRED)
endif()

set(MEMPROF "" CACHE STRING "Use one of \"ON\", \"FAST\" or \"SAFE\" to enable memory profiling. \
Works under macOS and Linux when compiled using glibc. \
In FAST mode stack is unwinded only using frame pointers, which may fail. \
In SAFE mode stack is unwinded using backtrace function from execinfo.h, which may be very slow. \
By default both methods are used to achieve maximum speed and accuracy")

if (CLANG OR GCC)
  if (MEMPROF)
    check_cxx_compiler_flag(-no-pie CXX_NO_PIE_FLAG)
    if (CXX_NO_PIE_FLAG)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
    elseif (APPLE)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_pie")
    endif()
  endif()
endif()

if (MSVC)
  if (CMAKE_CXX_FLAGS_DEBUG MATCHES "/RTC1")
    string(REPLACE "/RTC1" " " CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  endif()
  add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /W4 /wd4100 /wd4127 /wd4324 /wd4456 /wd4457 /wd4458 /wd4505 /wd4702")
elseif (CLANG OR GCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
  if (APPLE)
    #use "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/export_list" for exported symbols
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fvisibility=hidden -Wl,-dead_strip,-x,-S")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -fvisibility=hidden -Wl,-dead_strip,-x,-S")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    if (NOT USE_EMSCRIPTEN)
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
    if (NOT TON_USE_ASAN AND NOT TON_USE_TSAN AND NOT MEMPROF)
      if (NOT USE_EMSCRIPTEN)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
      endif()
    endif()
  endif()
endif()

if (WIN32)
  add_definitions(-DNTDDI_VERSION=0x06020000 -DWINVER=0x0602 -D_WIN32_WINNT=0x0602 -DNOMINMAX -DUNICODE -D_UNICODE)
endif()
if (CYGWIN)
  add_definitions(-D_DEFAULT_SOURCE=1 -DFD_SETSIZE=4096)
endif()

if (NOT ANDROID) # _FILE_OFFSET_BITS is broken in ndk r15 and r15b and doesn't work prior to Android 7.0
  add_definitions(-D_FILE_OFFSET_BITS=64)
endif()

set(INTERNAL_COMPILE "0")
#BEGIN internal
  add_definitions(-D_INTERNAL_COMPILE=1)
  set(INTERNAL_COMPILE "1")
#END internal

set(TONLIB_COMPILE "0")
#BEGIN tonlib
  add_definitions(-D_TONLIB_COMPILE=1)
  set(TONLIB_COMPILE "1")
#END tonlib

include(ton/CMake/AddCXXCompilerFlag.cmake)
if (MSVC)
  add_cxx_compiler_flag("/experimental:external /external:anglebrackets /external:W0")
endif()
if (NOT MSVC)
  add_cxx_compiler_flag("-Wall")
  add_cxx_compiler_flag("-Wextra")
endif()

add_cxx_compiler_flag("-Wimplicit-fallthrough=2")
add_cxx_compiler_flag("-Wpointer-arith")
add_cxx_compiler_flag("-Wcast-qual")
add_cxx_compiler_flag("-Wsign-compare")
add_cxx_compiler_flag("-Wduplicated-branches")
add_cxx_compiler_flag("-Wduplicated-cond")
add_cxx_compiler_flag("-Walloc-zero")
add_cxx_compiler_flag("-Wlogical-op")
add_cxx_compiler_flag("-Wno-tautological-compare")
add_cxx_compiler_flag("-Wpointer-arith")
add_cxx_compiler_flag("-Wvla")
add_cxx_compiler_flag("-Wnon-virtual-dtor")
add_cxx_compiler_flag("-Wno-unused-parameter")
add_cxx_compiler_flag("-Wconversion")
add_cxx_compiler_flag("-Wno-sign-conversion")
add_cxx_compiler_flag("-Qunused-arguments")
add_cxx_compiler_flag("-Wno-unused-private-field")
add_cxx_compiler_flag("-Wno-redundant-move")
#add_cxx_compiler_flag("-Werror")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/include/c++/v1")
if (CLANG)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()
if (TON_USE_ASAN)
  if (CLANG)
    add_cxx_compiler_flag("-stdlib=libc++")
  endif()
  add_cxx_compiler_flag("-fsanitize=address")
  add_definitions(-DTD_USE_ASAN=1)
endif()
if (TON_USE_TSAN)
  if (CLANG)
    add_cxx_compiler_flag("-stdlib=libc++")
  endif()
  add_cxx_compiler_flag("-fsanitize=thread")
endif()
if (TON_USE_UBSAN)
  if (CLANG)
    add_cxx_compiler_flag("-stdlib=libc++")
  endif()
  add_cxx_compiler_flag("-fsanitize=undefined")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions")

#Compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM PUBLIC ${lib_include_dirs})
    target_link_libraries(${target} PUBLIC ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)

set(TDUTILS_MIME_TYPE OFF CACHE BOOL "Generate mime type conversion")
add_subdirectory(ton/tdutils)
add_subdirectory(ton/memprof)
add_subdirectory(ton/tdactor)
add_subdirectory(ton/tdnet)
if (TON_USE_ROCKSDB)
  option(TDDB_USE_ROCKSDB "Use rockdb" ON)
endif()
add_subdirectory(ton/tddb)
add_subdirectory(ton/tdtl)
add_subdirectory(ton/tl)
add_subdirectory(ton/terminal)
add_subdirectory(ton/keys)
add_subdirectory(ton/tl-utils)
add_subdirectory(ton/adnl)
add_subdirectory(ton/crypto)
add_subdirectory(ton/lite-client)
add_subdirectory(ton/emulator)

#BEGIN tonlib
add_subdirectory(ton/tonlib)
#END tonlib

target_link_libraries(generator Tonlib)
target_link_libraries(generator ton_crypto)