# Address Generator
This is command line utility, which finds mnemonic and `wallet_id`, corresponding to an address of Wallet V3 R2, matching specific constraints.
Beginning and end of the address may be constrained

## ATTENTION
Generated address will have non-default `wallet_id` parameter, it must be saved along with mnemonic.
Unfortunately, there are no established wallets, supporting non-standard `wallet_id`.

You can import generated mnemonic and `wallet_id` to [Ton Air Gap wallet](https://github.com/ton-offline-storage/)
(It is not mandatory to use this wallet in offline setting)

Alternative is to use [TON binaries](https://docs.ton.org/develop/smart-contracts/environment/installation),
similar to guide described [here](https://github.com/ton-defi-org/ton-offline-transaction)

## Usage

### CPU Generator
Download binary (`cpu-generator`) for your platform from latest [release](https://github.com/ton-offline-storage/address-generator/releases) and run it in terminal, follow instructions.

### GPU Generator
Make sure you have NVIDIA GPU in your system. Optionally, update your GPU [driver](https://nvidia.com/drivers).

Download binary (`gpu-generator`) for your platform from latest [release](https://github.com/ton-offline-storage/address-generator/releases).
Run it in terminal, follow insturctions. 

#### Run options

Generator launches benchmark, to determine best parameters (`BLOCKS` and `THREADS`) for your GPU.
You can reuse this parameters, and skip benchmark by using flags, run binary from terminal the following way:

`gpu-generator -B <BLOCKS> -T <THREADS>`, where `<BLOCKS>` and `<THREADS>` are replaced by corresponding numbers

You can specify one parameter to benchmark another one. You can run only benchmark:

`gpu-generator -b`

#### Tips

By the way Tails OS is a distributive of Linux, so Linux version will go

## Perfomance
Table shows average time for finding an address, depending on CPU and number of characters constrained (to a single option).

|       | AMD Ryzen 5 3600(6C, 12T) | Intel Core i5-8350U(4C, 8T) |
| ----- | ------------------------- | --------------------------- |
| 4 characters |    < 1 second      |          3 seconds          |
| 5 characters |       26 seconds   |        4 min 20 sec         |
| 6 characters |      30 minutes    |        4 hours 40 min       |
| 7 characters |    31 hour 30 min  |        300 hours            |

## Constraints description
*This is a copy of description, displayed when running the generator*

### TLDR
Simple examples of constraint commands

`end[T][O][N]`, example result `UQCF1NqCCBCUaDsmm1fC51wMpV0vwt3loKf997ArRtNxmTON`

`start[*][T][O][N]`, example result `UQBTONmtYBErhtwdQqQQjUkracKxLkU6Kb7qJr3awhoWqLGJ`

### Full description
You can specify a number of consecutive symbols at the end of address.
For every symbol constraint looks like `[...]`, and inside brackets are
the characters you allow symbol at this position to be, or `*` if you allow any character.
Remember, that TON address consists only of characters from `A-Z`, `a-z`, `0-9` and `_` and `-`

To constrain end of the address, use `end` command with list of constraints for symbols,
like this: 

`end[T][O][N]`, or like this: `end[Tt][Oo][Nn]`

You can also specify a number of symbols at the start of address, but
TON address always starts with 2 characters `UQ`, and you can't change that.
Third symbol, after `UQ`, can only be one of `A`, `B`, `C`, `D`.

Use `start` command similarly to `end` command,
but remember, the first symbol in your command is third in the address.
With that in mind, you can specify start of the address like this:

`start[A][P][P][L][E]`, or like this: `start[*][T][O][N]`

You can also specify both start and end at the same time, like this:

`start[*][T][O][N] & end[T][O][N]`

You can also add several variants of constraints, such that any
of this constraints, if matched, satisfies you, like this:

`start[*][T][O][N] & end[T][O][N] | start[D][D][D] | end[0][0][0]`


## Compile from source

### CPU Generator
#### Linux

1. Run following commands in terminal:
   ```
   sudo apt install git build-essential pkg-config zlib1g-dev openssl libssl-dev && sudo snap install cmake --classic
   git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git
   ```
2. In the file `address-generator\ton\tonlib\tonlib\keys\Mnemonic.cpp` comment out line `221` (this line is 6-th from the end) using `//`.
   Line should look like this:
   
   `//LOG(INFO) << "Mnemonic generation debug stats: " << A << " " << B << " " << C << " " << timer;`
4. From the same directory(containing `address-generator` folder), run following commands in terminal:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build .
   ```
5. Binary with name `generator` will appear in folder `build`


#### Windows

1. Install git, for example from [here](https://gitforwindows.org/) or [here](https://git-scm.com/download/win)
2. Install cmake 3.27.4, e.g from [here](https://cmake.org/files/v3.27/), [direct link to 64 bit version](https://cmake.org/files/v3.27/cmake-3.27.4-windows-x86_64.msi). During installation, choose "Add cmake to PATH"
3. Install C++ compiler, for that install msys e.g from [here](https://www.msys2.org/). After installation finishes, in opened msys terminal run
   `pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain`. If you installed msys in root folder (default) add `C:\msys64\ucrt64\bin`, to windows system PATH variable. Otherwise adjust `C:\msys64\ucrt64\bin` accordingly.
4. Open windows command line, move to the directory, where you want to compile code
5. Run `git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git`
6. In the file `address-generator\ton\tonlib\tonlib\keys\Mnemonic.cpp` comment out line `221` (this line is 6-th from the end) using `//`.
   Line should look like this:
   
   `//LOG(INFO) << "Mnemonic generation debug stats: " << A << " " << B << " " << C << " " << timer;`
8. In directory, containing `address-generator` folder, run following commands:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ..
   cmake --build .
   ```
9. Binary with name `generator` will appear in folder `build`

### GPU Generator

#### Windows

1. Install git, for example from [here](https://gitforwindows.org/) or [here](https://git-scm.com/download/win)
2. Install cmake 3.27.4, e.g from [here](https://cmake.org/files/v3.27/), [direct link to 64 bit version](https://cmake.org/files/v3.27/cmake-3.27.4-windows-x86_64.msi). During installation, choose "Add cmake to PATH"
3. Install Build Tools for Visual Studio 2022 from [here](https://visualstudio.microsoft.com/downloads/#remote-tools-for-visual-studio-2022).
   During installation, check the Desktop development with C++ workload and select Install.
   ![MSVC installation](https://code.visualstudio.com/assets/docs/cpp/msvc/desktop_development_with_cpp-2022.png)
4. Add MSVC compiler to system `PATH` variable, usually it is installed to directory looking like this:
   
   `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64`, note that you may have another version.
5. Install NVIDIA CUDA Toolkit from [here](https://developer.nvidia.com/cuda-downloads)
6. Copy all files from (you may have other version)
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\extras\visual_studio_integration\MSBuildExtensions`
   to
   `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations`
7. Install `zlib` from [here](https://gnuwin32.sourceforge.net/downlinks/zlib.php), and add `C:\Program Files (x86)\GnuWin32\bin` to system `PATH` variable
8. In file `C:\Program Files (x86)\GnuWin32\include\zconf.h` find following code:
   ```
   #if 1           /* HAVE_UNISTD_H -- this line is updated by ./configure */
   #  include <sys/types.h> /* for off_t */
   #  include <unistd.h>    /* for SEEK_* and off_t */
   #  ifdef VMS
   #    include <unixio.h>   /* for off_t */
   #  endif
   #  define z_off_t off_t
   #endif
   ```
   Change first line to `#if 0           /* HAVE_UNISTD_H -- this line is updated by ./configure */` (You may need to change file permissions in file properties)
9. Install `pkg-config`, download following
   
   - `pkg-config_0.26-1_win32.zip` from [here](http://ftp.gnome.org/pub/gnome/binaries/win32/dependencies), extract and move `bin/pkg-config.exe` to `C:\Program Files (x86)\GnuWin32\bin`
   - `gettext-runtime_0.18.1.1-2_win32.zip` from [here](http://ftp.gnome.org/pub/gnome/binaries/win32/dependencies), extract and move `bin/intl.dll` to `C:\Program Files (x86)\GnuWin32\bin`
   - `glib_2.28.8-1_win32.zip` from [here](https://download.gnome.org/binaries/win32/glib/2.28/),  extract and move `bin/libglib-2.0-0.dll` to `C:\Program Files (x86)\GnuWin32\bin`
 
10. Somehow install OpenSSL for windows, for example from [here](https://slproweb.com/products/Win32OpenSSL.html)
11. If you installed OpenSSL from above link, add `C:\Program Files\OpenSSL-Win64\bin` to system `PATH` variable, and copy all files from `C:\Program Files\OpenSSL-Win64\lib\VC\x64\MT` to
    `C:\Program Files\OpenSSL-Win64\lib`
12. Open windows command line, move to the directory, where you want to compile code
13. Run `git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git`
14. In the file `address-generator\ton\tonlib\tonlib\keys\Mnemonic.cpp` comment out line `221` (this line is 6-th from the end) using `//`.
   Line should look like this:
   
   `//LOG(INFO) << "Mnemonic generation debug stats: " << A << " " << B << " " << C << " " << timer;`
   
15. From the same directory(containing `address-generator` folder), run following commands in terminal:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_GENERATOR=TRUE ..
   cmake --build .
   ```
16. Binary with name `generator` will appear in folder `build\Debug`
