# Address Generator
This is command line utility, which finds mnemonic and `wallet_id`, corresponding to an address, matching specific constraints.
Beginning and end of the address may be constrained

## ATTENTION
Generated address will have non-default `wallet_id` parameter, it must be saved along with mnemonic. Thus, there are a limited list of wallets, where this address may be imported to.

## Usage
Download binary for your platform from latest release and run it in terminal, follow instructions.

## Compile from source
### Windows

1. Install git, for example from [here](https://gitforwindows.org/) or [here](https://git-scm.com/download/win)
2. Install cmake 3.27.4, e.g from [here](https://cmake.org/files/v3.27/), [direct link to 64 bit version](https://cmake.org/files/v3.27/cmake-3.27.4-windows-x86_64.msi). During installation, choose "Add cmake to PATH"
3. Install C++ compiler, for that install msys e.g from [here](https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe). After installation finishes, in opened msys terminal run
   `pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain`. If you installed msys in root folder (default) add `C:\msys64\ucrt64\bin`, to windows system PATH variable. Otherwise adjust `C:\msys64\ucrt64\bin` accordingly.
4. Open windows command line, move to the directory, where you want to compile code
5. Run following commands:
   ```
   git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ..
   cmake --build .
   ```
6. Binary with name `generator` will appear in folder `build`
