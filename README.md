# Address Generator
This is command line utility, which finds mnemonic and `wallet_id`, corresponding to an address of Wallet V3 R2, matching specific constraints.
Beginning and end of the address may be constrained

like that, `UQCF1NqCCBCUaDsmm1fC51wMpV0vwt3loKf997ArRtJ_123456`

or like that `UQAPPLE4APeEfnGAbTIbXYGALf0rp9ALm5iJuFvddzg-6A0I`

## ATTENTION (Wallet ID)
Generated address will have non-default `wallet_id` parameter, it must be saved along with mnemonic.
Unfortunately, not every wallet supports non-standard `wallet_id`.

- You can import generated mnemonic and `wallet_id` to [Ton Air Gap wallet](https://github.com/ton-offline-storage/)
(It is not mandatory to use this wallet in offline setting)
- You can import to [TonDevWallet](https://github.com/TonDevWallet/TonDevWallet)
- Alternative is to use [TON binaries](https://docs.ton.org/develop/smart-contracts/environment/installation),
similar to guide described [here](https://github.com/ton-defi-org/ton-offline-transaction)

## Usage

### CPU Generator
Download binary (`cpu-generator`) for your platform from latest [release](https://github.com/ton-offline-storage/address-generator/releases) and run it in terminal, follow instructions. Linux version is suitable for Tails OS.

### GPU Generator
Make sure you have NVIDIA GPU in your system. You may need to update your GPU [driver](https://nvidia.com/drivers). Driver `555.*` or later is required

Download binary (`gpu-generator`) for your platform from latest [release](https://github.com/ton-offline-storage/address-generator/releases).
Run it in terminal, follow insturctions. 

If you want to use GPU generator in Tails OS, you may need to take additional actions, read [below](#gpu-generator-in-offline-system)

#### Run options

Generator launches benchmark, to determine best parameters (`BLOCKS` and `THREADS`) for your GPU.
You can reuse this parameters, and skip benchmark by using flags, run binary from terminal the following way:

`gpu-generator -B <BLOCKS> -T <THREADS>`, where `<BLOCKS>` and `<THREADS>` are replaced by corresponding numbers

You can specify one parameter to benchmark another one. You can run only benchmark:

`gpu-generator -b`

 You can specify constraints in command line (instead of typing after launch) with `-q` option:

 `gpu-generator -q "start[*][T][O][N] | end[1][2][3]"`


## Perfomance
Table shows average time for finding an address, depending on hardware and number of characters constrained (to a single option).

Lower numbers of characters are a matter of seconds for any hardware.

|                                 | 5 characters |  6 characters  |  7 characters  | 8 characters |
| ------------------------------- | ------------ | -------------- | -------------- | ------------ |
| **Intel Core i5-8350U(4C, 8T)** | 4 min 20 sec |   4 h 40 min   |   12,5 days    |  > 2 years   | 
| **AMD Ryzen 5 3600(6C, 12T)**   |    26 sec    |    30 min      | 31 h 30 min    |   84 days    |
|   **NVIDIA GTX 1650 SUPER**     |    2 sec     |     2 min      |    2 hours     |   5,5 days   |
|      **NVIDIA RTX 4090**        |    0 sec     |     13 sec     |    13,5 min    |  14,5 hours  |

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

## GPU generator in offline system

You may want to use a GPU generator on an offline computer, along with Ton Air Gap Wallet, to secure generated mnemonic words. There won't be any problems with CPU generator,
but for the GPU one you need to install an NVIDIA driver, and it is tricky to do it in Tails OS.
There is an option to buy not a USB stick, but a hard drive specifically for Air Gap Wallet and GPU generator, and install more common non-amnesic system like Windows/Ubuntu. Installing such OS on a USB stick is problematic because of low I/O speeds of USB sticks.

But it is not necessary, you can make a GPU generator running in an OS running from USB stick, one option is to use [Puppy Linux OS](https://puppylinux-woof-ce.github.io/). Here is the guide explaining how to make the GPU generator running in offline Puppy Linux. You can use Puppy Linux only once to use Generator, and then use received mnemonic phrase in Tails OS, or use Puppy Linux instead of Tails OS. We consider Tails OS a more secure option.

1. You need one USB stick to install Puppy Linux to (primary stick), and one auxiliary
2. Download Puppy Linux ISO image from official [page](https://forum.puppylinux.com/puppy-linux-collection), we recommend Bookworm 64-bit.
 You'll get to [this page](https://rockedge.org/kernels/data/ISO/Bookworm_Pup64/). Download  `BookwormPup64_10.0.7.iso`, and two files -
`devx_dpupbw64_10.0.7.sfs` and `kernel_sources-6.1.94-dpupbw64.sfs`, you will need them later.
3. Copy all 3 files to auxiliary USB stick, along with GPU-generator and, if you want to use offline wallet here, not in Tails OS, copy offline client also
4. Download NVIDIA [driver](https://nvidia.com/drivers). Choose Linux 64-bit OS, download driver `555.*` or later. Copy it to auxiliary USB.
5. Flash ISO image (`BookwormPup64_10.0.7.iso`) to the primary USB stick, using, for example, [balena etcher](https://etcher.balena.io/). Open etcher and follow instructions
6. Boot from primary USB, don't forget to disconnect computer from internet
7. In Puppy Linux: Menu(left-down corner) -> Setup -> Puppy Installer -> BootFlash -> Choose the primary USB stick -> Choose `Create UEFI USB` (First option) -> Choose `FAT32 partition + F2FS partition` -> confirm
8. Now opened the window for choosing Puppy Linux ISO image. Plug in auxiliary USB, copy ISO file to Puppy filesystem, and choose it in the window
9. After installation finishes, reboot, and choose to save the session. If asked, choose your primary USB stick(It's second f2fs partition. Disconnect other USBs if unsure) -> normal(no encryption) -> Choose to save in a folder -> ok -> Yes, save
10. Copy `devx_dpupbw64_10.0.7.sfs` and `kernel_sources-6.1.94-dpupbw64.sfs` from auxiliary USB to Puppy filesystem. For each, click on it -> Install SFS
11. Copy driver (`.run` file) to Puppy, run it, follow instructions. If you don't understand choices, choose the first option. **HOWEVER**, when asked:
`Would you like to run the nvidia-xconfig utility to automatically update your X configuration file...` choose **No**.
12. Reboot, choose to save the session.
13. Copy `gpu-generator`, and if needed, offline-client, to Puppy Linux. Now the generator will work.
    
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
3. From the same directory(containing `address-generator` folder), run following commands in terminal:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build .
   ```
4. Binary with name `generator` will appear in folder `build`


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
7. In directory, containing `address-generator` folder, run following commands:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ..
   cmake --build .
   ```
8. Binary with name `generator` will appear in folder `build`

### GPU Generator

#### Linux

1. Run following command in terminal:
   ```
   sudo apt install git build-essential pkg-config zlib1g-dev openssl libssl-dev && sudo snap install cmake --classic
   ```
2. Install CUDA from [here](https://developer.nvidia.com/cuda-downloads). Add following lines to your `.bashrc` file, located in `home` directory. You may need to change `cuda-12.5` to your CUDA version.
   ```
   export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```
   You may enable "Show hidden files" in your `home` directory, or use vim/nano/etc. to edit the file
3. Run `git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git`
2. In the file `address-generator\ton\tonlib\tonlib\keys\Mnemonic.cpp` comment out line `221` (this line is 6-th from the end) using `//`.
   Line should look like this:
   
   `//LOG(INFO) << "Mnemonic generation debug stats: " << A << " " << B << " " << C << " " << timer;`
3. From the same directory(containing `address-generator` folder), run following commands in terminal:
   ```
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_GENERATOR=TRUE ..
   cmake --build .
   ```
4. Binary with name `generator` will appear in folder `build`


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
7. Somehow install OpenSSL for windows, for example from [here](https://slproweb.com/products/Win32OpenSSL.html)
8. If you installed OpenSSL from above link, add `C:\Program Files\OpenSSL-Win64\bin` to system `PATH` variable, and copy all files from `C:\Program Files\OpenSSL-Win64\lib\VC\x64\MT` to
    `C:\Program Files\OpenSSL-Win64\lib`
9. Install msys e.g from [here](https://www.msys2.org/). After installation finishes, in opened msys terminal run
   `pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain`
10.  - Copy `zlib1.dll` from `C:\msys64\ucrt64\bin` to `C:\Program Files\OpenSSL-Win64\bin`
     - Copy `zconf.h` and `zlib.h` from `C:\msys64\ucrt64\include` to `C:\Program Files\OpenSSL-Win64\include`
     - Copy `libz.a` and `libz.dll.a` from `C:\msys64\ucrt64\lib` to `C:\Program Files\OpenSSL-Win64\lib`
     - Copy `pkg-config.exe` and `libpkgconf-5.dll` from `C:\msys64\ucrt64\bin` to `C:\Program Files\OpenSSL-Win64\bin`
11. In file `C:\Program Files\OpenSSL-Win64\include\zconf.h` find following code:
    ```
    #if 1    /* was set to #if 1 by ./configure */
    #  define Z_HAVE_UNISTD_H
    #endif
    ```
    Change first line to `#if 0    /* was set to #if 1 by ./configure */` (You may need to change file permissions in file properties)
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
