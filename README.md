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
Download binary for your platform from latest [release](https://github.com/ton-offline-storage/address-generator/releases) and run it in terminal, follow instructions.

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

### Linux

1. Run following commands in terminal:
   ```
   sudo apt install git build-essential pkg-config zlib1g-dev openssl libssl-dev && sudo snap install cmake --classic
   git clone --recurse-submodules https://github.com/ton-offline-storage/address-generator.git
   cd address-generator
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build .
   ```
2. Binary with name `generator` will appear in folder `build`


### Windows

1. Install git, for example from [here](https://gitforwindows.org/) or [here](https://git-scm.com/download/win)
2. Install cmake 3.27.4, e.g from [here](https://cmake.org/files/v3.27/), [direct link to 64 bit version](https://cmake.org/files/v3.27/cmake-3.27.4-windows-x86_64.msi). During installation, choose "Add cmake to PATH"
3. Install C++ compiler, for that install msys e.g from [here](https://www.msys2.org/). After installation finishes, in opened msys terminal run
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
