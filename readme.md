# SubGraph Isomorphism 

## Introduction
This project is a subgraph isomorphism implementation in C++ with CUDA.

## Build

### Prerequisites
* CUDA 10.1 or higher
* CMake 3.9 or higher
* C++ compiler with C++17 support

### Build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Run
```bash
cd build
./SubGraph -p <pattern_file> -d <data_file> -o <output_file> [-t <thread_num>] [-b <block_num>]
```
