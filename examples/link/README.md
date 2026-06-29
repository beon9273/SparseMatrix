# link example

Demonstrates consuming sparsemat as an installed CMake package via `find_package`.

The installed CMake target is `sparsemat::sparsemat`:

```cmake
find_package(sparsemat REQUIRED)
target_link_libraries(my_target PRIVATE sparsemat::sparsemat)
```

## Usage

First install sparsemat into a local prefix:

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=~/sparsemat-install
cmake --build build
cmake --install build
```

Then build this example pointing CMake at that prefix:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=~/sparsemat-install
cmake --build build
./build/link_example
```
