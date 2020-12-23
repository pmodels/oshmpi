## Introduction
The OSHMPI project provides an OpenSHMEM 1.4 implementation on top of MPI-3.
MPI is the standard communication library for HPC parallel programming. OSHMPI
provides a lightweight OpenSHMEM implementation on top of the portable MPI-3
interface and thus can utilize various high-performance MPI libraries on
HPC systems.


## Getting Started
#### 1. Installation
The simplest way to configure OSHMPI is using the following command-line:
```
./configure --prefix=/your/oshmpi/installation/dir CC=/path/to/mpicc CXX=/path/to/mpic++
```
Once you are done configuring, you can build the source and install it using:
```
make && make install
```

#### 2. Test
OSHMPI contains some simple test programs under the `tests/` directory. To check
if OSHMPI built correctly:
```
cd tests/ && make check
```
For comprehensive test, we recommend using external test suites such as
[SOS Test Suite](https://github.com/openshmem-org/tests-sos).

#### 3. Compile an OpenSHMEM program with OSHMPI
OSHMPI supports C and C++ programs. Example to compile the C program `test.c`:
```
/your/oshmpi/installation/dir/bin/oshcc -o test test.c
```
Example to compile the C++ program `test.cpp`:
```
/your/oshmpi/installation/dir/bin/oshc++ -o test test.cpp
```

#### 4. Execute an OSHMPI program
OSHMPI relies on the MPI library's startup command `mpiexec` or `mpirun`
to execute programs. Example to run the binary `test`:
```
/path/to/mpiexec -np 2 ./test
```
For more information about the MPI startup command, please check the MPI
library's documentation.


## Configure Options

Below are some commonly used configure options. For detailed explanation
please check `./configure --help`.

- Default configuration
```
./configure --prefix=/your/oshmpi/installation/dir CC=/path/to/mpicc CXX=/path/to/mpic++
```

- With fast build, no multi-threading, disable all active messages.
```
./configure --prefix=/your/oshmpi/installation/dir CC=/path/to/mpicc CXX=/path/to/mpic++ \
    --enable-fast --enable-threads=single --enable-amo=direct --enable-rma=direct
```

- With MPICH/OFI version 3.4b1 or newer, enable
  **dynamic window enhancement** for scalable memory space implementation.
```
./configure --prefix=/your/oshmpi/installation/dir CC=/path/to/mpicc CXX=/path/to/mpic++ \
    --enable-win-type=dynamic_win
```

- With CUDA GPU memory space support
```
./configure --prefix=/your/oshmpi/installation/dir CC=/path/to/mpicc CXX=/path/to/mpic++ \
    --with-cuda=/path/to/cuda/installation
```

## Examples

Examples to use memory space extension and CUDA GPU memory kind can be found at the
built-in test suite:
```
  tests/space.c
  tests/space_int_amo.c
  tests/space_int_put.c
  tests/space_ctx_int_put.c
```

## Environment Variables
#### OpenSHMEM Standard Environment Variables
  - **SHMEM_SYMMETRIC_SIZE** (default 128 MiB)
      Number of bytes to allocate for symmetric heap. The size value can be
      either number of bytes or scaled with a suffix of:
      + "K" or "k" for kilobytes (B * 1024)
      + "M" or "m" for Megabytes (KiB * 1024)
      + "G" or "g" for Gigabytes (MiB * 1024)
      + "T" or "t" for Terabytes (GiB * 1024)
  - **SHMEM_DEBUG** (0|1, default 0)
      Enables debugging messages from the OpenSHMEM runtime. It is
      invalid when configured with `--enable-fast` (see `--enable-fast`).

  - **SHMEM_VERSION** (0|1, default 0)
      Prints OpenSHMEM library version at `start_pes()`, `shmem_init()`,
      or `shmem_init_thread()`.

  - **SHMEM_INFO** (0|1, default 0)
      Prints the list of OpenSHMEM environment variables at `stdout`.

#### OSHMPI Environment Variables
  - **OSHMPI_VERBOSE** (0|1, default 0)
      Prints both the standard and OSHMPI environment variables at `stdout`.

  - **OSHMPI_AMO_OPS**  (Comma-separated operations, default "all")
      Defines the AMO operations used in the OpenSHMEM program. If all PEs issue
      concurrent AMOs only with the same operation, or with the same operation
      and `fetch`, then OSHMPI can directly utilize MPI accumulate operations.
      This is because, MPI grantees the atomicity of accumulate operations
      only with `same_op` or `same_op_no_op`.
      The default value of `OSHMPI_AMO_OPS` is
      `"cswap,finc,inc,fadd,add,fetch,set,swap,fand,and,for,or,fxor,xor"`
      (identical to `"all"`). Arbitrary combination of the above operations can
      be given at execution time.

      The variable can be adjusted only with configure `--enable-amo=auto`.

  - **OSHMPI_MPI_GPU_FEATURES** (Comma-separated features, default "all")
      Defines the list of GPU-aware communication functions provided by
      the underlying MPI library.
      The default value is `"pt2pt,put,get,acc"` (identical to `"all"`).
      Arbitrary combination of the above features can be given at execution
      time. If none of the features is supported, specify `"none"`.


  - **OSHMPI_ENABLE_ASYNC_THREAD** (0|1, default 0)
      Runtime control of asynchronous progress thread when MPI point-to-point based
      active messages are used internally. Both AMO and RMA may use the active
      message based approach:
      + When AMO cannot be directly translated to MPI accumulates (see `OSHMPI_AMO_OPS`),
        each AMO operation is issued via active messages.
      + When GPU buffer is used in an RMA operation but the MPI library does not support
        GPU awareness in the RMA mode (see `OSHMPI_MPI_GPU_FEATURES`),
        each RMA operation is issued via active messages.
      The variable can be adjusted only with configure `--enable-async-thread=auto`
      and `--enable-threads=multiple`.


## Debugging Options
- Enable debugging flag by setting the configure option `--enable-g`.
- Set environment variable `SHMEM_DEBUG=1` to print OSHMPI internal debugging message.


## Test Framework
OSHMPI uses [SOS Test Suite](https://github.com/openshmem-org/tests-sos) for
correctness test.
- Tested platforms: CentOS, MacOS (compilation-only)
- Tested MPI implementations:
    + MPICH-3.4rc1 (with CH3/TCP, CH4/OFI, or CH4/UCX)
    + MPICH main branch (with CH3/TCP, CH4/OFI, or CH4/UCX)
    + OpenMPI 4.0.5 (with UCX)


## Known Issues
1. Some OpenSHMEM features are not supported in OSHMPI.
    - **Context**:
    OSHMPI cannot create separate or shared context on top of MPI interfaces.
    This feature may be enabled if the MPI end-point concept is accepted
    at MPI forum and implemented in MPI libraries. Current version always
    returns `SHMEM_NO_CTX` error at `shmem_ctx_create()`; `shmem_ctx_destroy()`
    is a no-op.

2. OSHMPI does not provide Fortran APIs.

3. OSHMPI may not work on 32-bit platforms. This is because some internal routines
   rely on 64-bit integer, e.g., `shmem_set_lock()`, `shmem_clear_lock()`, `shmem_test_lock`.

4. OSHMPI may not correctly initialize symmetric data on OSX platform.


## Support
If you have problems or need any assistance about the OSHMPI
installation and usage, please contact oshmpi-users@lists.mcs.anl.gov mailing
list.


## Bug Reporting
If you have found a bug in OSHMPI, please contact
oshmpi-users@lists.mcs.anl.gov mailing list, or create an issue ticket on
github: https://github.com/pmodels/oshmpi.  If possible, please try to
reproduce the error with a smaller application or benchmark and send
that along in your bug report.
