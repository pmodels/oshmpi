#! /bin/sh

# Exit on error
set -ev

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"
SMP_OPT="$3"

# Environment variables
export CFLAGS="-std=c99"
#export MPICH_CC=$CC
export MPICC=mpicc

case "$os" in
    Darwin)
        ;;
    Linux)
       export PATH=$TRAVIS_ROOT/mpich/bin:$PATH
       export PATH=$TRAVIS_ROOT/open-mpi/bin:$PATH
       ;;
esac

# Capture details of build
case "$MPI_IMPL" in
    mpich)
        mpichversion
        mpicc -show
        ;;
    openmpi)
        mpicc --showme:command
        # see https://github.com/open-mpi/ompi/issues/2956
        export TMPDIR=/tmp
        ;;
esac

# Configure and build
./autogen.sh
case "$SMP_OPT" in
    0)
        ./configure --enable-g --disable-static
        ;;
    1)
        ./configure --enable-g --disable-static --enable-smp-optimizations
        ;;
esac

# Run unit tests
export SHMEM_SYMMETRIC_HEAP_SIZE=100M
make check
