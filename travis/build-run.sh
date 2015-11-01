#! /bin/sh

# Exit on error
set -ev

# Environment variables
export CFLAGS="-std=c99"
#export MPICH_CC=$CC
export MPICC=mpicc

SMP_OPT="$1"

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
make check
