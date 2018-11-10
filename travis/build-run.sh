#! /bin/sh

# Exit on error
set -ev

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"
OSHMPI_IMPL=$TRAVIS_ROOT
SOS_IMPL=$TRAVIS_ROOT/"tests-sos"
# Environment variables
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
    mpich_shm|mpich_odd)
        #mpichversion
        mpicc -show
        mpicxx -show
        ;;
    openmpi)
        # this is missing with Mac build it seems
        #ompi_info --arch --config
        mpicc --showme:command
        mpicxx --showme:command
        # see https://github.com/open-mpi/ompi/issues/2956
        export TMPDIR=/tmp
        ;;
esac

# OSHMPI configure and build
./autogen.sh
./configure CC=mpicc CXX=mpicxx CFLAGS="-std=c99" --prefix=$OSHMPI_IMPL
make V=1
make V=1 install

# SOS test suite configure and build
git clone https://github.com/openshmem-org/tests-sos.git
cd tests-sos
./autogen.sh
./configure CC=mpicc CXX=mpicxx CFLAGS="-I$OSHMPI_IMPL/include" LDFLAGS="-L$OSHMPI_IMPL/lib -loshmpi" \
  --prefix=$SOS_IMPL  \
  --enable-lengthy-tests --disable-fortran --disable-cxx # disable cxx tests because non-standard apis ar used
make
make install

cd $SOS_IMPL

# Set test configuration
export OSHMPI_VERBOSE=1
export MPIEXEC_TIMEOUT=600 # in seconds

MPICH_ODD_EVEN_CLIQUES=0
TEST_MPIEXEC=
case "$MPI_IMPL" in
    mpich_shm)
        TEST_MPIEXEC="mpiexec -np"
        ;;
    mpich_odd)
        TEST_MPIEXEC="mpiexec -np"
        MPICH_ODD_EVEN_CLIQUES=1
        ;;
    openmpi)
        # --oversubscribe fixes error "Either request fewer slots for your
        # application, or make more slots available for use." on osx.
        # see https://github.com/open-mpi/ompi/issues/3133
        TEST_MPIEXEC="mpiexec --oversubscribe -np"
        ;;
esac

NP=2

# Run unit tests
export MPIR_CVAR_ODD_EVEN_CLIQUES=$MPICH_ODD_EVEN_CLIQUES
export OSHMPI_VERBOSE=1
echo "Run sos tests with MPIR_CVAR_ODD_EVEN_CLIQUES=$MPIR_CVAR_ODD_EVEN_CLIQUES, NP=$NP"
$TEST_MPIEXEC $NP $SOS_IMPL/bin/shmem_latency_put_perf
make check TEST_RUNNER="$TEST_MPIEXEC $NP"
cat test/unit/test-suite.log
