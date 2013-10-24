OSHMPI: OpenSHMEM over MPI-3

Platform Support
=====

OSHMPI uses essentially all the new RMA features in MPI-3 and thus 
requires an MPI implementation that supports these properly.
Currently, MPICH 3.0.x and its derivatives support MPI-3.
We assume support for the UNIFIED memory model of MPI-3;
OSHMPI will abort if this is not provided by the implementation,
which should only be the case for non-cache-coherent systems.

Because SHMEM allows for communication against remote global 
variables and not just the symmetric heap, OSHMPI has an 
operating system dependency since accessing the text and data
segments is not portable.  Currently, we only support Linux;
Apple OSX support is not working but may be fixed in the future.
However, SHMEM programs that only communicate with symmetric
heap variables should be portable.

The platforms we currently test on are:
* Mac with LLVM 3.3+ and MPICH master
* Linux x86_64 with MPICH 3.0.x and MVAPICH2 1.9.x
 
Features
=====

OSHMPI attempts to use MPI-3 as effectively as possible.
To this end, we support all valid performance-related info
keys and ensure the correct semantics when they are enabled
and disabled.

When OSHMPI is used within an SMP, we employ shared-memory 
windows to bypass MPI in shmem_{put,get} and use only
load-store instructions.  However, for strided and atomic
operations, we still use MPI within an SMP for convenience.

Future Work
=====

* Eliminate all intranode MPI-RMA communication.
* Cache subcommunicators corresponding to PE subgroups.

Bugs
=====

See comments above regarding support for non-heap symmetric
data on Mac.
