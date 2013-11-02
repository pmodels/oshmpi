OSHMPI: OpenSHMEM over MPI-3
==

Publications
=====

Jeff R. Hammond, Sayan Ghosh, and Barbara M. Chapman, 
"Implementing OpenSHMEM using MPI-3 one-sided communication."
Preprint: https://github.com/jeffhammond/oshmpi/blob/master/docs/iwosh-paper.pdf

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
segments is not portable.

The platforms we currently test on are:
* Mac with LLVM 3.3+ and MPICH master (see Bugs below)
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

* Allow selection of tuning options at runtime.
* Eliminate all intranode MPI-RMA communication.
* Cache subcommunicators corresponding to PE subgroups.
* Support Cray XE/XK/XC systems.
* Support Intel Xeon Phi (MIC) systems.
* Support Power-based Linux systems.
* Support Blue Gene/Q (currently lacks MPI-3 support).

Bugs
=====

* Mac non-heap symmetric data cannot be accessed remotely.
