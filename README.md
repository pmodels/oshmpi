OSHMPI: OpenSHMEM over MPI-3
============================

**This project is no longer actively developed.**

Bug fixes and specific feature requests will be addressed as time permits.  We will try to read and comment on issues and pull requests promptly.

We recommend you use [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) to get the latest OpenSHMEM features.  We tried to support OpenSHMEM 1.2 but have no plans to support OpenSHMEM 1.3 or later.

Test Status
===========

We currently test with MPICH 3.2 on Ubuntu.

[![Build Status](https://travis-ci.org/jeffhammond/oshmpi.svg?branch=master)](https://travis-ci.org/jeffhammond/oshmpi)

Publications
============

Jeff R. Hammond, Sayan Ghosh, and Barbara M. Chapman, 
"Implementing OpenSHMEM using MPI-3 one-sided communication."  
Preprint: https://github.com/jeffhammond/oshmpi/blob/master/docs/iwosh-paper.pdf  
Workshop Proceedings: http://www.csm.ornl.gov/workshops/openshmem2013/documents/ImplementingOpenSHMEM%20UsingMPI-3.pdf  
Journal: http://dx.doi.org/10.1007/978-3-319-05215-1_4  

Platform Support
================

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
* Mac with GCC 4.8 or LLVM 3.3+ and MPICH master (see Bugs below)
* Linux x86_64 with MPICH 3+ and MVAPICH2 1.9+

We have tested on these platforms at one point or another:
* SGI ccNUMA with MPICH 3
 
Features
========

OSHMPI attempts to use MPI-3 as effectively as possible.
To this end, we support all valid performance-related info
keys and ensure the correct semantics when they are enabled
and disabled.

When OSHMPI is used within an SMP, we employ shared-memory 
windows to bypass MPI in Put, Get and Atomic operations to use only
load-store instructions or GCC intrinsics.
However, for strided, we still use MPI within an SMP because the lead
developer is a lazy bum.

Future Work
===========

We look forward to patches contributing the following:

* Allow selection of tuning options at runtime.
* Eliminate all intranode MPI-RMA communication.
* Cache subcommunicators corresponding to PE subgroups.

Bugs/Omissions
==============

We look forward to patches addressing the following:

* Mac non-heap symmetric data cannot be accessed remotely reliably.
* PSHMEM interface
* Fortran interface
* OpenSHMEM 1.3 changes
