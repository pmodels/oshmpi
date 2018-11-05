#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

########################################
# Generate releaes version
########################################

m4_include([maint/version.m4])
dnl 2nd arg is intentionally underquoted
AC_INIT([OSHMPI],
        OSHMPI_VERSION_m4,
        [oshmpi-users@lists.mpich.org],
        [oshmpi],
        [https://github.com/pmodels/oshmpi])

# build info
CONFIGURE_ARGS_CLEAN=`echo $* | tr '"' ' '`

# these values come from the m4_include above
OSHMPI_VERSION=OSHMPI_VERSION_m4
AC_SUBST(OSHMPI_VERSION)
OSHMPI_RELEASE_DATE="OSHMPI_RELEASE_DATE_m4"
AC_SUBST(OSHMPI_RELEASE_DATE)

if test -z "$OSHMPI_VERSION" ; then
    AC_MSG_ERROR([OSHMPI_VERSION is empty, check maint/version.m4 for errors])
fi

# Produce a numeric version assuming the following format:
# Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
# Example: 1.0.7rc1 has
#          MAJ = 1
#          MIN = 0
#          REV = 7
#          EXT = rc
#          EXT_NUMBER = 1
#
# Converting to numeric version will convert EXT to a format number:
#          ALPHA (a) = 0
#          BETA (b)  = 1
#          RC (rc)   = 2
#          PATCH (p) = 3
# Regular releases are treated as patch 0
#
# Numeric version will have 1 digit for MAJ, 2 digits for MIN,
# 2 digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER.
changequote(<<,>>)
V1=`expr $OSHMPI_VERSION : '\([0-9]*\)\.[0-9]*\.*[0-9]*[a-zA-Z]*[0-9]*'`
V2=`expr $OSHMPI_VERSION : '[0-9]*\.\([0-9]*\)\.*[0-9]*[a-zA-Z]*[0-9]*'`
V3=`expr $OSHMPI_VERSION : '[0-9]*\.[0-9]*\.*\([0-9]*\)[a-zA-Z]*[0-9]*'`
V4=`expr $OSHMPI_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*\([a-zA-Z]*\)[0-9]*'`
V5=`expr $OSHMPI_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*[a-zA-Z]*\([0-9]*\)'`
changequote([,])

if test "$V2" -le 9 ; then V2=0$V2 ; fi
if test "$V3" = "" ; then V3=0; fi
if test "$V3" -le 9 ; then V3=0$V3 ; fi
if test "$V4" = "a" ; then
    V4=0
elif test "$V4" = "b" ; then
    V4=1
elif test "$V4" = "rc" ; then
    V4=2
elif test "$V4" = "" ; then
    V4=3
    V5=0
elif test "$V4" = "p" ; then
    V4=3
fi
if test "$V5" -le 9 ; then V5=0$V5 ; fi

OSHMPI_NUMVERSION=`expr $V1$V2$V3$V4$V5 + 0`
AC_SUBST(OSHMPI_NUMVERSION)


########################################
# Check and configure setup
########################################

AC_CONFIG_AUX_DIR(confdb)
AC_CONFIG_MACRO_DIR(confdb)

# needed by hwloc in embedded mode.  Must come very early to avoid
# bizarre expansion ordering warnings
AC_CANONICAL_TARGET
AC_ARG_PROGRAM

# also needed by hwloc in embedded mode, must also come early for expansion
# ordering reasons
AC_USE_SYSTEM_EXTENSIONS

AM_INIT_AUTOMAKE([-Wall -Werror -Wno-portability-recursive silent-rules foreign 1.12.3 subdir-objects])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
AC_PROG_CC(mpicc)
PAC_POP_FLAG([CFLAGS])

AM_PROG_AR

LT_PREREQ([2.2.6])

# Set the appropriate macro for different platform
AS_CASE([$host_os],
    [darwin*],
        [AC_DEFINE(USE_MAC,1,[Define on Apple OSX.])],
    [linux*],
        [AC_DEFINE(USE_LINUX,1,[Define on Linux.])],
    [AC_MSG_ERROR([Unsupported operating system.])]
)

AC_CONFIG_HEADER([include/oshmpiconf.h])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
LT_INIT()
PAC_POP_FLAG([CFLAGS])
 
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN CC=$CC CFLAGS=$CFLAGS "
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN LDFLAGS=$LDFLAGS LIBS=$LIBS CPPFLAGS=$CPPFLAGS  "
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN $BUILD_INFO LT_SYS_LIBRARY_PATH=$LT_SYS_LIBRARY_PATH CPP=$CPP"
AC_SUBST(CONFIGURE_ARGS_CLEAN)

## Check for C99
AC_PROG_CC_C99
if test "$ac_cv_prog_cc_c99" = "no" ; then
  AC_ERROR([C99 not supported by the compiler])
fi

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h mpi.h])

# Checks for library functions.
AC_CHECK_FUNCS([memset memcpy atol atexit])

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

# Check MPI version support
AC_DEFUN([UD_CHECK_MPI_VERSION], [
  AC_MSG_CHECKING(if MPI_VERSION=$1 defined)
  AC_RUN_IFELSE(
    [AC_LANG_SOURCE([
    #include <stdlib.h>
    #include "mpi.h"
    int main (int argc, char **argv) {
        if(MPI_VERSION != $1) exit (-1);
    }])],
  AC_MSG_RESULT(yes)
  have_mpi$1=yes,
  AC_MSG_RESULT(no)
  have_mpi$1=no
  AC_MSG_ERROR([MPI-$1 is required.]))
])

UD_CHECK_MPI_VERSION(3)
AC_CHECK_FUNCS([MPI_Ibarrier MPI_Ibcast MPI_Iallreduce MPI_Ialltoall MPI_Iallgather MPI_Iallgatherv])

## Fast build (statically disable debug messages, disable assert)
AC_ARG_ENABLE(fast, AC_HELP_STRING([--enable-fast],[Enable fast execution of OSHMPI implementation.
                                It disables runtime control of debug message and disables all internal
                                error checking [default=no]]),
                                [ enable_fast=$enableval ],
                                [ enable_fast=no ])
AC_MSG_CHECKING(fast build support)
AC_MSG_RESULT($enable_fast)
if test "$enable_fast" = "yes"; then
  CFLAGS="-O2 -DOSHMPI_ENABLE_FAST"  # Overwrite default -g -O2
fi

## Debugging support
AC_ARG_ENABLE(g, AC_HELP_STRING([--enable-g],[Turn on debugging [default=no]]),
                                [ enable_debug=$enableval ],
                                [ enable_debug=no ])
AC_MSG_CHECKING(debugging support)
AC_MSG_RESULT($enable_debug)
# always disable debug if enable_fast is set
if test "$enable_fast" = "no"; then
  if test "$enable_debug" = "yes"; then
    CFLAGS="-g -O0 -DOSHMPI_ENABLE_DBG"  # Overwrite default -g -O2
  fi
fi

# Check for enabling strict
PAC_ARG_STRICT

# Check for support for weak symbols.
AC_ARG_ENABLE(weak-symbols, AC_HELP_STRING([--enable-weak-symbols],
                 [Use weak symbols to implement PSHMEM routines [default=yes]]),,
                 enable_weak_symbols=yes)
if test $enable_weak_symbols = yes ; then
    # Turn off weak symbols if they aren't available
    PAC_PROG_C_WEAK_SYMBOLS(,enable_weak_symbols=no)
fi
if test $enable_weak_symbols = "yes" ; then
    AC_DEFINE(USE_WEAK_SYMBOLS,1,[Define if weak symbols should be used])
    # Check for the ability to support multiple weak symbols
    if test "$pac_cv_prog_c_weak_symbols" = "pragma weak" ; then
       PAC_PROG_C_MULTIPLE_WEAK_SYMBOLS(AC_DEFINE(HAVE_MULTIPLE_PRAGMA_WEAK,1,[Define if multiple weak symbols may be defined]))
    fi
fi

## Direct AMO support
AC_ARG_ENABLE(direct-amo, 
[  --enable-direct-amo=option
                         Enable AMO implementation that directly uses MPI accumulates.
                         Because MPI-3 guanrantees atomicity only for same_op_no_op or same_op
                         accumulates, we cannot directly use MPI accumulates in OpenSHMEM AMO 
                         which can be any_op. Supported options include:
                         yes - always enable direct amo
                         no - disable direct amo (default)
                         runtime - enable when special info hints are accepted.],
                         [ enable_direct_amo=$enableval ],
                         [ enable_direct_amo=no ])
AC_MSG_CHECKING(direct atomics support)
AC_MSG_RESULT($enable_direct_amo)
case $enable_direct_amo in
    yes)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_DIRECT_AMO,1,[Enable direct atomics no matter whether MPI supports it])
    ;;
    runtime)
        AC_DEFINE_UNQUOTED(OSHMPI_RUNTIME_DIRECT_AMO,1,[Turn on runtime control of direct atomics])
    ;;
    *)
    ;;
esac

## Async thread support
AC_ARG_ENABLE(async-thread, 
[  --enable-async-thread=option
                         Enable asynchronous progress thread. 
                         Supported options include:
                         yes - always enable asynchronous thread
                         no - disable asynchronous thread (default)
                         runtime - enable by setting envioronment variable OSHMPI_ENABLE_ASYNC_THREAD.],
                         [ enable_async_thread=$enableval ],
                         [ enable_async_thread=no ])
AC_MSG_CHECKING(asynchronous thread support)
AC_MSG_RESULT($enable_async_thread)
if test "$enable_direct_amo" == "yes" ; then
    enable_async_thread=no
else
    # enable async thread only when am AMO may be used
    case $enable_async_thread in
        yes)
            AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_AMO_ASYNC_THREAD,1,[Enable asynchronous progress thread])
        ;;
        runtime)
            AC_DEFINE_UNQUOTED(OSHMPI_RUNTIME_AMO_ASYNC_THREAD,1,[Turn on runtime control of asynchronous progress thread])
        ;;
        *)
        ;;
    esac
fi

# Thread package used in critical section and asynchronous thread
AC_ARG_WITH([thread-package],
[  --with-thread-package=posix|pthread
                          Thread package to implement internal critical section 
                          when multiple threads are present. Supported thread 
                          packages include:
                          posix or pthreads - POSIX threads (default)
],,with_thread_package=default)
AC_MSG_CHECKING(Thread package to implement critical section)
AC_MSG_RESULT($with_thread_package)

if test "$enable_async_thread" != "no" ; then
    case $with_thread_package in
        default|posix|pthreads)
            # check if pthread is supported (e.g., invalid on windows or solaris)
            have_pthreads=no
            AC_CHECK_HEADERS(pthread.h)
            AC_CHECK_LIB([pthread],[pthread_create],have_lpthread=yes)
            if test "$have_lpthread" = "yes" ; then
                PAC_PREPEND_FLAG([-lpthread],[LIBS])
                # this check should come after the AC_CHECK_LIB for -lpthread
                AC_CHECK_FUNCS([pthread_mutex_lock],have_pthreads=yes,AC_MSG_ERROR([unable to find pthreads library.]))
            fi
        ;;
        *)
            AC_MSG_ERROR([The specified thread package, $with_thread_package, is not supported.])
        ;;
    esac
fi

# check for attribute support
PAC_C_GNU_ATTRIBUTE

# Define corresponding MPI datatypes
# size_t is a unsigned integer type
AC_CHECK_SIZEOF(size_t)
AC_CHECK_SIZEOF(uint8_t)
AC_CHECK_SIZEOF(uint16_t)
AC_CHECK_SIZEOF(uint32_t)
AC_CHECK_SIZEOF(uint64_t)
# ptrdiff_t is a signed integer type
AC_CHECK_SIZEOF(ptrdiff_t)
AC_CHECK_SIZEOF(int8_t)
AC_CHECK_SIZEOF(int16_t)
AC_CHECK_SIZEOF(int32_t)
AC_CHECK_SIZEOF(int64_t)

if test "$ac_cv_sizeof_size_t" = "$ac_cv_sizeof_uint8_t"; then
    OSHMPI_MPI_SIZE_T=MPI_UINT8_T
elif test "$ac_cv_sizeof_size_t" = "$ac_cv_sizeof_uint16_t"; then
    OSHMPI_MPI_SIZE_T=MPI_UINT16_T
elif test "$ac_cv_sizeof_size_t" = "$ac_cv_sizeof_uint32_t"; then
    OSHMPI_MPI_SIZE_T=MPI_UINT32_T
elif test "$ac_cv_sizeof_size_t" = "$ac_cv_sizeof_uint64_t"; then
    OSHMPI_MPI_SIZE_T=MPI_UINT64_T
else
    AC_MSG_ERROR([size of size_t is $ac_cv_sizeof_size_t, cannot find corresponding MPI datatype.])
fi
AC_DEFINE_UNQUOTED(OSHMPI_MPI_SIZE_T,$OSHMPI_MPI_SIZE_T,[MPI datatype to use for size_t])

if test "$ac_cv_sizeof_ptrdiff_t" = "$ac_cv_sizeof_int8_t"; then
    OSHMPI_MPI_PTRDIFF_T=MPI_INT8_T
elif test "$ac_cv_sizeof_ptrdiff_t" = "$ac_cv_sizeof_int16_t"; then
    OSHMPI_MPI_PTRDIFF_T=MPI_INT16_T
elif test "$ac_cv_sizeof_ptrdiff_t" = "$ac_cv_sizeof_int32_t"; then
    OSHMPI_MPI_PTRDIFF_T=MPI_INT32_T
elif test "$ac_cv_sizeof_ptrdiff_t" = "$ac_cv_sizeof_int64_t"; then
    OSHMPI_MPI_PTRDIFF_T=MPI_INT64_T
else
    AC_MSG_ERROR([size of ptrdiff_t is $ac_cv_sizeof_ptrdiff_t, cannot find corresponding MPI datatype.])
fi
AC_DEFINE_UNQUOTED(OSHMPI_MPI_PTRDIFF_T,$OSHMPI_MPI_PTRDIFF_T,[MPI datatype to use for ptrdiff_t])

AC_CONFIG_FILES([include/shmem.h
  include/shmemx.h
  include/shmemx.fh])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS([tests])
AC_OUTPUT
