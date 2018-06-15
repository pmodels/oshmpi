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

# Checks for libraries.
AC_CHECK_LIB([mpi], [MPI_Win_allocate_shared], have_mpi3=yes, AC_MSG_ERROR([MPI-3 is required.]))

# Checks for library functions.
AC_CHECK_FUNCS([memset memcpy atol atexit])

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

# Check for enabling strict
PAC_ARG_STRICT

## Debugging support
AC_ARG_ENABLE(g, AC_HELP_STRING([--enable-g],[Turn on debugging [default=no]]),
                                [ debug=$enableval ],
                                [ debug=no ])
AC_MSG_CHECKING(debugging support)
AC_MSG_RESULT($debug)
if test "$debug" = "yes"; then
  CFLAGS="$CFLAGS -g -O0"
fi

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

AC_CONFIG_FILES([include/shmem.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS([tests])
AC_OUTPUT
