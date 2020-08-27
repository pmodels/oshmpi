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

# suppress default "-g -O2" from AC_PROG_CXX
: ${CXXFLAGS=""}
# Bug in automake always check dependency even AC_PROG_CXX is conditionally checked.
AC_PROG_CXX(mpicxx)
AM_PROG_AR

LT_PREREQ([2.2.6])

# Set the appropriate macro for different platform
AS_CASE([$host_os],
    [darwin*],
        [AC_DEFINE(USE_OSX,1,[Define on Apple OSX.])],
    [linux*],
        [AC_DEFINE(USE_LINUX,1,[Define on Linux.])],
    [freebsd*],
        [AC_DEFINE(USE_FREEBSD,1,[Define on FreeBSD.])],
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

# Get top srcdir
# This test is complicated by the fact that top_srcdir is not set until
# the very end of configure.  Instead, we get it ourselves
if test -z "$top_srcdir" ; then
    oshmpi_top_srcdir=$srcdir   
else
    oshmpi_top_srcdir=$top_srcdir
fi
if test -z "$oshmpi_top_srcdir" ; then 
    # This needs to be an absolute pathname
    case "$oshmpi_top_srcdir" in
    /*) ;;
    *)
        oshmpi_top_srcdir=`(cd $oshmpi_top_srcdir && pwd)`
    ;;  
    esac
fi

# Get the directory that we're running in...
if test -z "$oshmpi_top_builddir" ; then
   oshmpi_top_builddir="`pwd`"
fi

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

AC_ARG_ENABLE(fast,
[  --enable-fast=[option]
                         Options to enable fast execution of OSHMPI implementation.
                         Supported options include (select multiple options separated by commas):
                            O<n>   - Appends optimization flag -O<n> to CFLAGS. (default -O2)
                            ndebug - Appends "-DOSHMPI_DISABLE_DEBUG -DOSHMPI_DISABLE_ERROR_CHECKING" to CFLAGS.
                                     It disables runtime control of debug message and disables all internal
                                     error checking.
                            ipo    - Appends "-DOSHMPI_ENABLE_IPO -ipo" to CFLAGS.
                            most|yes   - "O2" and "ndebug" are enabeld
                            all    - "most" and "ipo" are enabled
                            none   - None of above options, i.e. --disable-fast],
                          [ enable_fast=$enableval ],
                          [ enable_fast=O2 ])
AC_MSG_CHECKING(fast build support)
AC_MSG_RESULT($enable_fast)

enable_fast_opts=O2
enable_fast_ndebug=no
enable_fast_ipo=no
# strip off multiple options, separated by commas
save_IFS="$IFS"
IFS=","
for option in $enable_fast ; do
    case "$option" in
        O*)
        enable_fast_opts=$option
        ;;
        ndebug)
        enable_fast_ndebug=yes
        ;;
        ipo)
        enable_fast_ipo=yes
        ;;
        most|yes)
        enable_fast_ndebug=yes
        enable_fast_opts=O2
        ;;
        all)
        enable_fast_ndebug=yes
        enable_fast_opts=O2
        enable_fast_ipo=yes
        ;;
        none|no)
        enable_fast_ndebug=no
        enable_fast_opts=
        ;;
        *)
  IFS="$save_IFS"
        AC_MSG_WARN([Unknown value $option for --enable-fast])
  IFS=","
        ;;
    esac
done
IFS="$save_IFS"

if test -n "$enable_fast_opts" ; then
   # Allows O<n> where <n> can be [0-9] or ' '.
   opt_flags=`echo $enable_fast_opts | sed -e 's%\(O[0-9] \)%\1%g'`
   if test -n "$opt_flags" ; then
      PAC_APPEND_FLAG([-$enable_fast_opts -DOSHMPI_FAST_OPTS=$enable_fast_opts],[CFLAGS])
   else
      AC_MSG_WARN([Unknown value $enable_fast_opts for --enable-fast])
   fi
fi

if test "$enable_fast_ndebug" = "yes" ; then
    PAC_APPEND_FLAG([-DOSHMPI_DISABLE_DEBUG -DOSHMPI_DISABLE_ERROR_CHECKING],[CFLAGS])
fi

if test "$enable_fast_ipo" = "yes" ; then
    PAC_APPEND_FLAG([-DOSHMPI_ENABLE_IPO -ipo],[CFLAGS])
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
    PAC_APPEND_FLAG([-g -O0 -DOSHMPI_ENABLE_DBG],[CFLAGS])
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

# Check for C++ binding
AC_ARG_ENABLE(cxx, AC_HELP_STRING([--enable-cxx],
                 [Enable C++ bindings [default=yes]]),,
                 enable_cxx=yes)
if test $enable_cxx = "yes" ; then
    AC_MSG_CHECKING(if mpicxx is provided)
    AC_LANG_PUSH([C++])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
      #include <mpi.h>
    ])],
    AC_MSG_RESULT(yes),
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([mpicxx is required.]))
    AC_LANG_POP([C++])
    
    AC_DEFINE(OSHMPI_HAVE_CXX,1,[Define if C++ binding is enabled])
fi
AM_CONDITIONAL([OSHMPI_HAVE_CXX], [test "$enable_cxx" == "yes" ])

# Thread-safety
AC_ARG_ENABLE(threads, 
[  --enable-threads=[option]
                         Choose OSHMPI supported thread safety level. Set multiple by default.
                         For program uses single/funneled/serialized thread levels, it is 
                         recommanded to set lower thread-safety to avoid internal overhead
                         when possible. At runtime, the default thread safety is single.
                         The user should request any higher thread safety at shmem_init_thread.
                         The request thread safety cannot be higher than the one set at
                         configure. Supported options include:
                           single          - No threads (SHMEM_THREAD_SINGLE)
                           funneled        - Only the main thread calls SHMEM (SHMEM_THREAD_FUNNELED)
                           serialized      - User serializes calls to SHMEM (SHMEM_THREAD_SERIALIZED)
                           multiple        - Fully multi-threaded (SHMEM_THREAD_MULTIPLE)],
                         [ enable_threads=$enableval ],
                         [ enable_threads=multiple ])
AC_MSG_CHECKING(Thread safety support)
AC_MSG_RESULT($enable_threads)
case $enable_threads in
    single)
        AC_DEFINE(OSHMPI_ENABLE_THREAD_SINGLE,1,[Define if single thread safety is set at configure])
    ;;
    funneled)
        AC_DEFINE(OSHMPI_ENABLE_THREAD_FUNNELED,1,[Define if funneled thread safety is set at configure])
    ;;
    serialized)
        AC_DEFINE(OSHMPI_ENABLE_THREAD_SERIALIZED,1,[Define if serialized thread safety is set at configure])
    ;;
    multiple)
        AC_DEFINE(OSHMPI_ENABLE_THREAD_MULTIPLE,1,[Define if multiple thread safety is set at configure])
    ;;
    *)
        AC_MSG_ERROR([Unknown thread safety, $enable_threads.])
    ;;
esac

## Check for AMO method
AC_ARG_ENABLE(amo, 
[  --enable-amo=[option]
                         Choose AMO implementing methods.
                         Because MPI-3 guanrantees atomicity only for same_op_no_op or same_op
                         accumulates, we cannot directly use MPI accumulates in OpenSHMEM AMO 
                         which can be any_op. Instead, we have to fallback all AMOs to active
                         message in OSHMPI. However, OSHMPI accepts special user hints which provide
                         same restriction to same_op_no_op or same_op (see OSHMPI_AMO_OPS in README),
                         hence direct AMO can be enabled at runtime.
                         Supported options include:
                           auto - runtime chooses direct or am methods based on user hints (default)
                           direct - force direct amo where all atomics uses MPI accumulates
                           am - force active message based amo.],
                         [ enable_amo=$enableval ],
                         [ enable_amo=auto ])
AC_MSG_CHECKING(atomics method)
AC_MSG_RESULT($enable_amo)
case $enable_amo in
    direct)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_DIRECT_AMO,1,[Force to use direct atomics no matter whether MPI supports it])
    ;;
    am)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_AM_AMO,1,[Force to use active message based atomics])
    ;;
    auto)
    ;;
    *)
        AC_MSG_ERROR([Unknown AMO implementation method, $enable_amo.])
    ;;
esac

## Check for RMA method
AC_ARG_ENABLE(rma, 
[  --enable-rma=[option]
                         Choose RMA implementing methods.
                         To fully support GPU based RMA, we can directly use MPI RMA in
                         OSHMPI RMA only when the MPI implementations supports GPU PUT/GET.
                         Otherwise, we fallback the unsupported RMA to active message in OSHMPI
                         (require GPU PT2PT support in MPI). The user can pass the GPU features
                         of the MPI implementation by setting OSHMPI_MPI_GPU_FEATURES at runtime.
                         Supported options include:
                           auto - runtime chooses direct or am methods based on user hints (default)
                           direct - force direct rma where all rma uses MPI rma 
                           am - force active message based rma.],
                         [ enable_rma=$enableval ],
                         [ enable_rma=auto ])
AC_MSG_CHECKING(rma method)
AC_MSG_RESULT($enable_rma)
case $enable_rma in
    direct)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_DIRECT_RMA,1,[Force to use direct RMA no matter whether MPI supports it])
    ;;
    am)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_AM_RMA,1,[Force to use active message based RMA])
    ;;
    auto)
    ;;
    *)
        AC_MSG_ERROR([Unknown RMA implementation method, $enable_rma.])
    ;;
esac

## Check operation tracking optimization
AC_ARG_ENABLE(op-tracking, 
[  --enable-op-tracking=[option]
                         Enable tracking for outstanding operations. 
                         Supported options include:
                           yes - enable tracking (default)
                           no -  disable tracking],
                         [ enable_op_tracking=$enableval ],
                         [ enable_op_tracking=yes ])
AC_MSG_CHECKING(outstanding op tracking support)
AC_MSG_RESULT($enable_op_tracking)
if test "$enable_op_tracking" == "yes" ; then
    AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_OP_TRACKING,1,[Enable tracking for outstanding operations])
fi

## Check for strided datatype cache
AC_ARG_ENABLE(strided-cache, 
[  --enable-strided-cache=[option]
                         Enable datatype cache for strided data access.
                         Supported options include:
                           yes - enable (default)
                           no -  disable],
                         [ enable_strided_cache=$enableval ],
                         [ enable_strided_cache=yes ])
AC_MSG_CHECKING(strided datatype cache)
AC_MSG_RESULT($enable_strided_cache)
if test "$enable_strided_cache" == "yes" ; then
    AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_STRIDED_DTYPE_CACHE,1,[Enable datatype cache for strided data access])
fi

## Async thread support
AC_ARG_ENABLE(async-thread, 
[  --enable-async-thread=[option]
                         Enable asynchronous progress thread. 
                         Supported options include:
                           yes - always enable asynchronous thread
                           no -  always disable asynchronous thread
                           auto - runtime enables when either RMA or AMO is AM-based
                                  (see --enable-amo and --enable-rma). User can overwrite it
                                  by setting envioronment variable OSHMPI_ENABLE_ASYNC_THREAD (default).],
                         [ enable_async_thread=$enableval ],
                         [ enable_async_thread=auto ])
AC_MSG_CHECKING(asynchronous thread support)
AC_MSG_RESULT($enable_async_thread)

if test "$enable_async_thread" == "no" ; then
    AC_DEFINE_UNQUOTED(OSHMPI_DISABLE_AM_ASYNC_THREAD,1,[Always disable asynchronous progress thread])
else
    if test "$enable_threads" == "multiple"; then
      case $enable_async_thread in
          yes)
              AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_AM_ASYNC_THREAD,1,[Always enable asynchronous progress thread])
          ;;
          auto)
          ;;
          *)
          AC_MSG_ERROR([Unknown asynchronous progress thread option, $enable_async_thread.])
          ;;
      esac
    else
      AC_MSG_ERROR([To enable asynchronous thread, multiple thread safety is required, but $enable_threads is set.
                    Set --enable-async-thread=no to disable asynchronous thread if it is not required (i.e., no
                    active message based operation in OSHMPI).])
    fi
fi

# OPA library used for thread atomic operations in multitheaded program
opasrcdir=""
AC_SUBST([opasrcdir])
opalibdir=""
AC_SUBST([opalibdir])
opalib=""
AC_SUBST([opalib])
if test "$enable_threads" == "multiple"; then
    if test -e "${oshmpi_top_srcdir}/src/openpa" ; then
        opasrcdir="src/openpa"
        # only need header files
        PAC_APPEND_FLAG([-I${oshmpi_top_srcdir}/${opasrcdir}/src],[CPPFLAGS])
        PAC_APPEND_FLAG([-I${oshmpi_top_builddir}/${opasrcdir}/src],[CPPFLAGS])

        # no need for libtool versioning when embedding OPA
        opa_subdir_args="--disable-versioning --enable-embedded"
        PAC_CONFIG_SUBDIR_ARGS([$opasrcdir],[$opa_subdir_args],[],[AC_MSG_ERROR([OpenPA configure failed])])
    else
        AC_MSG_ERROR([unable to find embedded OpenPA source in "src/openpa".])
    fi 
fi

AM_CONDITIONAL([OSHMPI_HAVE_OPA], [test "$enable_threads" == "multiple" ])

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

if test "$enable_threads" == "multiple"; then
    THREAD_PACKAGE_NAME=NONE
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
            THREAD_PACKAGE_NAME=OSHMPI_THREAD_CS_LOCK__PTHREAD_MUTEX
        ;;
        *)
            AC_MSG_ERROR([The specified thread package, $with_thread_package, is not supported.])
        ;;
    esac
    AC_DEFINE_UNQUOTED(OSHMPI_THREAD_CS_LOCK,[$THREAD_PACKAGE_NAME],
                       [Thread package to implement critical section])
fi

## Check for internal window type
AC_ARG_ENABLE(win-type, 
[  --enable-win-type=[option]
                         Choose used MPI window type for access to symmetric segments.
                         The dynamic_win option creates a single window and attaches all symmetric data
                         segment to this window. The win_creates option creates separate window for each
                         segment. The former is more scalable in resource consumption but rely on non-standard
                         MPI hint "coll_attach" to gain performance. If the hint is not supported by the underlying
                         MPI, it is recommended to use the win_creates option.
                         Supported options include:
                           dynamic_win - use a single dynamic window for all symmetric data segments
                           win_creates - create separate window for each symmetric data segment (default).],
                         [ enable_win_type=$enableval ],
                         [ enable_win_type=win_creates ])
AC_MSG_CHECKING(Window type)
AC_MSG_RESULT($enable_win_type)
case $enable_win_type in
    dynamic_win)
        AC_DEFINE_UNQUOTED(OSHMPI_ENABLE_DYNAMIC_WIN,1,[Use a single dynamic window for all symmetric data segments])
    ;;
    win_creates)
    ;;
    *)
        AC_MSG_ERROR([Unknown Window type, $enable_win_type.])
    ;;
esac

AC_CHECK_HEADERS(sys/mman.h sys/stat.h sys/time.h fcntl.h)
AC_CHECK_FUNC(mmap, [HAVE_MMAP=1], [HAVE_MMAP=0])

gl_FUNC_RANDOM_R
if test "$HAVE_RANDOM_R" = "1" -a "$HAVE_STRUCT_RANDOM_DATA" = "1" -a "$HAVE_MMAP" = "1"; then
    AC_DEFINE(OSHMPI_ENABLE_SYMM_ALLOC,1,[Define if we can enable symmatric allocatio])
    AC_MSG_NOTICE([Enable symmetric allocation])
else
    AC_MSG_NOTICE([Disable symmetric allocation])
fi

## GPU space support
AC_ARG_ENABLE(cuda, 
[  --enable-cuda=[option]
                          Enable CUDA memory kind in space allocation. CUDA support is required.
                          Supported options include:
                            yes - enable CUDA memory kind space.
                            no - disable CUDA memory kind space (default)],
                         [ enable_cuda=$enableval ],
                         [ enable_cuda=no ])
AC_MSG_CHECKING(CUDA support)
AC_MSG_RESULT($enable_cuda)

if test "$enable_cuda" = "yes" ; then
    # check if CUDA is supported 
    have_cuda=no
    PAC_SET_HEADER_LIB_PATH(cuda)
    PAC_PUSH_FLAG(LIBS)
    PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaMalloc],[have_cuda=yes],[have_cuda=no])
    if test "${have_cuda}" = "yes" ; then
        AS_IF([test -d "${with_cuda}/lib64"],
            [PAC_APPEND_FLAG([-L${with_cuda}/lib64 -Wl,-rpath -Wl,${with_cuda}/lib64],[WRAPPER_LDFLAGS])])
        AS_IF([test -d "${with_cuda}/lib"],
            [PAC_APPEND_FLAG([-L${with_cuda}/lib -Wl,-rpath -Wl,${with_cuda}/lib64],[WRAPPER_LDFLAGS])])
        PAC_APPEND_FLAG([-lcuda -lcudart],[WRAPPER_LIBS])

        AC_DEFINE(OSHMPI_ENABLE_CUDA, 1, [Enable CUDA memory kind])
    else
        enable_cuda=no
        AC_MSG_ERROR(['cuda_runtime_api.h or libcudart library not found.'])
    fi
    PAC_POP_FLAG(LIBS)
fi
AM_CONDITIONAL([OSHMPI_ENABLE_CUDA], [test "$enable_cuda" == "yes" ])

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

AC_SUBST(WRAPPER_LDFLAGS)
AC_SUBST(WRAPPER_LIBS)

AC_CONFIG_FILES([include/shmem.h
  include/shmemx.h
  include/shmemx.fh])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS([tests])
AC_OUTPUT
