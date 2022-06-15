#! /usr/bin/env bash
#
# (C) 2018 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.

##########################################
## Generic Utility Functions
##########################################
ROOTDIR=$(pwd)

recreate_tmp() {
    rm -rf .tmp
    mkdir .tmp 2>&1 >/dev/null
}

echo_n() {
    # "echo_n" isn't portable, must portably implement with printf
    printf "%s" "$*"
}

insert_file_by_key() {
    key=$1
    file=$2
    origfile=$3
    awk -v f=$file "//; /$key/{while(getline<f){print}};" $origfile >tmp
    mv tmp $origfile
}

# checking submodules
check_submodule_presence() {
    if test ! -f "$ROOTDIR/$1/configure.ac"; then
        echo "Submodule $1 is not checked out"
        exit 1
    fi
}

##########################################
## Autotools Version Check
##########################################

echo_n "Checking for autoconf version..."
recreate_tmp
ver=2.69
cat > .tmp/configure.ac<<EOF
AC_INIT
AC_PREREQ($ver)
AC_OUTPUT
EOF
if (cd .tmp && autoreconf -vif >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "autoconf version mismatch ($ver) required"
    exit 1
fi


echo_n "Checking for automake version..."
recreate_tmp
ver=1.15
cat > .tmp/configure.ac<<EOF
AC_INIT(testver,1.0)
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])
m4_ifdef([AM_INIT_AUTOMAKE],,[m4_fatal([AM_INIT_AUTOMAKE not defined])])
AM_INIT_AUTOMAKE([$ver foreign])
AC_MSG_RESULT([A message])
AC_OUTPUT([Makefile])
EOF
cat <<EOF >.tmp/Makefile.am
ACLOCAL_AMFLAGS = -I m4
EOF
if [ ! -d .tmp/m4 ] ; then mkdir .tmp/m4 >/dev/null 2>&1 ; fi
if (cd .tmp && autoreconf -vif >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "automake version mismatch ($ver) required"
    exit 1
fi

echo_n "Checking for libtool version..."
recreate_tmp
ver=2.4.6
cat <<EOF >.tmp/configure.ac
AC_INIT(testver,1.0)
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])
m4_ifdef([LT_PREREQ],,[m4_fatal([LT_PREREQ not defined])])
LT_PREREQ($ver)
LT_INIT()
AC_MSG_RESULT([A message])
EOF
cat <<EOF >.tmp/Makefile.am
ACLOCAL_AMFLAGS = -I m4
EOF
if [ ! -d .tmp/m4 ] ; then mkdir .tmp/m4 >/dev/null 2>&1 ; fi
if (cd .tmp && autoreconf -vif >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "libtool version mismatch ($ver) required"
    exit 1
fi
echo ""


##########################################
## Automatically generate typed/sized APIs
##########################################

echo "Generating initial header file ./include/shmem.h.in"
cp ./include/shmem.h.in.tpl ./include/shmem.h.in
echo "Initial header file ./include/shmem.h.in generated"
echo ""

echo "Generating RMA APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/rma_typedef.txt \
    --tplfile ./include/shmem_rma_typed.h.tpl --outfile ./include/shmem_rma_typed.h
insert_file_by_key "SHMEM_RMA_TYPED_H start" ./include/shmem_rma_typed.h include/shmem.h.in
echo "-- replaced SHMEM_RMA_TYPED_H in include/shmem.h.in"

./maint/build_sized_api.pl --sizefile ./maint/rma_sizedef.txt \
    --tplfile ./include/shmem_rma_sized.h.tpl --outfile ./include/shmem_rma_sized.h
insert_file_by_key "SHMEM_RMA_SIZED_H start" ./include/shmem_rma_sized.h include/shmem.h.in
echo "-- inserted SHMEM_RMA_SIZED_H in include/shmem.h.in"
echo ""

echo "Generating AMO typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/amo_std_typedef.txt \
    --tplfile ./include/shmem_amo_std_typed.h.tpl --outfile ./include/shmem_amo_std_typed.h
insert_file_by_key "SHMEM_AMO_STD_TYPED_H start" ./include/shmem_amo_std_typed.h include/shmem.h.in
echo "-- inserted SHMEM_AMO_STD_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/amo_ext_typedef.txt \
    --tplfile ./include/shmem_amo_ext_typed.h.tpl --outfile ./include/shmem_amo_ext_typed.h
insert_file_by_key "SHMEM_AMO_EXT_TYPED_H start" ./include/shmem_amo_ext_typed.h include/shmem.h.in
echo "-- inserted SHMEM_AMO_EXT_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/amo_bitws_typedef.txt \
    --tplfile ./include/shmem_amo_bitws_typed.h.tpl --outfile ./include/shmem_amo_bitws_typed.h
insert_file_by_key "SHMEM_AMO_BITWS_TYPED_H start" ./include/shmem_amo_bitws_typed.h include/shmem.h.in
echo "-- inserted SHMEM_AMO_BITWS_TYPED_H in include/shmem.h.in"
echo ""

echo "Generating Collective typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/coll_typedef.txt \
    --tplfile ./include/shmem_coll_typed.h.tpl --outfile ./include/shmem_coll_typed.h
insert_file_by_key "SHMEM_COLL_TYPED_H start" ./include/shmem_coll_typed.h include/shmem.h.in
echo "-- inserted SHMEM_COLL_TYPED_H in include/shmem.h.in"

echo "Generating Collective reduction active-set-based typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/reduce_minmax_aset_typedef.txt \
    --tplfile ./include/shmem_reduce_minmax_aset_typed.h.tpl --outfile ./include/shmem_reduce_minmax_aset_typed.h
insert_file_by_key "SHMEM_REDUCE_MINMAX_ASET_TYPED_H start" ./include/shmem_reduce_minmax_aset_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_MINMAX_ASET_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_aset_typedef.txt \
    --tplfile ./include/shmem_reduce_sumprod_aset_typed.h.tpl --outfile ./include/shmem_reduce_sumprod_aset_typed.h
insert_file_by_key "SHMEM_REDUCE_SUMPROD_ASET_TYPED_H start" ./include/shmem_reduce_sumprod_aset_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_SUMPROD_ASET_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_aset_typedef.txt \
    --tplfile ./include/shmem_reduce_bitws_aset_typed.h.tpl --outfile ./include/shmem_reduce_bitws_aset_typed.h
insert_file_by_key "SHMEM_REDUCE_BITWS_ASET_TYPED_H start" ./include/shmem_reduce_bitws_aset_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_BITWS_ASET_TYPED_H in include/shmem.h.in"
echo ""

echo "Generating Collective reduction active-set-based typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/reduce_minmax_team_typedef.txt \
    --tplfile ./include/shmem_reduce_minmax_team_typed.h.tpl --outfile ./include/shmem_reduce_minmax_team_typed.h
insert_file_by_key "SHMEM_REDUCE_MINMAX_TEAM_TYPED_H start" ./include/shmem_reduce_minmax_team_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_MINMAX_TEAM_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_team_typedef.txt \
    --tplfile ./include/shmem_reduce_sumprod_team_typed.h.tpl --outfile ./include/shmem_reduce_sumprod_team_typed.h
insert_file_by_key "SHMEM_REDUCE_SUMPROD_TEAM_TYPED_H start" ./include/shmem_reduce_sumprod_team_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_SUMPROD_TEAM_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_team_typedef.txt \
    --tplfile ./include/shmem_reduce_bitws_team_typed.h.tpl --outfile ./include/shmem_reduce_bitws_team_typed.h
insert_file_by_key "SHMEM_REDUCE_BITWS_TEAM_TYPED_H start" ./include/shmem_reduce_bitws_team_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_BITWS_TEAM_TYPED_H in include/shmem.h.in"
echo ""

echo "Generating Signal APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/signal_typedef.txt \
    --tplfile ./include/shmem_signal_typed.h.tpl --outfile ./include/shmem_signal_typed.h
insert_file_by_key "SHMEM_SIGNAL_TYPED_H start" ./include/shmem_signal_typed.h include/shmem.h.in
echo "-- replaced SHMEM_SIGNAL_TYPED_H in include/shmem.h.in"

./maint/build_sized_api.pl --sizefile ./maint/signal_sizedef.txt \
    --tplfile ./include/shmem_signal_sized.h.tpl --outfile ./include/shmem_signal_sized.h
insert_file_by_key "SHMEM_SIGNAL_SIZED_H start" ./include/shmem_signal_sized.h include/shmem.h.in
echo "-- inserted SHMEM_SIGNAL_SIZED_H in include/shmem.h.in"
echo ""

echo "Generating Point-To-Point synchronization typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/p2p_typedef.txt \
    --tplfile ./include/shmem_p2p_typed.h.tpl --outfile ./include/shmem_p2p_typed.h
insert_file_by_key "SHMEM_P2P_TYPED_H start" ./include/shmem_p2p_typed.h include/shmem.h.in
echo "-- inserted SHMEM_P2P_TYPED_H in include/shmem.h.in"
echo ""

# clean up header file after all template replacement
./maint/code-cleanup.sh ./include/shmem.h.in
echo "Header file ./include/shmem.h.in format cleaned"
echo ""

echo "Generating RMA APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/rma_typedef.txt \
    --tplfile ./src/shmem/rma_typed.c.tpl --outfile ./src/shmem/rma_typed.c
echo "-- ./src/shmem/rma_typed.c done"
./maint/code-cleanup.sh ./src/shmem/rma_typed.c
echo "-- ./src/shmem/rma_typed.c format cleaned"

./maint/build_sized_api.pl --sizefile ./maint/rma_sizedef.txt \
    --tplfile ./src/shmem/rma_sized.c.tpl --outfile ./src/shmem/rma_sized.c
echo "-- ./src/shmem/rma_sized.c done"
./maint/code-cleanup.sh ./src/shmem/rma_sized.c
echo "-- ./src/shmem/rma_sized.c format cleaned"
echo ""

echo "Generating AMO sized APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/amo_std_typedef.txt \
    --tplfile ./src/shmem/amo_std_typed.c.tpl --outfile ./src/shmem/amo_std_typed.c
echo "-- ./src/shmem/amo_std_typed.c done"
./maint/code-cleanup.sh ./src/shmem/amo_std_typed.c
echo "-- ./src/shmem/amo_std_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/amo_ext_typedef.txt \
    --tplfile ./src/shmem/amo_ext_typed.c.tpl --outfile ./src/shmem/amo_ext_typed.c
echo "-- ./src/shmem/amo_ext_typed.c done"
./maint/code-cleanup.sh ./src/shmem/amo_ext_typed.c
echo "-- ./src/shmem/amo_ext_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/amo_bitws_typedef.txt \
    --tplfile ./src/shmem/amo_bitws_typed.c.tpl --outfile ./src/shmem/amo_bitws_typed.c
echo "-- ./src/shmem/amo_bitws_typed.c done"
./maint/code-cleanup.sh ./src/shmem/amo_bitws_typed.c
echo "-- ./src/shmem/amo_bitws_typed.c format cleaned"
echo ""

echo "Generating Collective typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/coll_typedef.txt \
    --tplfile ./src/shmem/coll_typed.c.tpl --outfile ./src/shmem/coll_typed.c
echo "-- ./src/shmem/coll_typed.c done"
./maint/code-cleanup.sh ./src/shmem/coll_typed.c
echo "-- ./src/shmem/coll_typed.c format cleaned"

echo "Generating Collective reduction active-set-based typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/reduce_minmax_aset_typedef.txt \
    --tplfile ./src/shmem/reduce_minmax_aset_typed.c.tpl --outfile ./src/shmem/reduce_minmax_aset_typed.c
echo "-- ./src/shmem/reduce_minmax_aset_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_minmax_aset_typed.c
echo "-- ./src/shmem/reduce_minmax_aset_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_aset_typedef.txt \
    --tplfile ./src/shmem/reduce_sumprod_aset_typed.c.tpl --outfile ./src/shmem/reduce_sumprod_aset_typed.c
echo "-- ./src/shmem/reduce_sumprod_aset_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_sumprod_aset_typed.c
echo "-- ./src/shmem/reduce_sumprod_aset_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_aset_typedef.txt \
    --tplfile ./src/shmem/reduce_bitws_aset_typed.c.tpl --outfile ./src/shmem/reduce_bitws_aset_typed.c
echo "-- ./src/shmem/reduce_bitws_aset_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_bitws_aset_typed.c
echo "-- ./src/shmem/reduce_bitws_aset_typed.c format cleaned"
echo ""

echo "Generating Collective reduction team-based typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/reduce_minmax_team_typedef.txt \
    --tplfile ./src/shmem/reduce_minmax_team_typed.c.tpl --outfile ./src/shmem/reduce_minmax_team_typed.c
echo "-- ./src/shmem/reduce_minmax_team_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_minmax_team_typed.c
echo "-- ./src/shmem/reduce_minmax_team_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_team_typedef.txt \
    --tplfile ./src/shmem/reduce_sumprod_team_typed.c.tpl --outfile ./src/shmem/reduce_sumprod_team_typed.c
echo "-- ./src/shmem/reduce_sumprod_team_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_sumprod_team_typed.c
echo "-- ./src/shmem/reduce_sumprod_team_typed.c format cleaned"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_team_typedef.txt \
    --tplfile ./src/shmem/reduce_bitws_team_typed.c.tpl --outfile ./src/shmem/reduce_bitws_team_typed.c
echo "-- ./src/shmem/reduce_bitws_team_typed.c done"
./maint/code-cleanup.sh ./src/shmem/reduce_bitws_team_typed.c
echo "-- ./src/shmem/reduce_bitws_team_typed.c format cleaned"
echo ""

echo "Generating Signaling APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/signal_typedef.txt \
    --tplfile ./src/shmem/signal_typed.c.tpl --outfile ./src/shmem/signal_typed.c
echo "-- ./src/shmem/signal_typed.c done"
./maint/code-cleanup.sh ./src/shmem/signal_typed.c
echo "-- ./src/shmem/signal_typed.c format cleaned"

./maint/build_sized_api.pl --sizefile ./maint/signal_sizedef.txt \
    --tplfile ./src/shmem/signal_sized.c.tpl --outfile ./src/shmem/signal_sized.c
echo "-- ./src/shmem/signal_sized.c done"
./maint/code-cleanup.sh ./src/shmem/signal_sized.c
echo "-- ./src/shmem/signal_sized.c format cleaned"
echo ""

echo "Generating Point-To-Point synchronization typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/p2p_typedef.txt \
    --tplfile ./src/shmem/p2p_typed.c.tpl --outfile ./src/shmem/p2p_typed.c
echo "-- ./src/shmem/p2p_typed.c done"
./maint/code-cleanup.sh ./src/shmem/p2p_typed.c
echo "-- ./src/shmem/p2p_typed.c format cleaned"
echo ""

##########################################
## Others
##########################################

subdirs=tests

# workaround empty include directory which is required when generate configure
for subdir in $subdirs ; do
    if [ ! -d $subdir/include ]; then
        mkdir -p $subdir/include
    fi
done

# copy confdb
echo ""
for subdir in $subdirs ; do
    subconfdb_dir=$subdir/confdb
    echo_n "Syncronizing confdb -> $subconfdb_dir... "
    if [ -x $subconfdb_dir ] ; then
        rm -rf "$subconfdb_dir"
    fi
    cp -pPR confdb "$subconfdb_dir"
    echo "done"
done

# autogen for submodules
extdirs="src/openpa"

for extdir in $extdirs ; do
    echo ""
    echo "=== Running third-party initialization in $extdir ==="
    check_submodule_presence $extdir
    if [ -d "$extdir" -o -L "$extdir" ] ; then
      (cd $extdir && ./autogen.sh) || exit 1
      echo "done"
    fi
done

# generate configures
for subdir in . $subdirs ; do
    echo ""
    echo "=== Generating configure in $subdir ==="
    (cd $subdir && autoreconf -vif) || exit 1
    echo "done"
done


