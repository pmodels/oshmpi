#! /usr/bin/env bash
#
# (C) 2018 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.

##########################################
## Generic Utility Functions
##########################################
ROOTDIR=$(pwd)

echo_n() {
    # "echo_n" isn't portable, must portably implement with printf
    printf "%s" "$*"
}

check_autotools_version()
{
    tool=$1
    req_ver=$2
    curr_ver=$($tool --version | head -1 | cut -f4 -d' ' | xargs echo -n)
    if [ "$curr_ver" != "$req_ver" ]; then
        echo ""
        echo "$tool version mismatch ($req_ver) required"
        exit 1
    fi
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
check_autotools_version autoconf 2.69
echo "done"

echo_n "Checking for automake version..."
check_autotools_version automake 1.15.1
echo "done"

echo_n "Checking for libtool version..."
check_autotools_version libtool 2.4.6
echo "done"
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

echo "Generating Collective reduction typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/reduce_maxmin_typedef.txt \
    --tplfile ./include/shmem_reduce_minmax_typed.h.tpl --outfile ./include/shmem_reduce_minmax_typed.h
insert_file_by_key "SHMEM_REDUCE_MINMAX_TYPED_H start" ./include/shmem_reduce_minmax_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_MINMAX_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_typedef.txt \
    --tplfile ./include/shmem_reduce_sumprod_typed.h.tpl --outfile ./include/shmem_reduce_sumprod_typed.h
insert_file_by_key "SHMEM_REDUCE_SUMPROD_TYPED_H start" ./include/shmem_reduce_sumprod_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_SUMPROD_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_typedef.txt \
    --tplfile ./include/shmem_reduce_bitws_typed.h.tpl --outfile ./include/shmem_reduce_bitws_typed.h
insert_file_by_key "SHMEM_REDUCE_BITWS_TYPED_H start" ./include/shmem_reduce_bitws_typed.h include/shmem.h.in
echo "-- inserted SHMEM_REDUCE_BITWS_TYPED_H in include/shmem.h.in"
echo ""

echo "Generating Point-To-Point synchronization typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/p2p_typedef.txt \
    --tplfile ./include/shmem_p2p_typed.h.tpl --outfile ./include/shmem_p2p_typed.h
insert_file_by_key "SHMEM_P2P_TYPED_H start" ./include/shmem_p2p_typed.h include/shmem.h.in
echo "-- inserted SHMEM_P2P_TYPED_H in include/shmem.h.in"
echo ""

echo "Generating RMA APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/rma_typedef.txt \
    --tplfile ./src/shmem/rma_typed.c.tpl --outfile ./src/shmem/rma_typed.c
echo "-- ./src/shmem/rma_typed.c done"

./maint/build_sized_api.pl --sizefile ./maint/rma_sizedef.txt \
    --tplfile ./src/shmem/rma_sized.c.tpl --outfile ./src/shmem/rma_sized.c
echo "-- ./src/shmem/rma_sized.c done"
echo ""

echo "Generating AMO sized APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/amo_std_typedef.txt \
    --tplfile ./src/shmem/amo_std_typed.c.tpl --outfile ./src/shmem/amo_std_typed.c
echo "-- ./src/shmem/amo_std_typed.c done"

./maint/build_typed_api.pl --typefile ./maint/amo_ext_typedef.txt \
    --tplfile ./src/shmem/amo_ext_typed.c.tpl --outfile ./src/shmem/amo_ext_typed.c
echo "-- ./src/shmem/amo_ext_typed.c done"

./maint/build_typed_api.pl --typefile ./maint/amo_bitws_typedef.txt \
    --tplfile ./src/shmem/amo_bitws_typed.c.tpl --outfile ./src/shmem/amo_bitws_typed.c
echo "-- ./src/shmem/amo_bitws_typed.c done"
echo ""

echo "Generating Collective reduction typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/reduce_maxmin_typedef.txt \
    --tplfile ./src/shmem/reduce_minmax_typed.c.tpl --outfile ./src/shmem/reduce_minmax_typed.c
echo "-- ./src/shmem/reduce_minmax_typed.c done"

./maint/build_typed_api.pl --typefile ./maint/reduce_sumprod_typedef.txt \
    --tplfile ./src/shmem/reduce_sumprod_typed.c.tpl --outfile ./src/shmem/reduce_sumprod_typed.c
echo "-- ./src/shmem/reduce_sumprod_typed.c done"

./maint/build_typed_api.pl --typefile ./maint/reduce_bitws_typedef.txt \
    --tplfile ./src/shmem/reduce_bitws_typed.c.tpl --outfile ./src/shmem/reduce_bitws_typed.c
echo "-- ./src/shmem/reduce_bitws_typed.c done"
echo ""

echo "Generating Point-To-Point synchronization typed APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/p2p_typedef.txt \
    --tplfile ./src/shmem/p2p_typed.c.tpl --outfile ./src/shmem/p2p_typed.c
echo "-- ./src/shmem/p2p_typed.c done"
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


