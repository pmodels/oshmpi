#! /usr/bin/env bash
#
# (C) 2018 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.

##########################################
## Generic Utility Functions
##########################################

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
        exit
    fi
}

replace_file_lines_by_key() {
    key=$1
    file=$2
    origfile=$3
    awk -v f=$file "//; /$key/{while(getline<f){print}};" $origfile >tmp
    awk "!/$key/" tmp > $origfile
    rm -f tmp
}

##########################################
## Autotools Version Check
##########################################

echo_n "Checking for autoconf version..."
check_autotools_version autoconf 2.69
echo "done"

echo_n "Checking for automake version..."
check_autotools_version automake 1.15
echo "done"

echo_n "Checking for libtool version..."
check_autotools_version libtool 2.4.6
echo "done"
echo ""


##########################################
## Automatically generate typed/sized APIs
##########################################

echo "Generating RMA APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/rma_typedef.txt \
    --tplfile ./include/shmem_rma_typed.h.tpl --outfile ./include/shmem_rma_typed.h
replace_file_lines_by_key "@SHMEM_RMA_TYPED_H@" ./include/shmem_rma_typed.h include/shmem.h.in
echo "-- replaced SHMEM_RMA_TYPED_H in include/shmem.h.in"

./maint/build_sized_api.pl --sizefile ./maint/rma_sizedef.txt \
    --tplfile ./include/shmem_rma_sized.h.tpl --outfile ./include/shmem_rma_sized.h
replace_file_lines_by_key "@SHMEM_RMA_SIZED_H@" ./include/shmem_rma_sized.h include/shmem.h.in
echo "-- replaced SHMEM_RMA_SIZED_H in include/shmem.h.in"
echo ""

echo "Generating AMO typed APIs header file..."
./maint/build_typed_api.pl --typefile ./maint/amo_std_typedef.txt \
    --tplfile ./include/shmem_amo_std_typed.h.tpl --outfile ./include/shmem_amo_std_typed.h
replace_file_lines_by_key "@SHMEM_AMO_STD_TYPED_H@" ./include/shmem_amo_std_typed.h include/shmem.h.in
echo "-- replaced SHMEM_AMO_STD_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/amo_ext_typedef.txt \
    --tplfile ./include/shmem_amo_ext_typed.h.tpl --outfile ./include/shmem_amo_ext_typed.h
replace_file_lines_by_key "@SHMEM_AMO_EXT_TYPED_H@" ./include/shmem_amo_ext_typed.h include/shmem.h.in
echo "-- replaced SHMEM_AMO_EXT_TYPED_H in include/shmem.h.in"

./maint/build_typed_api.pl --typefile ./maint/amo_bitws_typedef.txt \
    --tplfile ./include/shmem_amo_bitws_typed.h.tpl --outfile ./include/shmem_amo_bitws_typed.h
replace_file_lines_by_key "@SHMEM_AMO_BITWS_TYPED_H@" ./include/shmem_amo_bitws_typed.h include/shmem.h.in
echo "-- replaced SHMEM_AMO_BITWS_TYPED_H in include/shmem.h.in"
echo ""

# clean up header file after all template replacement
./maint/code-cleanup.sh ./include/shmem.h.in
echo "Header file ./include/shmem.h.in format cleaned"
echo ""

echo "Generating RMA APIs source files..."
./maint/build_typed_api.pl --typefile ./maint/rma_typedef.txt \
    --tplfile ./src/shmem/rma_typed.tpl --outfile ./src/shmem/rma_typed.c
echo "-- ./src/shmem/rma_typed.c done"
./maint/code-cleanup.sh ./src/shmem/rma_typed.c
echo "-- ./src/shmem/rma_typed.c format cleaned"

./maint/build_sized_api.pl --sizefile ./maint/rma_sizedef.txt \
    --tplfile ./src/shmem/rma_sized.tpl --outfile ./src/shmem/rma_sized.c
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

# generate configures
for subdir in . $subdirs ; do
    echo ""
    echo "=== Generating configure in $subdir ==="
    (cd $subdir && autoreconf -vif) || exit 1
    echo "done"
done


