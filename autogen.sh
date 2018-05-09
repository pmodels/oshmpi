#! /bin/sh
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
## Others
##########################################

subdirs=

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


