#!/bin/sh

#set -e
set -x

os=`uname`
TOP="$1"

      MAKE_JNUM=2
      LIBTOOL_VERSION=2.4.6
      AUTOCONF_VERSION=2.69
      AUTOMAKE_VERSION=1.15.1

      mkdir -p ${TOP}

      cd ${TOP}
      TOOL=libtool
      TDIR=${TOOL}-${LIBTOOL_VERSION}
      FILE=${TDIR}.tar.gz
      BIN=${TOP}/bin/${TOOL}
      if [ ! -f ${FILE} ] ; then
        wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
      else
        echo ${FILE} already exists! Using existing copy.
      fi
      if [ ! -d ${TDIR} ] ; then
        echo Unpacking ${FILE}
        tar -xzf ${FILE}
      else
        echo ${TDIR} already exists! Using existing copy.
      fi
      if [ -f ${BIN} ] ; then
        echo ${BIN} already exists! Skipping build.
      else
        cd ${TOP}/${TDIR}
        ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
        if [ "x$?" != "x0" ] ; then
          echo FAILURE 2
          exit
        fi
      fi

      cd ${TOP}
      TOOL=autoconf
      TDIR=${TOOL}-${AUTOCONF_VERSION}
      FILE=${TDIR}.tar.gz
      BIN=${TOP}/bin/${TOOL}
      if [ ! -f ${FILE} ] ; then
        wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
      else
        echo ${FILE} already exists! Using existing copy.
      fi
      if [ ! -d ${TDIR} ] ; then
        echo Unpacking ${FILE}
        tar -xzf ${FILE}
      else
        echo ${TDIR} already exists! Using existing copy.
      fi
      if [ -f ${BIN} ] ; then
        echo ${BIN} already exists! Skipping build.
      else
        cd ${TOP}/${TDIR}
        ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
        if [ "x$?" != "x0" ] ; then
          echo FAILURE 3
          exit
        fi
      fi

      cd ${TOP}
      TOOL=automake
      TDIR=${TOOL}-${AUTOMAKE_VERSION}
      FILE=${TDIR}.tar.gz
      BIN=${TOP}/bin/${TOOL}
      if [ ! -f ${FILE} ] ; then
        wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
      else
        echo ${FILE} already exists! Using existing copy.
      fi
      if [ ! -d ${TDIR} ] ; then
        echo Unpacking ${FILE}
        tar -xzf ${FILE}
        else
          echo ${TDIR} already exists! Using existing copy.
        fi
        if [ -f ${BIN} ] ; then
          echo ${BIN} already exists! Skipping build.
        else
          cd ${TOP}/${TDIR}
          ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 4
            exit
          fi
        fi
