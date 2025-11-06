#!/bin/bash
echo "configure..."
tar --strip-components=1 -xf  cadna_c_half-3.1.12.tar.gz
flag1=--enable-half-emulation
flag2=--disable-dependency-tracking

if [ $# -eq 2 ]
then
  echo CC=$1 CXX=$2 --prefix=`pwd` $flag1 $flag2
  ./configure CC=$1 CXX=$2 --prefix=`pwd` $flag1 $flag2
else
  echo CXX=g++ --prefix=`pwd` $flag1 $flag2
  ./configure CXX=g++ --prefix=`pwd` $flag1 $flag2
fi

echo "make install..."
make install