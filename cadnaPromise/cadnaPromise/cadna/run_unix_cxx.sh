#!/bin/bash
echo "configure..."
tar --strip-components=1 -xf  cadna_c_half-3.1.12.tar.gz
./configure CXX=$1 --prefix=`pwd` --enable-half-emulation --disable-dependency-tracking
echo "make install.."
make install