#!/bin/sh
libtoolize -c
aclocal
automake -a -c --foreign
autoconf
