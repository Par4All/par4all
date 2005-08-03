#!/bin/sh
libtoolize -c --force
aclocal -I m4
automake -a -c --foreign
autoconf
