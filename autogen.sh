#!/bin/sh
libtoolize -c --force
aclocal
automake -a -c --foreign
autoconf
