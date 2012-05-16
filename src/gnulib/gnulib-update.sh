#!/bin/sh
# use this script to update gnulib
# because of naming convention limitations introduced by FC
# namely not to accept file names with special characters such as '+'.
# we cannot solely rely on a gnulib-tool --update

set -e
gnulib-tool --lgpl --import --dir=../.. --lib=libgnu --source-base=src/gnulib --m4-base=src/gnulib/m4 --doc-base=doc --tests-base=tests --aux-dir=src/gnulib/aux --libtool --macro-prefix=gl memset strdup-posix
echo "Renaming c++defs into CPPdefs"
mv aux/snippet/c\+\+defs\.h "aux/snippet/cPPdefs.h"
find . -type f -exec sed -i -e 's/c++defs\.h/cPPdefs.h/g' {} \;
find . -name '*~' -delete
