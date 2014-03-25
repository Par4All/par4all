#!/bin/sh
# use this script to update gnulib
# because of naming convention limitations introduced by FC
# we cannot solely rely on a gnulib-tool --update

set -e
gnulib-tool --import --dir=../../.. --lib=libgnu --source-base=src/Libs/gnulib --m4-base=src/Libs/gnulib/m4 --doc-base=doc --tests-base=tests --aux-dir=src/Libs/gnulib/aux --libtool --macro-prefix=gl strdup-posix strndup vasprintf strchrnul
echo "Renaming c++defs into CPPdefs"
mv aux/snippet/c\+\+defs\.h "aux/snippet/cPPdefs.h"
find . -type f -exec sed -i -e 's/c++defs\.h/cPPdefs.h/g' {} \;
find . -name '*~' -delete
echo "Renaming float+ into float_p"
mv float\+\.h "float_p.h"
find . -type f -exec sed -i -e 's/float+\.h/float_p.h/g' {} \;
find . -name '*~' -delete

