TARGET=$1.database/$1/$1.c
TARGETREF=$1.database/$1REF/$1REF.c
MAIN=$1.database/MAIN/MAIN.c

# add simd header 
if ! grep -q '\#include "../../simd.h"' $TARGET ; then
    sed -i -e '1 i \#include "../../simd.h"' $TARGET
fi

#fix main
if ! grep -q '\#include "../../fixmain.h"' $MAIN ; then
    sed -i -e '1 i \#include "../../fixmain.h"' $MAIN
fi

# dirty fix
for f in $TARGET $TARGETREF; do
if ! grep -q '\#include "../../dirtyfix.h"' $f ; then
    sed -i -e '1 i \#include "../../dirtyfix.h"' $f
fi
done

gcc -O6 -mmmx -msse $MAIN $TARGETREF $TARGET -I . -o $1.out

#gcc -O6 -mmmx -msse $1.database/$1/$1.c -I . -S

#rm -r -f $1.database
