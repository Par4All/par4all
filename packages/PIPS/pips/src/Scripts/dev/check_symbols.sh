#!/bin/bash
set -e
case "$1" in
	-l|--list)
	mode="list"
	where="$2"
	;;
	-c|--check)
	mode="check"
	where="$2"
	;;
	*)
	echo 1>&2 "usage: check_symbols.sh (--list|--check) [dirname]"
	exit 0
	;;
esac

# disclaimer
cat 1>&2 << EOF
=======================
 PIPS object inspector
  by serge guelton
=======================
EOF

# check PWD 
test -e ./config.status || ( echo 1>&2 "not in top_builddir !" ; exit 1 )

echo 1>&2 "= building list of symbols ..."
all_undefined_symbols="`mktemp -t symbols.XXXXXX`"
find -name '*.o' -exec nm {} \; | cut -b 10- | grep '^U' | cut -b 3- | sort -u > $all_undefined_symbols

if test $mode = check ; then
echo 1>&2 "= unused symbols from '$where' ..."
find "$where" -name '*.o' -exec nm {} \; | cut -b 10- | grep '^T' | cut -b 3- | sort -u | \
	while read line ; do if ! grep -q "^${line}$" $all_undefined_symbols && ! grep -q " ${line}(" src/Documentation/pipsmake/builder_map.h ; then echo "$line" ; fi ; done
else
echo 1>&2 "= exported symbols from '$where' ..."
find "$where" -name '*.o' -exec nm {} \; | cut -b 10- | grep '^T' | cut -b 3- | sort -u | \
	while read line ; do if ! grep -q " ${line}(" src/Documentation/pipsmake/builder_map.h ; then echo "$line" ; fi ; done
fi

echo 1>&2 "======= o(^_-)0 ======="
rm -f $all_undefined_symbols

