#!/bin/bash
cat << EOF
==========================
PIPS property analyzer
by serge guelton o(^_-)O
==========================
EOF



if ! test -f "$1" || ! test `basename "$1"` = properties.rc ; then
	echo "must be called with the property file as first parameter" 1>&2
	exit 1
fi
if ! test -f "$2/src/Libs/ri-util/ri-util-local.h" ; then
	echo "must be called with the root source dir as second parameter" 1>&2
	exit 1
fi
echo "verify all properties in '$1' are used at least once" 1>&2
all_ok=0
all_sources="`find $2/src -name '*.[chly]'`" 
sed -e '/^$/ d' $1 | cut -d ' ' -f 1 | \
while read line ; do
	if test "$line" ; then
		if ! grep -q $line $all_sources && ! grep -q $line $0; then
			echo "$line never used"
			all_ok=1
		fi
	fi
done 

test $all_ok = 0 && echo "nothing found" 1>&2
exit $all_ok

# use a white list because some directives are not grepable
HPFC_IGNORE_FCD_SYNCHRO
HPFC_IGNORE_FCD_SET
HPFC_IGNORE_FCD_TIME
