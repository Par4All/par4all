#! /bin/sh
#
# $Id$
#
# test type translation...
#

test $# -eq 1 || {
    echo "USAGE: $0 database-directory"
    exit 1;
}

database=$1
newgen=$1/Metadata/database.NEWGEN
mv $newgen $newgen.old

test_newgen_type_translation.pl $1/*/* < $newgen.old > $newgen
