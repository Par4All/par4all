#!/bin/sh

[ $# -eq 4 ] ||
{
    echo "usage: $0 phases.h resources.h printable_resources.h properties.rc" >&2
    exit 1
}

echo "static char *tp_phase_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $1
echo "(char*)NULL};"

echo "static char *tp_resource_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $2
echo "(char*)NULL};"

echo "static char *tp_file_rsc_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $3
echo "(char*)NULL};"

echo "static char *tp_property_names[]={"
sed '/#/d;/^[ 	]*$/d;s|^[ ]*\([^ ][^ ]*\).*|	"\1" ,|;' \
    $4
echo "(char*)NULL};"
