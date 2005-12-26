#!/bin/sh

[ $# -eq 1 ] ||
{
    echo "usage: $0 root-directory" >&2
    exit 1
}

dir=$1
shift

echo "static char *tp_phase_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $dir/include/phases.h
echo "(char*)NULL};"

echo "static char *tp_resource_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $dir/include/resources.h
echo "(char*)NULL};"

echo "static char *tp_file_rsc_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $dir/include/printable_resources.h
echo "(char*)NULL};"

echo "static char *tp_property_names[]={"
sed '/#/d;/^[ 	]*$/d;s|^[ ]*\([^ ][^ ]*\).*|	"\1" ,|;' \
    $dir/etc/properties.rc
echo "(char*)NULL};"
