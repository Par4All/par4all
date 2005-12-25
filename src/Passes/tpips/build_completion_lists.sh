#!/bin/sh

echo "static char *tp_phase_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' $PIPS_ROOT/include/phases.h
echo "(char*)NULL};"

echo "static char *tp_resource_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' $PIPS_ROOT/include/resources.h
echo "(char*)NULL};"

echo "static char *tp_file_rsc_names[]={"
sed '/define/!d;s|.*\(".*"\).*|	\1 ,|;' \
    $PIPS_ROOT/include/printable_resources.h
echo "(char*)NULL};"

echo "static char *tp_property_names[]={"
sed '/#/d;/^[ 	]*$/d;s|^[ ]*\([^ ][^ ]*\).*|	"\1" ,|;' \
    $PIPS_ROOT/etc/properties.rc
echo "(char*)NULL};"
