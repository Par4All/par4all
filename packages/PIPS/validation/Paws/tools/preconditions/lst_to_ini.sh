#!/bin/sh

# Convert .lst to .ini files ; disabled by default

for cat in properties analyses phases ; do
    #sed 's/^\([^;]*\);/\1 = /' $cat.lst  | sed 's/<</\n[/' | sed 's/>>/]\n/' > toolinfo.ini
done