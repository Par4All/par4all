#!/bin/bash

# test for tpips
echo "using the folowing tpips program:"
which tpips
if [ ! $? -eq 0 ]
then
    echo "ERROR: no tpips found."
    echo "You may need to install the PIPS environment or"
    echo "You may need to set up your PIPS environment"
    echo ""
    exit 1
fi
