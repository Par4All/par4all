#!/bin/bash

# test for tpips
echo "using the folowing tpips program:"
which tpips
if [ ! $? -eq 0 ]
then
    echo "ERROR: no tpips found."
    echo "You may need to install the Par4All environment or"
    echo "You may need to set up your Par4All environment"
    echo ""
    exit 1
fi
