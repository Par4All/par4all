#!/bin/bash
head -n 1 "$1"
sed -e '1 d' "$1" |\
	awk '{ print $1,-($3/$2 - 1)*100 }'
