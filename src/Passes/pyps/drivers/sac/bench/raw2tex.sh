#!/bin/bash
echo '\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\hline'
sed -e 's/\t/\&/g' -e 's/_/\\_/g' "$1" |\
	sed -r -e 's/^/\\texttt{/' -e 's/&/}\&/' -e 's/&?$/\\\\\\hline/g'
echo '\hline\end{tabular}'
