set terminal pdf
set output "pips-graph.pdf"
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 1
set xtic rotate by -45 scale 0
set ylabel "Normalized execution time"
set key outside right
plot "pips-graph.dat" using 2:xtic(1) ti col, '' u 3 ti col, '' u 4 ti col, '' u 5 ti col, '' u 6 ti col
