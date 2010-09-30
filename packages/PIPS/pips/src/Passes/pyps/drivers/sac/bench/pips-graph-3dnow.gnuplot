set terminal pdf
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 1
set xtic rotate by -45 scale 0
set ylabel "Speedup in % relative to GCC -O3"
set key outside right
set output "pips-3dnow-gcc.pdf"
set title "Initial code VS code transformed by SAC with 3Dnow intrisics (comp.: GCC)"
plot "pips-graph-3dnow.dat" using 2:xtic(1) ti col
