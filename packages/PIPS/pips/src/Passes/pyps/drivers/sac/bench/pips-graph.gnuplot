set terminal pdf
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 1
set xtic rotate by -45 scale 0
set ylabel "Speedup in % relative to \nGCC -O3 -fno-tree-vectorize"
set key outside right
set output "pips-seq.pdf"
set title "Initial code VS code transformed by SAC with SIMD simulator (comp.: GCC)"
plot "pips-graph.dat" using 3:xtic(1) ti col, '' u 6 ti col

set output "pips-sse-gcc.pdf"
set title "Initial code VS code transformed by SAC with SSE intrisics (comp.: GCC)"
plot "pips-graph.dat" using 3:xtic(1) ti col, '' u 7 ti col

set output "pips-sse-icc.pdf"
set title "Initial code VS code transformed by SAC with SSE intrisics (comp.: ICC)"
plot "pips-graph.dat" using 4:xtic(1) ti col, '' u 8 ti col

set output "pips-sse-llvm.pdf"
set title "Initial code VS code transformed by SAC with SSE intrisics (comp.: LLVM)"
plot "pips-graph.dat" using 5:xtic(1) ti col, '' u 9 ti col
