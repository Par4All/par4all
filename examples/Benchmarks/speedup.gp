set terminal postscript enhanced eps color size 10,2
set output "speedup.ps"




#set yrange [0.1:40]
#set xrange [0:30]
#set auto x
set logscale y 2


unset xtics
set xtics out rotate by -30 nomirror offset -1,0
#set xtics scale 0

bw = 1
g = 2

set boxwidth bw
set style fill solid 0.8 border -1
set style data histograms
set style histogram cluster gap 2
set key outside center bottom horizontal Right noreverse enhanced autotitles columnhead nobox 
