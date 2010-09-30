# re-run the benchmarking suite if a benchmark program is modified
benchfiles = $(wildcard *.c)

all: pips-graph.pdf

pips-graph.pdf: pips-graph.dat pips-graph.gnuplot
	gnuplot pips-graph.gnuplot

pips-graph.dat: $(benchfiles)
	./benchmark_sac.py --normalize --outfile $@

.PHONY: clean
clean:
	-rm pips-graph.dat pips-graph.pdf
