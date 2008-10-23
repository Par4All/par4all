# $Id$ clean

clean: NO_INCLUDES=1
export NO_INCLUDES

# old stuff:
# paf-util pip prgm_mapping scheduling static_controlize reindexing array_dfg

# there is no rationnal order to compile the libraries:-(
# see local TODO
FWD_DIRS	= \
	misc newgen properties text-util pipsdbm \
	top-level ri-util conversion movements \
	comp_sections transformer bootstrap control flint \
	syntax c_syntax prettyprint \
	effects effects-generic effects-simple semantics complexity \
	continuation reductions regions effects-convex alias-classes \
	callgraph icfg ricedg \
	chains rice hyperplane transformations expressions \
	statistics instrumentation hpfc atomizer safescale sac phrase wp65 \
	preprocessor pipsmake
