TARGET	= transformations
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libtransformations.la
libtransformations_la_SOURCES=\
		loop_unroll.c \
		prettyprintcray.c \
		strip_mine.c \
		interactive_loop_transformation.c \
		loop_interchange.c \
		loop_normalize.c \
		interchange.c \
		target.c \
		nest_parallelization.c \
		coarse_grain_parallelization.c \
		dead_code_elimination.c \
		trivial_test_elimination.c \
		privatize.c \
		array_privatization.c \
		standardize_structure.c \
		simplify_control.c \
		declarations.c \
		clone.c \
		transformation_test.c \
		freeze_variables.c \
		array_resizing_bottom_up.c \
		array_resizing_top_down.c \
		array_resizing_statistic.c \
		partial_redundancy_elimination.c \
		variable_expansion.c \
		internalize_parallel_code.c \
		index_set_splitting.c \
		for_loop_recovering.c \
		generate_pragma.c \
		manage_pragma.c \
		scalarization.c \
		flatten_code.c \
		induction_substitution.c \
		loop_fusion.c \
		set_return_type.c \
		cast_at_call_sites.c \
		manage_parallel_loops.c \
		registers.c \
		identity_elimination.c

include $(srcdir)/../pipslibs_includes.mk
