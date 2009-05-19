# $Id$

TARGET	= transformations

LIB_CFILES =	replace.c \
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
		use_def_elimination.c \
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
		inlining.c \
		expression_substitution.c \
		loop_peeling.c \
		for_loop_recovering.c \
		scalarization.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
