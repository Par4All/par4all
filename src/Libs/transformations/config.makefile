#
# $Id$
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

LIB_CFILES =	replace.c \
		loop_unroll.c \
		prettyprintcray.c \
		strip_mine.c \
		interactive_loop_transformation.c \
		loop_interchange.c \
		interchange.c \
		target.c \
		nest_parallelization.c\
                coarse_grain_parallelization.c\
		dead_code_elimination.c \
                trivial_test_elimination.c \
		privatize.c \
		array_privatization.c \
		standardize_structure.c \
		use_def_elimination.c \
		loop_normalize.c  \
		declarations.c \
		clone.c \
	        transformation_test.c \
		freeze_variables.c \
		array_resizing_bottom_up.c \
		array_resizing_top_down.c\
		array_resizing_statistic.c\
		partial_redundancy_elimination.c

#		optimize_misc.c

LIB_HEADERS =	transformations-local.h

LIB_OBJECTS =	$(LIB_CFILES:%.c=%.o)








