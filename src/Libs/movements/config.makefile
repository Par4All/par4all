#
# Do not include the main program source file.
# database.c and options.c should not be put in semantics library
# but used to produce the semantics pass
LIB_CFILES= movement_computation.c  bound_generation.c operation.c\
	complex_bound_generation.c constraint_distribution.c \
	 make_loop_body.c build_sc_machine.c build_sc_tile.c sc_add_variable.c

LIB_HEADERS= movements-local.h


LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
























