#
# Source, header and object files used to build the library.
LIB_CFILES= 	change_of_Pbase.c \
		code_change_of_basis.c \
		loop_iteration_domaine_to_sc.c \
		look_for_nested_loops.c \
		system_to_code.c

LIB_HEADERS=	conversion-local.h
LIB_OBJECTS= 	$(LIB_CFILES:.c=.o) 

#
# end of config.makefile
#
