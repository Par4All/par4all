#
#
#   	WP65: PUMA ESPRIT PROJECT 2701
#   	------------------------------
#
# Corinne Ancourt, Francois Irigoin, Lei Zhou	     17 October 1991
#
####### The source files directly involved in wp65 are:
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
# Do include the .c, .h and .o extensions.

LIB_CFILES=	code.c tiling.c variable.c instruction_to_wp65_code.c wp65.c basis.c \
		find_iteration_domain.c model.c references.c communications.c 
LIB_HEADERS=	wp65-local.h
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)

Lint:  
	$(LINT) $(PIPS_LINTFLAGS) $(LOCAL_FLAGS) $(PIPS_CPPFLAGS) $(LIB_CFILES) ../conversion/loop_iteration_domaine_to_sc.c| sed '/possible pointer alignment/d;/gen_alloc/d'
### End of config.makefile
