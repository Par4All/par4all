#
# $Id$
#
# $Log: config.makefile,v $
# Revision 1.8  2001/05/25 09:49:44  nguyen
# Add new phases : alias_propagation and alias_check
# Move to transformation : partial_redundancy_elimination,
# array_resizing_top_down and array_resizing_instrumentation
# Rename : array_bound_check_top_down, array_bound_check_bottom_up and
# array_bound_check_interprocedural
#
# Revision 1.7  2001/01/04 09:48:36  nguyen
# Add new phase : adn_instrumentation
#
# Revision 1.6  2000/12/12 16:40:50  nguyen
# Add new phases : interprocedural_array_bound_check and top_down_array_declaration_normalization
#
# Revision 1.5  2000/09/22 09:32:59  nguyen
# Add Partial_redundancy_elimination for logical expression
#
# Revision 1.4  2000/08/22 09:56:21  nguyen
# Change the name of phases
# Add array_bound_check_instrumentation
#
# Revision 1.3  2000/06/07 15:22:15  nguyen
# Add new array bounds check version, based on regions
#
# Revision 1.2  2000/03/16 09:18:07  coelho
# array bound check moved here.
#
# Revision 1.1  2000/03/16 09:09:40  coelho
# Initial revision
#
#

LIB_CFILES 	= \
	array_bound_check_bottom_up.c \
	array_bound_check_top_down.c \
	array_bound_check_instrumentation.c \
	array_bound_check_interprocedural.c \
	alias_propagation.c \
	alias_check.c

LIB_HEADERS	= instrumentation-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)














