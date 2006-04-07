#
# $Id$
#
# updates if no wpips on a pips architecture
# wpips was never compiled with aix for instance, hence

CPPFLAGS 	+=	-DFPIPS_WITHOUT_WPIPS 

# Remove the -lwpips:
FPIPS_ADDED_LIBS =
