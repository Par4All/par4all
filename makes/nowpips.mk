#
# $Id$
#
# updates if no wpips on a pips architecture
# wpips was never compiled with aix for instance, hence

CPPFLAGS 	+=	-DFPIPS_WITHOUT_WPIPS 

WPIPS_ADDED_LIBS =

FPIPS_ADDED_LIBS =	-lpips -ltpips $(TPIPS_ADDED_LIBS)
