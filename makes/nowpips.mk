#
# $Id$
#
# updates if no wpips on a pips architecture
# wpips was never compiled with aix for instance, hence

#debug_output := $(shell echo nowpips.mk  > /dev/tty)

CPPFLAGS 	+=	-DFPIPS_WITHOUT_WPIPS 

# Skip compiling WPips:
NOWPIPS = 1
