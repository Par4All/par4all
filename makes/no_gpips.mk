#
# $Id$
#
# updates if no gpips on a pips architecture, for example if the GTK
# development infrastructure is not installed.

#debug_output := $(shell echo no_gpips.mk  > /dev/tty)

# Skip compiling GPips:
PIPS_NO_GPIPS = 1

# Do not link gpips into fpips:
CPPFLAGS 	+=	-DFPIPS_WITHOUT_GPIPS
