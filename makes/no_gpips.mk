#
# $Id: no_wpips.mk 888 2008-03-31 10:12:55Z keryell $
#
# updates if no wpips on a pips architecture
# wpips was never compiled with aix for instance, hence

#debug_output := $(shell echo no_gpips.mk  > /dev/tty)

# Skip compiling GPips:
PIPS_NO_GPIPS = 1

# Do not link gpips into fpips:
CPPFLAGS 	+=	-DFPIPS_WITHOUT_GPIPS
