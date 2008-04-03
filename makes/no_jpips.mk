#
# $Id: nowpips.mk 881 2008-02-19 17:27:24Z keryell $
#
# To skip JPips building if the Java compiler is not available.

#debug_output := $(shell echo no_jpips.mk  > /dev/tty)

# Skip compiling JPips:
PIPS_NO_JPIPS = 1
