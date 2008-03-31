# $Id$

include $(ROOT)/makes/DEFAULT.mk

# ansi required for newgen (otherwise __STDC__ or __STRICT_ANSI__ not def).
CANSI	= -qlanglvl=ansi 

# -ma to allow alloca():
CFLAGS	= -g -O2 -qmaxmem=8192 -qfullpath -ma 

# if -E is ommitted, the file is *compiled*:-(
CMKDEP	= -E -M

# -lbsd added so signal work:
LDFLAGS	+= -lbsd

FFLAGS	= -g -O2 -u

#
# others

TAR	= tar
ZIP	= gzip -v9
DIFF	= diff

#
# Well, AIX is not unix, and xlc does not have the expected behavior under -M
DEPFILE = *.u

# no -ltermcap under AIX. -lcurses instead.
TPIPS_ADDED_LIBS =	-lreadline -lcurses

# wpips was never compiled under aix.
include $(ROOT)/makes/no_wpips.mk

