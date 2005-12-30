#
# $Id$
#

AR	= ar
ARFLAGS	= rv
CC	= cc
CFLAGS	= -O -g
CMKDEP	= -M
LD	= $(CC)
RANLIB	= ranlib
LEX	= flex
LFLAGS	=
FC	= f77
FFLAGS	= -O -g
LINT	= lint
LINTFLAGS= -habxu

# svn stuff
SVN =		svn
IS_SVN_WC =	test -d .svn
BRANCH = 	svn_branch.sh
IS_SVN_BRANCH =	$(BRANCH) test
BRANCH_FLAGS =

# The parser can no longer be compiled with yacc...
YACC	= bison
YFLAGS	= -y

PROTO   = cproto
PRFLAGS    = -evcf2

# end of it!
#
