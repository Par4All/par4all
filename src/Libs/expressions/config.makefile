#
# $Id$
#
# This library holds pips transformations which deal with expressions.
# For expression utilities, rather look in "ri-util/expression.c"
#

LIB_CFILES =	partial_eval.c \
		simple_atomize.c \
		forward_substitution.c \
		optimize.c \
		sequence_gcm_cse.c

LIB_HEADERS =	expressions-local.h

LIB_OBJECTS =	$(LIB_CFILES:%.c=%.o)
