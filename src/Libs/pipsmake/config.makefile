#
# $Id$
#

YFLAGS+=	-d

# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES=	pipsmake.c \
		activate.c \
		callgraph.c \
		openclose.c \
		callback.c \
		unsplit.c

LIB_HEADERS=	readmakefile.l \
		readmakefile.y \
		pipsmake-local.h

DERIVED_HEADERS= pipsmake_yacc.h
DERIVED_CFILES= pipsmake_yacc.c pipsmake_lex.c

LIB_OBJECTS=	$(DERIVED_CFILES:.c=.o)  $(LIB_CFILES:.c=.o) 

default: all

pipsmake_lex.c: readmakefile.l pipsmake_yacc.h
	$(SCAN) readmakefile.l | \
	sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' \
		> $@

pipsmake_yacc.c pipsmake_yacc.h: readmakefile.y
	$(PARSE) readmakefile.y
	sed 's/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' y.tab.c > pipsmake_yacc.c
	sed 's/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' y.tab.h > pipsmake_yacc.h
	$(RM) y.tab.[hc] y.output

.depend: $(DERIVED_CFILES)
.header: $(DERIVED_HEADERS)

# end of $RCSfile: config.makefile,v $
#
