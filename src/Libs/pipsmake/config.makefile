#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/25 15:38:51 $, 
#
YFLAGS+=	-d
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	pipsmake.c activate.c initializer.c callgraph.c \
		openclose.c callback.c
# FI: I do not know why these files are in LIB_HEADERS; they are probably put here
# to be copied in Production (my guess, 28 January 1992)
# LIB_HEADERS=	readmakefile.l readmakefile.y pipsmake-local.h phases.h
LIB_HEADERS=	readmakefile.l readmakefile.y pipsmake-local.h
DERIVED_HEADERS= y.tab.h
DERIVED_CFILES= y.tab.c lex.yy.c

LIB_OBJECTS=	$(DERIVED_CFILES:.c=.o)  $(LIB_CFILES:.c=.o) 

default: all

lex.yy.c: readmakefile.l y.tab.h
	$(SCAN) readmakefile.l | \
	sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' > lex.yy.c

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.
y.tab.c y.tab.h: readmakefile.y
	$(YACC) $(YFLAGS) readmakefile.y
	sed -e '/extern char \*malloc/d;s/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' \
		y.tab.c > m.tab.c
	mv m.tab.c y.tab.c
	sed -e 's/YY/PIPSMAKE_/g;s/yy/pipsmake_/g' y.tab.h > m.tab.h
	mv m.tab.h y.tab.h

m.tab.c: y.tab.c

depend: y.tab.c lex.yy.c

super-clean: clean
	rm -f y.tab.c lex.yy.c y.tab.h y.output

$(TARGET).h: $(DERIVED_HEADERS)
