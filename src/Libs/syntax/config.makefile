#
# -O2 is too much indeed for syntax, FC 09/06/94:-)
# bof...
ifeq ($(ARCH),SUN4)
CFLAGS=	-g
else
CFLAGS=	-g -Wall -ansi
endif

# bison does not like this pips grammar for Fortran, it reports errors:-)
YACC=yacc
YFLAGS=

PARSER_SRC= 

LIB_CFILES=	util.c \
		declaration.c \
		expression.c \
		equivalence.c \
		parser.c \
		procedure.c \
		reader.c \
		statement.c \
		return.c \
		malloc-info.c \
		clean.c

LIB_HEADERS=	f77keywords \
		f77symboles \
		gram.y \
		scanner.l \
		warning.h \
		syntax-local.h

# headers made by some rule (except $INC_TARGET)

DERIVED_HEADERS=toklex.h keywtbl.h tokyacc.h
DERIVED_CFILES= y.tab.c scanner.c

LIB_OBJECTS=	$(DERIVED_CFILES:.c=.o)  $(LIB_CFILES:.c=.o) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.

y.tab.c: tokyacc.h gram.y
	cat tokyacc.h gram.y > yacc.in
	$(PARSE) yacc.in
	sed -e '/extern char \*malloc/d;s/YY/MM/g;s/yy/ss/g' y.tab.c > s.tab.c
	mv s.tab.c y.tab.c


# For gcc: lex generated array initializations are reformatted with sed to
# avoid lots of gcc warnings; the two calls to sed are *not* mandatory;
#

scanner.c: scanner.l toklex.h
	$(SCAN) scanner.l | sed -e 's/YY/MM/g;s/yy/ss/g' | \
	sed -e '/sscrank\[\]/,/^0,0};/s/^/{/;/sscrank\[\]/,/^{0,0};$$/s/,$$/},/;/sscrank\[\]/,/^{0,0};$$/s/,	$$/},/;/sscrank\[\]/,/^{0,0};$$/s/,	/},	{/g;s/^{0,0};$$/{0,0}};/;/sscrank\[\]/s/{//' | \
	sed -e 's/^0,	0,	0,/{0,	0,	0},/;s/^0,	0,	0};/{0,	0,	0}};/;/^sscrank+/s/^/{/;/^{sscrank+/s/,$$/},/;/^{sscrank+/s/,	$$/},/' > scanner.c

keywtbl.h: toklex.h f77keywords
	cp toklex.h keywtbl.h
	echo "struct Skeyword keywtbl[] = {"	>> keywtbl.h
	sed "s/^.*/{\"&\", TK_&},/" f77keywords	>> keywtbl.h
	echo "{0, 0}"				>> keywtbl.h
	echo "};"				>> keywtbl.h

toklex.h: warning.h f77keywords f77symboles
	cat f77keywords f77symboles | nl -s: | cat warning.h - | sed "s/\([^:]*\):\(.*\)/#define TK_\2 \1/" > toklex.h

tokyacc.h: warning.h f77keywords f77symboles
	cat f77keywords f77symboles | nl -s: | cat warning.h - | sed "s/\([^:]*\):\(.*\)/%token TK_\2 \1/" > tokyacc.h

depend: scanner.c y.tab.c

CLEAN: clean
	-rm -f keywtbl.h toklex.h tokyacc.h scanner.c yacc.in y.tab.c \
	y.output y.tab.h lex.yy.c 

#
# end of config.makefile
#
