#
# Hpfc Makefile, Fabien COELHO
#
# $RCSfile: config.makefile,v $ ($Date: 1994/03/08 11:47:34 $) version $Revision$, got on %D%, %T%
# %A%
#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS)
CPPFLAGS=	$(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS)
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS)
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS)
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)

#
# The following macros define your library.
#
# Name of the library without the .a suffix.
TARGET= 	hpfc
#
# Name of the main program to test or use the library
MAIN=		main
#
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
PARSER_SRC= \

LIB_CFILES=	parser.c parser-util.c debug-util.c hpfc-util.c \
		norm-decl.c compile-decl.c compiler-util.c compiler.c \
		compile.c run-time.c generate.c statement.c  norm-code.c \
		io.c io-effects.c local-ri-util.c inits.c o-analysis.c \
		align-checker.c messages.c overlap.c normalize.c guard.c \
		ranges.c message-utils.c reduction.c build-system.c 

LIB_HEADERS=	f77keywords f77symboles hpfkeywords gram.y scanner.l \
		warning.h hpfc-local.h defines-local.h compiler_parameters.h \
		access_description.h \
		add-includes filter-hpf

# headers made by some rule (except $INC_TARGET)
DERIVED_HEADERS=toklex.h keywtbl.h tokyacc.h
LIB_OBJECTS= y.tab.o scanner.o $(LIB_CFILES:.c=.o) 

# parser.o y.tab.o scanner.o parser-util.o debug-util.o hpfc-util.o \
# norm-decl.o compile-decl.o compiler-util.o compiler.o compile.o \
# run-time.o generate.o statement.o  norm-code.o io.o io-effects.o \
# local-ri-util.o inits.o o-analysis.o align-checker.o messages.o \
# overlap.o normalize.o guard.o ranges.o message-utils.o reduction.o

todo: init .runables

sccs_close:
	@echo "closing the sccs session"
	sccs delget `sccs tell -u`

RUNABLES= Run RunHpfcTest add-includes filter-hpf

.runables: $(RUNABLES)
	-chmod ug+x $(RUNABLES)
	touch .runables

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.
y.tab.c: tokyacc.h gram.y
	cat tokyacc.h gram.y > yacc.in
	$(PARSE) -l yacc.in
	sed -e '/extern char \*malloc/d;s/YY/HH/g;s/yy/hh/g' y.tab.c | \
	  grep -v "#line" > s.tab.c
	mv s.tab.c y.tab.c

init: toklex.h keywtbl.h scanner.c y.tab.c

# For gcc: lex generated array initializations are reformatted with sed to avoid lots
#	of gcc warnings; the two calls to sed are *not* mandatory;
scanner.c: scanner.l
	$(SCAN) scanner.l | sed -e 's/YY/HH/g;s/yy/hh/g' | \
	sed -e '/hhcrank\[\]/,/^0,0};/s/^/{/;/hhcrank\[\]/,/^{0,0};$$/s/,$$/},/;/hhcrank\[\]/,/^{0,0};$$/s/,	$$/},/;/hhcrank\[\]/,/^{0,0};$$/s/,	/},	{/g;s/^{0,0};$$/{0,0}};/;/hhcrank\[\]/s/{//' | \
	sed -e 's/^0,	0,	0,/{0,	0,	0},/;s/^0,	0,	0};/{0,	0,	0}};/;/^hhcrank+/s/^/{/;/^{hhcrank+/s/,$$/},/;/^{hhcrank+/s/,	$$/},/' > scanner.c

keywtbl.h: toklex.h f77keywords hpfkeywords
	cp toklex.h keywtbl.h
	echo "struct Skeyword keywtbl[] = {"	>> keywtbl.h
	sed "s/^.*/{\"&\", TK_&},/" f77keywords	>> keywtbl.h
	sed "s/^.*/{\"&\", TK_&},/" hpfkeywords	>> keywtbl.h
	echo "{0, 0}"				>> keywtbl.h
	echo "};"				>> keywtbl.h

toklex.h: warning.h f77keywords hpfkeywords f77symboles
	cat f77keywords hpfkeywords f77symboles | \
	  nl -s: | \
	  cat warning.h - | \
	  sed "s/\([^:]*\):\(.*\)/#define TK_\2 \1/" > toklex.h

tokyacc.h: warning.h f77keywords hpfkeywords f77symboles
	cat f77keywords hpfkeywords f77symboles | \
	  nl -s: | \
	  cat warning.h - | \
	  sed "s/\([^:]*\):\(.*\)/%token TK_\2 \1/" > tokyacc.h

depend: scanner.c y.tab.c

CLEAN: clean
	-$(RM) keywtbl.h toklex.h tokyacc.h scanner.c yacc.in y.tab.c \
	y.output y.tab.h lex.yy.c 

tar:
	rm -f hpfc.tar.Z
	tar cvf hpfc.tar $(LIB_HEADERS) $(LIB_CFILES) Run RunHpfcTest \
		tests/*.f main.c hpfc.h \
		fortran-run-time/*.f \
		fortran-run-time/*.h \
		fortran-run-time/Makefile* \
		fortran-run-time/*.c \
		fortran-run-time/*.ref \
		RunHpfcTest
	compress -v hpfc.tar

$(TARGET).h: toklex.h tokyacc.h keywtbl.h
