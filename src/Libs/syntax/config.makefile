# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/10 15:57:05 $m 
#
# -O2 is too much indeed for syntax, FC 09/06/94:-)
# bof...
ifeq ($(CC),gcc)
CFLAGS=	-g -Wall -ansi
else
CFLAGS=	-g
endif

# I wanna the header file for the lexer
YFLAGS+=-d

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

DERIVED_HEADERS= keywtbl.h tokyacc.h syn_yacc.h
DERIVED_CFILES= syn_yacc.c scanner.c

LIB_OBJECTS=	$(DERIVED_CFILES:.c=.o)  $(LIB_CFILES:.c=.o) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.

syn_yacc.c syn_yacc.h: tokyacc.h gram.y
	cat tokyacc.h gram.y > yacc.in
	$(PARSE) yacc.in
	sed 's/YY/SYN_/g;s/yy/syn_/g' y.tab.c > syn_yacc.c
	sed 's/YY/SYN_/g;s/yy/syn_/g' y.tab.h > syn_yacc.h
	$(RM) y.tab.c y.tab.h yacc.in


# For gcc: lex generated array initializations are reformatted with sed to
# avoid lots of gcc warnings; the two calls to sed are *not* mandatory;
#

scanner.c: scanner.l 
	$(SCAN) scanner.l | \
	sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/SYN_/g;s/yy/syn_/g' > $@

keywtbl.h: warning.h f77keywords
	@echo "Generating $@"
	{ cat warning.h ; \
	echo "#include \"syn_yacc.h\"" ; \
	echo "struct Skeyword keywtbl[] = {" ;\
	sed "s/^.*/{\"&\", TK_&},/" f77keywords ;\
	echo "{0, 0}" ;\
	echo "};" ; } > keywtbl.h

tokyacc.h: warning.h f77keywords f77symboles
	cat f77keywords f77symboles | nl -s: | cat warning.h - | sed "s/\([^:]*\):\(.*\)/%token TK_\2 \1/" > tokyacc.h

# end of $RCSfile: config.makefile,v $
#
