#
# $Id$
# 

LIB_CFILES = \
	initializer.c \
	source_file.c \
	split_file.c \
	csplit_file.c

LIB_HEADERS = preprocessor-local.h \
              splitc.y \
              splitc.l

DERIVED_HEADERS= splitc.h
DERIVED_CFILES= splitcyaccer.c splitclexer.c

LIB_OBJECTS = $(LIB_CFILES:.c=.o)

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

cyaccer.c cyacc.h: splitc.y
	$(PARSE) cyacc.y
	sed 's/YY/SPLITC_/g;s/yy/splitc_/g' y.tab.c > splitcyaccer.c
	sed 's/YY/SPLITC_/g;s/yy/splitc_/g' y.tab.h > splitc.h
	$(RM) y.tab.c y.tab.h

splitclexer.c: splitc.l splitc.h
	$(SCAN) splitc.l | \
	sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/SPLITC_/g;s/yy/splitc_/g' > $@
