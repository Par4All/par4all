#
# $Id$
#

TARGET	= tpips

YOPT	= -v -d

INC_CFILES	= tpips.c wrapper.c

OTHER_CFILES	= main_tpips.c

DERIVED_HEADERS	= tp_yacc.h completion_list.h

DERIVED_CFILES	= tp_yacc.c tp_lex.c

DERIVED_FILES	= y.output

LIB_CFILES	= $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

BIN_TARGET	= $(TARGET)

INSTALL_LIB = 	$(ARCH)/main_tpips.o

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) $(DERIVED_FILES)

$(ARCH)/tpips: $(ARCH)/$(TPIPS_MAIN)

$(ARCH)/$(LIB_TARGET): completion_list.h

tp_lex.c: tp_lex.l tp_yacc.h
	$(SCAN) tp_lex.l | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/TP_/g;s/yy/tp_/g' > tp_lex.c

tp_yacc.c tp_yacc.h: tp_yacc.y
	$(PARSE) tp_yacc.y
	sed 's/YY/TP_/g;s/yy/tp_/g' y.tab.c > tp_yacc.c
	sed -e 's/YY/TP_/g;s/yy/tp_/g' y.tab.h > tp_yacc.h
	$(RM) y.output y.tab.c y.tab.h

completion_list.h :	$(PIPS_ROOT)/include/resources.h \
			$(PIPS_ROOT)/include/phases.h \
			$(PIPS_ROOT)/include/printable_resources.h \
			$(PIPS_ROOT)/etc/properties.rc
	./build_completion_lists.sh $(PIPS_ROOT) > $@
