# $Id$
#
# Newgen should be quite particular...
#

############################################################## STANDARD DEFINES

LIB_HEADERS=\
        genC.h \
        newgen_assert.h \
        newgen_types.h \
        newgen_hash.h \
        newgen_set.h \
        newgen_list.h \
        newgen_generic_mapping.h \
        newgen_generic_stack.h \
        newgen_generic_function.h \
        newgen_map.h \
        newgen_stack.h \
	newgen_array.h

DERIVED_LIB_HEADERS	= \
	newgen_auto_string.h

OTHER_HEADERS=\
        newgen_include.h \
        read.y \
        read.l \
        gram.y \
        token.l

DERIVED_HEADERS=\
	genread.h \
	genspec.h \
	$(DERIVED_LIB_HEADERS)

DERIVED_CFILES=\
	genread_yacc.c \
	genread_lex.c \
	genspec_yacc.c \
	genspec_lex.c

LIB_CFILES=\
        build.c \
        genC.c \
        genClib.c \
	tabulated.c \
        hash.c \
        set.c \
        list.c \
        stack.c \
	array.c \
	string.c \
	$(DERIVED_CFILES)

OTHER_CFILES=\
	new.c \
	genLisp.c \
	genSML.c

INSTALL_INC	= $(LIB_HEADERS) $(DERIVED_LIB_HEADERS)
LIB_TARGET	= libgenC.a
BIN_TARGET	= newC

################################################################## COMMON STUFF

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif
include $(ROOT)/makes/main.mk

# LEX=flex
YFLAGS+= -d -v

# all: $(ARCH)/newC $(ARCH)/libgenC.a
#recompile: all quick-install

# dependences.
$(addprefix $(ARCH)/,$(OTHER_CFILES:.c=.o)) $(LIB_OBJECTS): $(DERIVED_HEADERS)

# local rules
newgen_auto_%.h: %.c
	$(RM) $@ ; $(PROTOIZE) $< > $@

genread.h genread_yacc.c: read.y
	$(PARSE) $< 
	sed 's,YY,GENREAD_,g;s,yy,genread_,g' < y.tab.c > genread_yacc.c
	sed 's,YY,GENREAD_,g;s,yy,genread_,g' < y.tab.h > genread.h
	$(RM) y.output y.tab.c y.tab.h

#genread_lex.o: genread.h
genread_lex.c: read.l 
	$(SCAN) $< | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s,YY,GENREAD_,g;s,yy,genread_,g;' > $@

genspec.h genspec_yacc.c: gram.y
	$(PARSE) $< 
	sed 's,YY,GENSPEC_,g;s,yy,genspec_,g' < y.tab.c > genspec_yacc.c
	sed 's,YY,GENSPEC_,g;s,yy,genspec_,g' < y.tab.h > genspec.h
	$(RM) y.output y.tab.c y.tab.h

#genspec_lex.o: genspec.h
genspec_lex.c: token.l
	$(SCAN) $< | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s,YY,GENSPEC_,g;s,yy,genspec_,g' > $@

$(ARCH)/newC:	$(ARCH)/new.o $(ARCH)/libgenC.a
	$(LINK) $@ $(ARCH)/new.o $(ARCH)/libgenC.a

$(ARCH)/newLisp: $(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genLisp.o
	$(LINK) $@ $(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genLisp.o

$(ARCH)/newSML:	$(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genSML.o
	$(LINK) $@ $(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genSML.o

clean-compiled:
	$(RM) $(ARCH)/libgenC.a $(ARCH)/*.o $(ARCH)/newC 

local-clean: 
	$(RM) 	$(DERIVED_CFILES) $(DERIVED_HEADERS) \
		y.tab.c y.tab.h y.output

clean: clean-compiled local-clean
