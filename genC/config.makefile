#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/16 15:21:01 $, 
#
# Newgen should be quite particular...

YFLAGS+= -d -v

all: $(ARCH)/newC $(ARCH)/libgenC.a
recompile: all quick-install

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
        newgen_stack.h

OTHER_HEADERS=\
        newgen_include.h \
        read.y \
        read.l \
        gram.y \
        token.l

DERIVED_HEADERS=\
	x.tab.h \
	z.tab.h

LIB_CFILES=\
        build.c \
        genC.c \
        genClib.c \
        hash.c \
        set.c \
        list.c \
        stack.c

OTHER_CFILES=\
	new.c \
	genLisp.c \
	genSML.c

DERIVED_CFILES=\
	x.tab.c \
	lex.xx.c \
	z.tab.c \
	lex.zz.c

DERIVED_FILES=

CFILES=  $(LIB_CFILES) $(DERIVED_CFILES) $(OTHER_CFILES)
LIB_OBJECTS= $(addprefix $(ARCH)/,$(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o))

SOURCES= $(LIB_HEADERS) $(LIB_CFILES) \
	$(OTHER_HEADERS) $(OTHER_CFILES) \
	config.makefile

INSTALL_FILE= $(LIB_HEADERS)
INSTALL_LIB= $(ARCH)/libgenC.a
INSTALL_SHR= 
INSTALL_EXE= $(ARCH)/newC

#
# local rules

$(ARCH)/libgenC.a: $(LIB_OBJECTS)
	$(RM) $@
	$(ARCHIVE) $@ $+
	ranlib $@

x.tab.h x.tab.c: read.y
	$(PARSE) $< 
	sed 's,YY,XX,g;s,yy,xx,g' < y.tab.c > x.tab.c
	sed 's,YY,XX,g;s,yy,xx,g' < y.tab.h > x.tab.h

lex.xx.o: x.tab.h
lex.xx.c: read.l 
	$(SCAN) $< | sed 's,YY,XX,g;s,yy,xx,g;s,\([^<]string\),\1_flex,' > $@

z.tab.h z.tab.c: gram.y
	$(PARSE) $< 
	sed 's,YY,ZZ,g;s,yy,zz,g' < y.tab.c > z.tab.c
	sed 's,YY,ZZ,g;s,yy,zz,g' < y.tab.h > z.tab.h

lex.zz.o: z.tab.h
lex.zz.c: token.l
	$(SCAN) $< | sed 's,YY,ZZ,g;s,yy,zz,g' > $@

$(ARCH)/newC:	$(ARCH)/new.o $(ARCH)/libgenC.a
	$(LINK) $@ $+

$(ARCH)/newLisp: $(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genLisp.o
	$(LINK) $@ $+

$(ARCH)/newSML:	$(ARCH)/new.o $(ARCH)/build.o $(ARCH)/genSML.o
	$(LINK) $@ $+

clean-compiled:
	$(RM) $(ARCH)/libgenC.a $(ARCH)/*.o $(ARCH)/newC 

clean: clean-compiled
	$(RM) 	$(DERIVED_CFILES) $(DERIVED_HEADERS) \
		y.tab.c y.tab.h y.output

#

