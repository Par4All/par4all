#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/08 10:49:15 $, 
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
	genread.h \
	genspec.h

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
	genread.c \
	genread_lex.c \
	genspec.c \
	genspec_lex.c

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

genread.h genread.c: read.y
	$(PARSE) $< 
	sed 's,YY,GENREAD_,g;s,yy,genread_,g' < y.tab.c > genread.c
	sed 's,YY,GENREAD_,g;s,yy,genread_,g' < y.tab.h > genread.h

genread_lex.o: genread.h
genread_lex.c: read.l 
	$(SCAN) $< | sed 's,YY,GENREAD_,g;s,yy,genread_,g;' > $@

genspec.h genspec.c: gram.y
	$(PARSE) $< 
	sed 's,YY,GENSPEC_,g;s,yy,genspec_,g' < y.tab.c > genspec.c
	sed 's,YY,GENSPEC_,g;s,yy,genspec_,g' < y.tab.h > genspec.h

genspec_lex.o: genspec.h
genspec_lex.c: token.l
	$(SCAN) $< | sed 's,YY,GENSPEC_,g;s,yy,genspec_,g' > $@

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

