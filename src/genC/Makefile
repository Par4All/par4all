# $Id$
#
# Newgen should be quite particular...
#

########################################################## STANDARD DEFINES

# Have a look to standard_defines.mk
include standard_defines.mk


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
