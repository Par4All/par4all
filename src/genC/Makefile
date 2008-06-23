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

#
# GENREAD
#

YY2GR	= sed '/^FILE *\*/s,=[^,;]*,,g;s,YY,GENREAD_,g;s,yy,genread_,g'

genread_yacc.c: read.y
	$(PARSE) $< 
	$(YY2GR) < y.tab.c > genread_yacc.c
	$(YY2GR) < y.tab.h > genread.h
	$(RM) y.output y.tab.c y.tab.h

genread.h: genread_yacc.c

genread_lex.c: read.l genread.h
	$(SCAN) $< | \
	$(YY2GR) > $@

#
# GENSPEC
#

YY2GS	= sed '/^FILE *\*/s,=[^,;]*,,g;s,YY,GENSPEC_,g;s,yy,genspec_,g'

genspec_yacc.c: gram.y
	$(PARSE) $< 
	$(YY2GS) < y.tab.c > genspec_yacc.c
	$(YY2GS) < y.tab.h > genspec.h
	$(RM) y.output y.tab.c y.tab.h

genspec.h: genspec_yacc.c

genspec_lex.c: token.l genspec.h
	$(SCAN) $< | \
	$(YY2GS) > $@

#
# AH AH AH: parallel make issue
# artificial dependency otherwise there is a conflict y.tab.[ch]
#
genspec_yacc.c: genread.h

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
