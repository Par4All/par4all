#
# $Id$
#

# where to find newgen generated files...
inc_dir = $(PIPS_ROOT)/Include
#inc_dir = $(PIPS_DEVEDIR)/Documentation/newgen
CPPOPT = -I$(inc_dir) -I$(NEWGEN_DEVEDIR)/genC

LIB_CFILES = \
		newgen.c \
		Pvecteur.c \
		Ppolynome.c \
		Psc.c

NEWGEN_CFILES = $(notdir $(wildcard $(inc_dir)/*.c))

$(NEWGEN_CFILES):
	for f in $(NEWGEN_CFILES:.c=) ; do \
		ln -s $(inc_dir)/$$f.c . ; \
	done

LIB_OBJECTS	=  $(LIB_CFILES:.c=.o) $(NEWGEN_CFILES:.c=.o)

clean: local-clean
local-clean:; $(RM) $(NEWGEN_CFILES)
