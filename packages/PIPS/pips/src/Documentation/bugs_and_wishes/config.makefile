# $Id$

ETEX = 	pips_bugs.tex

SOURCES = $(ETEX)

DVI = $(ETEX:.tex=.dvi)
PS = $(DVI:.dvi=.ps)

INSTALL_DOC= $(PS)

all: $(PS)
dvi: $(DVI)
ps: $(PS)

clean: local-clean
local-clean:
	$(RM) $(DVI) $(PS)
