# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/21 13:11:29 $, 

ETEX = 	pips_bugs.tex

SOURCES = $(ETEX)

DVI = $(ETEX:.tex=.dvi)
PS = $(DVI:.dvi=.ps)

all: $(PS)
dvi: $(DVI)
ps: $(PS)

INSTALL_DOC= $(PS)

clean: local-clean
local-clean:
	$(RM) *.dvi

# end of $RCSfile: config.makefile,v $
#
