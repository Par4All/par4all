# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/21 13:11:48 $, 

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
	$(RM) *.dvi

# end of $RCSfile: config.makefile,v $
#
