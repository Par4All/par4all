# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/21 11:31:29 $, 
#
# Newgen documention

FTEX =	greco.ftex

ETEX = 	manual.tex \
	paper.tex

SOURCES =	$(FTEX) $(ETEX)

PS =	$(FTEX:.ftex=.ps) $(ETEX:.tex=.ps)

INSTALL_DOC =	$(PS)
INSTALL_HTM =	$(PS:.ps=.html) $(PS:.ps=)

all: $(INSTALL_DOC) $(INSTALL_HTM)
ps: $(PS)
dvi: $(PS:.ps=.dvi)

clean: local-clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) *.dvi

# end of $RCSfile: config.makefile,v $
#
