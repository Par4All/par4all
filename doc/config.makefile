# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/23 11:52:19 $, 
#
# Newgen documention

FTEX =	tutoriel_newgen.ftex

ETEX = 	newgen_manual.tex \
	newgen_paper.tex

SOURCES =	$(FTEX) $(ETEX)

PS =	$(FTEX:.ftex=.ps) $(ETEX:.tex=.ps)

INSTALL_DOC =	$(PS)
INSTALL_HTM =	$(PS:.ps=.html) $(PS:.ps=)

all: $(INSTALL_DOC) $(PS:.ps=.html) 
ps: $(PS)
dvi: $(PS:.ps=.dvi)

clean: local-clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) *.dvi $(FTEX:.ftex=.tex)

# end of $RCSfile: config.makefile,v $
#
