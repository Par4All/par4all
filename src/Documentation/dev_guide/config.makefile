#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 19:04:31 $, 

G= developer_guide

SOURCES=	$(G).tex \
		$(G).bib

INSTALL_DOC=	$(G).ps

INSTALL_HTM=	$(G).html \
		$(G)

all: $(INSTALL_DOC) $(INSTALL_HTM)

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) *.dvi

# end of $RCSfile: config.makefile,v $
#
