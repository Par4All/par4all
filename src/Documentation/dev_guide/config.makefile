#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/09/02 11:13:24 $, 

G= developer_guide

SOURCES=	$(G).tex \
		$(G).bib

INSTALL_DOC=	$(G).ps

INSTALL_HTM=	$(G)

all: $(INSTALL_DOC) $(G).html

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) *.dvi $(G).html

# end of $RCSfile: config.makefile,v $
#
