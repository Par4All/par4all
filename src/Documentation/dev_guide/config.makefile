#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 18:56:35 $, 

G= developer_guide

SOURCES=	$(G).tex \
		$(G).bib

INSTALL_DOC=	$(G).ps

INSTALL_HTM=	$(G).html \
		$(G)

all: $(INSTALL_DOC) $(INSTALL_HTM)

# end of $RCSfile: config.makefile,v $
#
