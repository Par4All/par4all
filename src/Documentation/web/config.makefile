# 
# $RCSfile$ (version $Revision$)
# $Date$, 

HTMS =	index.html \
	bibliography.html \
	history.html \
	home.html \
	man_pages.html \
	related_projects.html \
	batch-interface.html \
	line-interface.html \
	outline.html \
	pips_summary.html \
	pipsmake.html \
	poly_meth.html \
	regions.html \
	summary_hpfc.html \
	window-interface.html \
	wp65.html \
	wp65_summary.html \
	distribution.html

SOURCES= $(HTMS)

all: home.html index.html
clean: local_clean

INSTALL_HTM= $(HTMS)

# To deal with non-framed viewer and no server side include:
APPLY_CPP = cpp -C -P
home.html : home.cpp.html home_content.html
	$(APPLY_CPP) home.cpp.html > $@

index.html : index.cpp.html home_content.html
	$(APPLY_CPP) index.cpp.html > $@

local_clean:
	$(RM) home.html index.html

# end of $RCSfile$
