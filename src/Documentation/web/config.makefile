# 
# $RCSfile$ (version $Revision$)
# $Date$, 

# To deal with non-framed viewer and no server side include:
HTML_AUTO = \
	index.cpp.html \
	bibliography.cpp.html \
	current_team.cpp.html \
	distribution.cpp.html \
	history.cpp.html \
	home.cpp.html \
	man_pages.cpp.html \
	navigation.cpp.html \
	related_projects.cpp.html \
	search.cpp.html \
	technical_pages.cpp.html

HTML_OTHERS =	\
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
	home_content.html


HTMS =	$(HTML_AUTO:.cpp.html=.html) $(HTML_OTHERS)

SCRIPTS =	generate_pips_distributions

SOURCES= $(HTML_AUTO) $(HTML_OTHERS) $(SCRIPTS)

# Ask to make the html files:
all: $(HTMS)
clean: local-clean

INSTALL_HTM= $(HTMS)
INSTALL_UTL= $(SCRIPTS)

# To deal with non-framed viewer and no server side include:
APPLY_CPP = cpp -C -P

# Overkill for home.html and index.html but anyway...
%.html : %.cpp.html go_back.html
	$(APPLY_CPP) $< > $@

home.html : home_content.html

index.html : home_content.html

local-clean:
	$(RM) $(HTML_AUTO:.cpp.html=.html) 

# end of $RCSfile$
