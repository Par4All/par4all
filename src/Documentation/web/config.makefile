# 
# $RCSfile$ (version $Revision$)
# $Date$, 

# To deal with non-framed viewer and no server side include:
HTML_AUTO = index.html \
	bibliography.html \
	current_team.html \
	distribution.html \
	history.html \
	home.html \
	man_pages.html \
	navigation.html \
	related_projects.html \
	search.html \
	technical_pages.html

HTMS =	$(HTML_AUTO) \
	\
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

# Ask to make the html files:
all: $(HTML_AUTO)
clean: local_clean

INSTALL_HTM= $(HTMS)

# To deal with non-framed viewer and no server side include:
APPLY_CPP = cpp -C -P

# Overkill for home.html and index.html but anyway...
%.html : %.cpp.html go_back.html
	$(APPLY_CPP) $< > $@

home.html : home_content.html

index.html : home_content.html

local_clean:
	$(RM) $(HTML_AUTO)

# end of $RCSfile$
