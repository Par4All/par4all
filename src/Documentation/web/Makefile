# 
# $Id$
#

PIPS_HTML_HOST = ssh-cri.ensmp.fr
PIPS_HTML_DIR = ~pips/public_html

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
	go_back.html \
	home_content.html \
	images \
	line-interface.html \
	outline.html \
	pips_summary.html \
	pipsmake.html \
	poly_meth.html \
	regions.html \
	summary_hpfc.html \
	tpips.css \
	window-interface.html \
	wp65.html \
	wp65_summary.html


HTMS =	$(HTML_AUTO:.cpp.html=.html) $(HTML_OTHERS)

SCRIPTS =	generate_pips_distributions

SOURCES= $(HTML_AUTO) $(HTML_OTHERS) $(SCRIPTS)

# Ask to make the html files:
all: $(HTMS)
clean: local-clean

INSTALL_HTM= $(HTMS)
INSTALL_UTL= $(SCRIPTS)

# To deal with non-framed viewer and no server side include:
APPLY_CPP = $(CC) -E -C -P -

# Overkill for home.html and index.html but anyway...
%.html : %.cpp.html go_back.html
	$(APPLY_CPP) < $< > $@

home.html : home_content.html

index.html : home_content.html

local-clean:
	$(RM) $(HTML_AUTO:.cpp.html=.html) 

# Assume that the files are already generated:
publish :
	chmod g+w $(HTMS)
	rsync --archive --update --verbose $(HTMS) $(PIPS_HTML_HOST):$(PIPS_HTML_DIR)
