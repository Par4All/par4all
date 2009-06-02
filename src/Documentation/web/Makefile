# 
# $Id$
#

#USER	= $(shell whoami)
#DEST	= $(USER)@ssh.cri.ensmp.fr:~pips/public_html

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

SOURCES= $(HTML_AUTO) $(HTML_OTHERS)

# Ask to make the html files:
all: $(HTMS)
clean: local-clean

INSTALL_HTM= $(HTMS)

# To deal with non-framed viewer and no server side include:
APPLY_CPP = $(CC) -E -C -P -

# Overkill for home.html and index.html but anyway...
%.html : %.cpp.html go_back.html
	$(APPLY_CPP) < $< > $@

home.html : home_content.html

index.html : home_content.html

clean: local-clean

local-clean:
	$(RM) $(HTML_AUTO:.cpp.html=.html) 

droits: $(HTMS)
	chmod -R a+r,g+w $(HTMS)

# # consider --dry-run
# RSOPT	=
# RSYNC	= rsync --rsh=ssh --archive --update --delete --verbose \
# 		--exclude=.svn $(RSOPT)

# # Assume that the files are already generated:
# publish : droits
# 	$(RSYNC) $(HTMS) $(DEST)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
