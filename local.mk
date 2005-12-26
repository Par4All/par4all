# $Id$

# create links for *pips shell wrapper
LINKS	= ./bin/pips ./bin/tpips ./bin/wpips ./bin/fpips

links:
	$(RM) $(LINKS)
	for l in $(LINKS) ; do ln -s pips.sh $$l ; done

phase3: links
