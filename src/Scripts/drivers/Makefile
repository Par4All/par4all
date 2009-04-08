# $Id$
#
# pips shell drivers
#

SCRIPTS =	Display \
		Pips \
		Init \
		Perform \
		Delete \
		Build \
		Select \
		pips.sh

INSTALL_EXE =	$(SCRIPTS)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

#PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff:
LINKS =	pips tpips wpips fpips gpips

.build_exe: .build_links

.build_links:
	$(MAKE) links
	touch $@

links:
	# no direct dependency, as usual
	$(MAKE) $(EXE.d)
	for l in $(LINKS) ; do $(RM) $(EXE.d)/$$l ; done
	for l in $(LINKS) ; do ln -s pips.sh $(EXE.d)/$$l ; done

clean: links-clean

links-clean:
	$(RM) .build_links
