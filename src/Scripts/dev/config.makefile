#
# $RCSfile: config.makefile,v $ for dev
#

SCRIPTS = 	pips-makemake \
		install_pips_sources \
		make-tags \
		pips_install_file \
		analyze_libraries \
		clean-pips \
		grep_libraries \
		search-pips \
		checked-out \
		order_libraries \
		pips-experiment \
		pips_at_night \
		remove_from_sccs_file

SRC_MACROS =	auto.h \
		define_libraries.sh \
		makefile_macros.. \
		makefile_macros.DEFAULT \
		makefile_macros.GNU \
		makefile_macros.SUN4 \
		makefile_macros.TEST \
		makefile_macros.GNULL \
		makefile_macros.GNUSOL2LL \
		makefile_macros.GPROF

DDC_MACROS = 	define_libraries.make \
		auto-dash.h \
		auto-number.h

MACROS	=	$(DDC_MACROS) $(SRC_MACROS) 

COPY	=	cp -f

INSTALL_UTL=	$(SCRIPTS)
INSTALL_INC=	$(MACROS)

SOURCES	=	$(SCRIPTS) $(SRC_MACROS) forward_gnu_makefile

quick-install: install_forward_makefiles install_macros 
all: $(DDC_MACROS)

auto-dash.h: auto.h
	sed 's,^...,-- ,;s,^..,--,' $< > $@

auto-number.h: auto.h
	sed 's,^...,#  ,;s,^..,# ,' $< > $@

define_libraries.make: define_libraries.sh
	# make macros and sh variables can be initialized nearly the same way
	sed "s,$<,$@,g;s,',,g" $< > $@

clean: local-clean
local-clean:; $(RM) $(DDC_MACROS)

#
# bootstraping temporarily include files if needed...
$(PIPS_ROOT)/Include/makefile_macros.$(ARCH):; touch $@
$(PIPS_ROOT)/Include/define_libraries.make:; touch $@

install_macros: $(DDC_MACROS)
	#
	# installing makefile macros for pips/newgen/linear
	#
	$(COPY) $(MACROS) $(PIPS_ROOT)/Include
	$(COPY) $(MACROS) $(NEWGEN_ROOT)/Include
	$(COPY) $(MACROS) $(LINEAR_ROOT)/Include

#
# where to install forward makefiles

SUBDIRS = . Src Libs Passes Scripts Runtimes Documentation \
	Src/Libs Src/Passes Src/Scripts Src/Runtimes Src/Documentation

DIRS= 	$(PIPS_ROOT) $(NEWGEN_ROOT) $(LINEAR_ROOT) \
	$(PIPS_DEVEDIR) $(NEWGEN_DEVEDIR) $(LINEAR_DEVEDIR)

install_forward_makefiles: 
	# 
	# copying directory makefiles where required (and usefull)
	#
	for d in $(DIRS) ; do \
	  for s in $(SUBDIRS) ; do \
	    test ! -d $$d/$$s || \
		{ echo "copying forward makefile to $$d/$$s"; \
		$(COPY) forward_gnu_makefile $$d/$$s/Makefile; } ; done; done;\

# that is all
#
