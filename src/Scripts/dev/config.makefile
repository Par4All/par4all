#
# $Id$
#

UTILS=		pips-makemake \
		install_pips_sources \
		pips_install_file \
		pips_set_directory_for_binaries

SCRIPTS = 	$(UTILS) \
		make-tags \
		analyze_libraries \
		grep_libraries \
		checked-out \
		order_libraries \
		pips-experiment \
		pips_at_night \
		pips_at_night_at_cri \
		remove_from_sccs_file \
		build_so_files

SRC_MACROS =	auto.h \
		define_libraries.sh \
		makefile_macros.comp \
		makefile_macros.doc \
		makefile_macros.flags \
		makefile_macros.gnu \
		makefile_macros.gnutools \
		makefile_macros.linux \
		makefile_macros.pentium \
		makefile_macros.nowpips \
		makefile_macros.ll \
		makefile_macros.misc \
		makefile_macros.. \
		makefile_macros.GNUSOL1 \
		makefile_macros.DEFAULT \
		makefile_macros.GNU \
		makefile_macros.SUN4 \
		makefile_macros.TEST \
		makefile_macros.GNULL \
		makefile_macros.LINUXI86 \
		makefile_macros.LINUXI86LL \
		makefile_macros.GNUSOL2LL \
		makefile_macros.SOL2LL \
		makefile_macros.GPROF \
		makefile_macros.IBMAIX \
		makefile_macros.IBMAIXLL \
		makefile_macros.CRAY-T3D \
		makefile_macros.CRAY-T3D-F90 \
		makefile_macros.CRAY-T3E \
		makefile_macros.OSF1 \
		makefile_macros.GNUOSF1 \
		makefile_macros.OSF1-SHORT \
		makefile_macros.GNUSOL2LLPROF \
		makefile_macros.HPUXLL

DDC_MACROS = 	define_libraries.make \
		auto-dash.h \
		auto-number.h

MACROS	=	$(DDC_MACROS) $(SRC_MACROS) 

COPY	=	cp

INSTALL_UTL=	$(SCRIPTS)
INSTALL_INC=	$(MACROS)

SOURCES	=	$(SCRIPTS) $(SRC_MACROS) forward_gnu_makefile

quick-install: install_forward_makefiles install_macros install_utils
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
	$(INSTALL) $(PIPS_ROOT)/Include $(MACROS) 
	$(INSTALL) $(NEWGEN_ROOT)/Include $(MACROS) 
	$(INSTALL) $(LINEAR_ROOT)/Include $(MACROS) 

install_utils: 
	#
	# installing shared utils
	#
	$(INSTALL) $(PIPS_ROOT)/Utils $(UTILS)
	$(INSTALL) $(NEWGEN_ROOT)/Utils $(UTILS)
	$(INSTALL) $(LINEAR_ROOT)/Utils $(UTILS)

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
		$(RM) $$d/$$s/Makefile; \
		$(COPY) forward_gnu_makefile $$d/$$s/Makefile; } ; done; done;\

# that is all
#
