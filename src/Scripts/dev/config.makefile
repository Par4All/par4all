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

MACROS	=	makefile_macros.. \
		makefile_macros.DEFAULT \
		makefile_macros.GNU \
		makefile_macros.SUN4 \
		makefile_macros.TEST \
		makefile_macros.GNULL \
		makefile_macros.GNUSOL2LL

COPY	=	cp -f

INSTALL_UTL=	$(SCRIPTS)
INSTALL_INC=	$(MACROS)

SOURCES	=	$(SCRIPTS) $(MACROS) forward_gnu_makefile config.makefile

quick-install: install_forward_makefiles install_macros

install_macros:
	#
	# installing makefile macros for pips/newgen/linear
	#
	# $(COPY) $(MACROS) $(PIPS_INCLUDEDIR)
	$(COPY) $(MACROS) $(NEWGEN_INCLUDEDIR)
	$(COPY) $(MACROS) $(LINEAR_INCLUDEDIR)

install_forward_makefiles: 
	# 
	# copying directory makefiles where required (and usefull)
	#
	$(COPY) forward_gnu_makefile ${PIPS_DEVEDIR}/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_DEVEDIR}/Lib/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_DEVEDIR}/Passes/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_DEVEDIR}/Scripts/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_DEVEDIR}/Runtime/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_PRODDIR}/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_SRCDIR}/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_SRCDIR}/Lib/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_SRCDIR}/Passes/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_SRCDIR}/Scripts/Makefile
	$(COPY) forward_gnu_makefile ${PIPS_SRCDIR}/Runtime/Makefile
	$(COPY) forward_gnu_makefile ${NEWGEN_DEVEDIR}/Makefile
	$(COPY) forward_gnu_makefile ${NEWGEN_SRCDIR}/Makefile
	$(COPY) forward_gnu_makefile ${LINEAR_DEVEDIR}/Makefile
	$(COPY) forward_gnu_makefile ${LINEAR_SRCDIR}/Makefile

# that is all
#
