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

FILES=		forward_gnu_makefile
COPY=	cp -f

forward: 
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
