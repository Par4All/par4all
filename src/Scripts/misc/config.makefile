#
# $RCSfile: config.makefile,v $ for misc
#

INSTALL_SHR=	pips-deal-with-include \
		pips-process-module \
		pips-split \
		pips-unsplit-workspace 

UTL_SCRIPTS = 	check_os \
		filter_verbatim \
		job-make \
		job-receive \
		trunc \
		unjustify

FILES =		trunc.awk \
		extract-doc.awk \
		accent.sed

INSTALL_UTL=	$(UTL_SCRIPTS) $(FILES)
SCRIPTS=	$(INSTALL_SHR) $(UTL_SCRIPTS)
SOURCES=	$(SCRIPTS) $(FILES)

all: .runable

# that is all
#
