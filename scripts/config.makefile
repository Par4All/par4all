# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/02 19:51:13 $, 

all: .runable

SCRIPTS=\
	make_all_specs\
	newgen

SOURCES= $(SCRIPTS) config.makefile

INSTALL_SHR=	$(SCRIPTS)

.runable: $(SCRIPTS)
	-[ "$(SCRIPTS)" ] && chmod a+x $(SCRIPTS)
	touch .runable

.quick-install: .runable $(RFILES)

#
