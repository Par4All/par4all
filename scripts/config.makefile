# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/10 18:21:09 $, 

all: .runable
recompile: .quick-install
clean:
	$(RM) *~

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
