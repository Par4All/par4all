#
# $RCSfile: config.makefile,v $ for pips drivers
# 
# version $Revision$ ($Date: 1996/08/13 17:15:50 $, )
#

SCRIPTS =	Display \
		Pips \
		Init \
		Perform \
		Delete \
		Build \
		Select \
		Interchange

SOURCES =	$(SCRIPTS) config.makefile

INSTALL_SHR=	$(SCRIPTS)

# that is all
#
