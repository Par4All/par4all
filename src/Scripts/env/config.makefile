#
# $RCSfile: config.makefile,v $ for env
#

SCRIPTS =	

SFILES =	pipsrc.ref \
		make-pipsrc.sh \
		make-pipsrc.csh

RFILES = 	pipsrc.sh \
		pipsrc.csh

all: pipsrc.sh pipsrc.csh

pipsrc.sh: pipsrc.ref
	$(SHELL) make-pipsrc.sh 

pipsrc.csh: pipsrc.sh
	$(SHELL) make-pipsrc.csh 

# that is all
#
