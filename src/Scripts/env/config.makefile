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
	chmod a-w pipsrc.sh

pipsrc.csh: pipsrc.sh
	$(SHELL) make-pipsrc.csh 
	chmod a-w pipsrc.csh

clean:
	$(RM) pipsrc.sh pipsrc.csh

# that is all
#
