#
# $RCSfile: config.makefile,v $ for env
#

SCRIPTS =	

SFILES =	pipsrc.ref \
		make-pipsrc.sh \
		make-pipsrc.csh \
		Xressources \
		xinitrc

RFILES = 	pipsrc.sh \
		pipsrc.csh

all: pipsrc.sh pipsrc.csh

pipsrc.sh: pipsrc.ref
	$(RM) pipsrc.sh
	$(SHELL) make-pipsrc.sh 
	chmod a-w pipsrc.sh

pipsrc.csh: pipsrc.sh
	$(RM) pipsrc.csh
	$(SHELL) make-pipsrc.csh 
	chmod a-w pipsrc.csh

clean:
	$(RM) pipsrc.sh pipsrc.csh

# that is all
#
