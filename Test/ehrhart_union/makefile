#
# $Id: makefile,v 1.3 2002/10/15 10:08:52 kienhuis Exp $
#

ROOT  = ../..

EHRHART_UNION =  \
	ue1.in \
	ue2.in

TEST=$(EHRHART_UNION:%.in=%.tested) 

test:
	@failedtest=0; \
	for x in $(EHRHART_UNION) ; do \
		echo "Verify file $$x" ; \
		$(ROOT)/bin/verif_ehrhart.exe < $$x > xyz;\
		diff -w xyz `basename $$x .in`.out ; \
		result=$$?; \
		if [ "$$result" -eq "1" ]; then \
			echo "Error: $$x is not the same"; \
			failedtest=`expr $$failedtest + 1`; \
		else \
			echo "$$x passed"; \
	        fi; \
	done ; \
        if [ $$failedtest != 0 ]; then \
		echo "$$failedtest tests failed"; \
	else \
	        echo "Test ehrhart_union works correctly"; \
        fi

# Include the shared compilation rules
include $(ROOT)/vars.mk
include $(ROOT)/default_functions.mk
