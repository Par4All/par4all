#
# $Id: makefile,v 1.2 2002/10/15 10:08:52 kienhuis Exp $
#

ROOT  = ../..

GENERAL =  \
	convex1.in \
	convex2.in \
	gauss1.in \
	herve.in \
	simpl1.in \
	simpl2.in \
	simpl3.in \
	simpl4.in \
	simpl5.in \
	simpl6.in \
	simpl7.in \
	test1.in

#	test.in

test:
	@failedtest=0; \
	for x in $(GENERAL) ; do \
		echo "Verify file $$x"; \
		$(ROOT)/bin/polytest.exe < $$x > xyz;\
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
	        echo "Test polytest works correctly"; \
        fi


# Include the shared compilation rules
include $(ROOT)/vars.mk
include $(ROOT)/default_functions.mk
