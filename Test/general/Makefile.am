#
# $Id: Makefile.am,v 1.1 2004/01/31 22:17:46 verdoolaege Exp $
#

OBJ_DIR = $(top_builddir)

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

EXTRA_DIST = $(GENERAL) \
	$(GENERAL:%.in=%.out) 

#	test.in

check:
	@failedtest=0; \
	for x in $(GENERAL) ; do \
		echo "Verify file $$x"; \
		$(OBJ_DIR)/polytest < $(srcdir)/$$x > xyz;\
		diff -w xyz $(srcdir)/`basename $$x .in`.out ; \
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
