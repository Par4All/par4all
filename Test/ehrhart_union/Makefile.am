#
# $Id: Makefile.am,v 1.1 2004/01/31 22:17:46 verdoolaege Exp $
#

OBJ_DIR = $(top_builddir)

EHRHART_UNION =  \
	ue1.in \
	ue2.in

EXTRA_DIST = $(EHRHART_UNION) \
	$(EHRHART_UNION:%.in=%.out) 

TEST=$(EHRHART_UNION:%.in=%.tested) 

check:
	@failedtest=0; \
	for x in $(EHRHART_UNION) ; do \
		echo "Verify file $$x" ; \
		$(OBJ_DIR)/ehrhart_union < $(srcdir)/$$x > xyz;\
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
	        echo "Test ehrhart_union works correctly"; \
        fi
