#
# $Id: Makefile.am,v 1.4 2004/07/25 12:52:17 verdoolaege Exp $
#

OBJ_DIR = $(top_builddir)

EHRHART =  \
	8hedron.in \
	alex11.in \
	alex1.in \
	c3.in\
	c4.in\
	e0.in\
	e10.in\
	e11.in\
	e12.in\
	e13.in\
	e14.in\
	e15.in\
	e16.in\
	e1.in\
	e2b.in\
	e2.in\
	e3.in\
	e4.in\
	e5.in\
	e6.in\
	e7.in\
	e8.in\
	e9.in\
	esum.in \
	g12.in\
	g13.in\
	g14.in\
	g1.in\
	g2.in\
	g4.in\
	g5.in\
	isnm2.in\
	t1.in\
	t2.in\
	t31.in\
	t3.in\
	t4.in\
	t12.in\
	tpar.in

EXTRA_DIST = $(EHRHART) \
	$(EHRHART:%.in=%.out) 

# These do not pass in 32 bits mode (integer overflow): isnm isnm2 e16
if BITS32
  OVER = isnm.in isnm2.in e16.in esum.in t12.in
endif

EHRHART_TESTS = $(filter-out $(LONG_EHRHART_TESTS) $(OVER), $(EHRHART))

check:
	@failedtest=0; \
	for x in $(EHRHART_TESTS) ; do \
		echo "Verify file $$x" ; \
		$(OBJ_DIR)/testehrhart < $(srcdir)/$$x >xyz; \
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
	        echo "Test ehrhart works correctly"; \
        fi

longtest:
	@for x in $(EHRHART_TESTS) ; do \
		echo "Verify Ehrhart for $$x"; \
		$(OBJ_DIR)/verif_ehrhart < $(srcdir)/$$x ;\
	done ;
